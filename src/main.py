import sys
import os
from typing import Any, Iterator

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

os.environ.setdefault("GDAL_NUM_THREADS", "1")
os.environ.setdefault("CPL_WORKER_THREADS", "1")

import math
os.environ.setdefault("GDAL_CACHEMAX", "16")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import argparse
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
import pickle
from affine import Affine  # type: ignore[import-untyped]

import numpy as np
import rasterio  # type: ignore[import-untyped]
from rasterio.windows import Window  # type: ignore[import-untyped]

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PolyModel import PolyMahalanobis
from cache import NullCache, SharedDirectMappedCache, Blake2bHashStrategy, XxHashStrategy
from signatures import make_cache_key_values, prepare_for_model
from audit.stats import TileStats, PipelineStats
from audit.writer import write_audit_json, write_tile_csv


class Config:
    DEFAULT_ORDER: int = 3
    DEFAULT_EXP_VALUE: float = -1.0
    DEFAULT_TILE_SIZE: int = 1024

    @staticmethod
    def effective_cpu_count() -> int:
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            return os.cpu_count() or 1


try:
    from osgeo import gdal  # type: ignore[import-untyped]
    GDAL_AVAILABLE = True
except Exception:
    gdal = None  # type: ignore[assignment]
    GDAL_AVAILABLE = False


@dataclass(frozen=True)
class Tile:
    tile_id: int
    row: int
    col: int
    h: int
    w: int


_SRC_RASTERIO_READER: Any | None = None
_POLY_MODEL: Any | None = None
_EXP_VALUE: float | None = None
_CACHE: Any | None = None
_CACHE_KEY_MODE: str | None = None
_CACHE_KEY_DECIMALS: int | None = None
_N_BANDS: int | None = None
_BAND_IDX: list[int] | None = None

_SUPPORTED_DTYPES: set[str] = {"uint8", "uint16", "float32", "float64"}

_HASH_STRATEGIES: dict[str, type] = {
    "blake2b": Blake2bHashStrategy,
    "xxhash": XxHashStrategy,
}


def validate_and_resolve_dtype(src: Any, idx: list[int]) -> tuple[np.dtype, bool]:
    """
    Valida que todas as bandas usadas têm o mesmo dtype, e que o dtype é suportado.

    Args:
        src: dataset rasterio aberto.
        idx: lista de índices de banda (1-indexed) que serão lidos.

    Returns:
        (effective_dtype, needs_float64_cast)
        effective_dtype: np.uint8, np.uint16 ou np.float32 (float64 é tratado como float32).
        needs_float64_cast: True se o dtype original era float64 (cast obrigatório por tile).

    Raises:
        ValueError: se as bandas tiverem dtypes diferentes entre si, ou se o dtype
                    não estiver em _SUPPORTED_DTYPES.
    """
    band_dtypes: set[str] = {src.dtypes[i - 1] for i in idx}
    if len(band_dtypes) > 1:
        raise ValueError(
            f"Bandas com dtypes diferentes não são suportadas: {band_dtypes}"
        )
    dtype_str: str = band_dtypes.pop()
    if dtype_str not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"dtype de entrada não suportado: {dtype_str!r}. "
            f"Suportados: {sorted(_SUPPORTED_DTYPES)}"
        )
    if dtype_str == "float64":
        return np.float32, True
    return np.dtype(dtype_str), False


def resolve_band_indices(src: Any) -> list[int]:
    """
    Returns 1-indexed band indices, excluding any band with ColorInterp.alpha.

    Args:
        src: rasterio DatasetReader aberto.

    Returns:
        Lista de índices de banda (1-indexed) sem bandas alpha.
    """
    return [
        i for i in range(1, src.count + 1)
        if src.colorinterp[i - 1] != rasterio.enums.ColorInterp.alpha
    ]


def distances_to_uint8(distances: np.ndarray, exp_value: float) -> np.ndarray:
    """
    Converte distâncias float32 (último nível de PolyModel.evaluate) para uint8 (0-255).

    REGRA FIXA — preserva o comportamento legado exatamente. Esta é a ÚNICA função
    do projeto que faz essa conversão. Nunca duplicar esta fórmula em outro lugar.
    PolyModel.py NUNCA faz essa conversão — ele só retorna distâncias float32 cruas.

    Fórmula (idêntica ao main.py legado):
        sim = clip(exp(exp_value * distance), 0, 1)
        value = round(255 - sim * 255)

    Args:
        distances: array (N,) float32, distância do último nível.
        exp_value: expoente negativo, vem do argumento CLI --exp.

    Returns:
        array (N,) uint8.
    """
    sim = np.clip(np.exp(exp_value * distances), 0.0, 1.0)
    return np.rint(255.0 - sim * 255.0).astype(np.uint8)


def build_cache(
    shared_cache_mb: float,
    n_bands: int,
    sig_dtype: np.dtype,
    hash_strategy_name: str,
) -> NullCache | SharedDirectMappedCache:
    """Build cache instance. Returns NullCache when shared_cache_mb <= 0, SharedDirectMappedCache otherwise."""
    if shared_cache_mb <= 0:
        return NullCache()
    strategy = _HASH_STRATEGIES[hash_strategy_name]()
    return SharedDirectMappedCache(
        size_mb=shared_cache_mb,
        n_bands=n_bands,
        sig_dtype=sig_dtype,
        hash_strategy=strategy,
    )


def setup_logging(log_path: str) -> None:
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    handlers = [
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def add_overviews_gdal(output_file: str) -> None:
    """Adiciona overviews e renomeia banda se GDAL estiver disponível."""
    if not GDAL_AVAILABLE:
        logging.warning("GDAL not found. Skipping overviews.")
        return

    try:
        ds = gdal.Open(output_file, gdal.GA_Update)
        if ds:
            if ds.RasterCount >= 1:
                band1 = ds.GetRasterBand(1)
                band1.SetDescription("polyhealth")
                band1.SetMetadataItem("DESCRIPTION", "polyhealth")

            logging.info("Building overviews...")
            ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32, 64, 128])
            ds.FlushCache()
            ds = None
            logging.info("Overviews built successfully.")
    except Exception as e:
        logging.warning(f"GDAL Error: {e}")


def positive_decimals(value: str) -> int:
    """Valida que o valor seja um inteiro positivo (>=1) para --cache-key-decimals."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--cache-key-decimals must be >= 1")
    return parsed


def get_arguments() -> argparse.Namespace:
    """
    Define e faz o parsing dos argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de Processamento de Imagens com Mahalanobis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input_path", type=str, help="Caminho da imagem de entrada")
    parser.add_argument("output_path", type=str, help="Caminho da imagem de saída (resultado)")
    parser.add_argument("samples_path", type=str, help="Arquivo de amostras (.txt)")
    parser.add_argument("log_path", type=str, help="Arquivo de log")

    parser.add_argument("--order", type=int, default=Config.DEFAULT_ORDER,
                        help="Ordem do polinômio")
    parser.add_argument("--exp", type=float, default=Config.DEFAULT_EXP_VALUE, dest="exp_value",
                        help="Expoente negativo para o cálculo")

    parser.add_argument("--tile-size", type=int, default=Config.DEFAULT_TILE_SIZE,
                        help="Tamanho do tile em pixels")

    default_workers = max(1, Config.effective_cpu_count())
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="Número de threads/workers paralelos")

    parser.add_argument("--overviews", action="store_true",
                        help="Gerar pirâmides (overviews) no final")
    parser.add_argument("--statistics", action="store_true",
                        help="Calcular estatísticas do heatmap no final")

    parser.add_argument("--alpha", action="store_true",
                        help="Incluir canal Alpha no resultado")

    parser.add_argument("--shared-cache-mb", type=float, default=0.0,
                        help="Tamanho máximo do cache compartilhado em MB. 0 desliga o cache.")
    parser.add_argument("--cache-key-mode", choices=["exact", "round"], default="exact",
                        help="exact preserva o resultado; round é aproximado.")
    parser.add_argument("--cache-key-decimals", type=positive_decimals, default=None,
                        help="Casas decimais para --cache-key-mode round. Só válido com round. Mínimo 1.")
    parser.add_argument("--hash-strategy", choices=["blake2b", "xxhash"], default="blake2b",
                        help="Estratégia de hash do cache compartilhado.")
    parser.add_argument("--audit-json", type=str, default=None,
                        help="Caminho para salvar o JSON de auditoria. Se omitido, não gera.")
    parser.add_argument("--audit-tile-csv", type=str, default=None,
                        help="Caminho para salvar o CSV por tile. Se omitido, não gera.")

    args = parser.parse_args()

    if args.cache_key_mode == "round" and args.cache_key_decimals is None:
        parser.error("--cache-key-decimals é obrigatório quando --cache-key-mode round")

    if args.cache_key_mode == "exact" and args.cache_key_decimals is not None:
        parser.error("--cache-key-decimals só é válido com --cache-key-mode round")

    if args.shared_cache_mb < 0:
        parser.error("--shared-cache-mb não pode ser negativo")

    return args


def init_worker(
    in_path: str,
    model_bytes: bytes,
    exp_value: float,
    n_bands: int,
    band_idx: list[int],
    cache_key_mode: str,
    cache_key_decimals: int | None,
    cache_enabled: bool,
    cache_attach_info: dict[str, object],
    hash_strategy_name: str,
) -> None:
    """Initialize a worker process: open rasterio reader, deserialize model, attach to shared cache."""
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
        threadpool_limits(1)
    except Exception:
        pass

    global _SRC_RASTERIO_READER, _POLY_MODEL, _EXP_VALUE, _CACHE
    global _CACHE_KEY_MODE, _CACHE_KEY_DECIMALS, _N_BANDS, _BAND_IDX

    _SRC_RASTERIO_READER = rasterio.open(in_path)
    _POLY_MODEL = pickle.loads(model_bytes)
    _EXP_VALUE = float(exp_value)
    _N_BANDS = n_bands
    _BAND_IDX = band_idx
    _CACHE_KEY_MODE = cache_key_mode
    _CACHE_KEY_DECIMALS = cache_key_decimals

    if not cache_enabled:
        _CACHE = NullCache()
    else:
        strategy = _HASH_STRATEGIES[hash_strategy_name]()
        _CACHE = SharedDirectMappedCache.attach(
            shm_name=cache_attach_info["shm_name"],
            n_slots=cache_attach_info["n_slots"],
            n_bands=cache_attach_info["n_bands"],
            sig_dtype=cache_attach_info["sig_dtype"],
            slot_size=cache_attach_info["slot_size"],
            effective_bytes=cache_attach_info["effective_bytes"],
            locks=cache_attach_info["locks"],
            hash_strategy=strategy,
        )


def process_tile(tile: Tile) -> tuple[Tile, np.ndarray, TileStats]:
    global _SRC_RASTERIO_READER, _POLY_MODEL, _EXP_VALUE, _CACHE
    global _CACHE_KEY_MODE, _CACHE_KEY_DECIMALS, _N_BANDS, _BAND_IDX

    t_start = time.perf_counter()
    window = Window(tile.col, tile.row, tile.w, tile.h)

    # --- 1. leitura do tile, SEM inversão de bandas ---
    t0 = time.perf_counter()
    tile_bhw = _SRC_RASTERIO_READER.read(indexes=_BAND_IDX, window=window)
    if tile_bhw.dtype == np.float64:
        tile_bhw = tile_bhw.astype(np.float32, copy=False)
    tile_hwb = np.moveaxis(tile_bhw, 0, -1)  # (h, w, n_bands)

    mask = _SRC_RASTERIO_READER.dataset_mask(window=window)  # 255=válido, 0=inválido
    t_read = time.perf_counter() - t0

    h, w = tile.h, tile.w
    out = np.zeros((h, w), dtype=np.uint8)  # D8: inválidos ficam 0

    flat_mask = mask.reshape(-1) > 0
    valid_pixels = tile_hwb.reshape(-1, _N_BANDS)[flat_mask]

    t_stats = TileStats(
        tile_id=tile.tile_id,
        row_off=tile.row,
        col_off=tile.col,
        height=h,
        width=w,
        total_pixels=h * w,
        valid_pixels=int(valid_pixels.shape[0]),
        unique_signatures=0,
        time_read_tile=t_read,
    )

    if valid_pixels.shape[0] == 0:
        t_stats.time_total = time.perf_counter() - t_start
        return tile, out, t_stats

    # --- 2. construção das chaves de cache ---
    key_values = make_cache_key_values(
        valid_pixels, mode=_CACHE_KEY_MODE, decimals=_CACHE_KEY_DECIMALS
    )

    # --- 3. dedup intra-tile ---
    t0 = time.perf_counter()
    unique_keys, inverse = np.unique(key_values, axis=0, return_inverse=True)
    t_stats.unique_signatures = unique_keys.shape[0]
    t_stats.time_unique = time.perf_counter() - t0

    # --- 4. lookup no cache global ---
    t0 = time.perf_counter()
    cached_values, hit_mask, lookup_stats = _CACHE.lookup_many(unique_keys)
    t_stats.time_cache_lookup = time.perf_counter() - t0
    t_stats.cache_lookups = lookup_stats.n_queries
    t_stats.cache_hits = lookup_stats.n_hits
    t_stats.cache_misses_empty = lookup_stats.n_misses_empty
    t_stats.cache_misses_collision = lookup_stats.n_misses_collision

    result_for_unique = np.empty(unique_keys.shape[0], dtype=np.uint8)
    result_for_unique[hit_mask] = cached_values[hit_mask]

    # --- 5. PolyModel só nos misses ---
    miss_keys = unique_keys[~hit_mask]
    t0 = time.perf_counter()
    if miss_keys.shape[0] > 0:
        model_input = prepare_for_model(miss_keys)
        miss_distances = _POLY_MODEL.evaluate(model_input)[:, -1]
        miss_values = distances_to_uint8(miss_distances, _EXP_VALUE)
        result_for_unique[~hit_mask] = miss_values
    t_stats.time_model_eval = time.perf_counter() - t0

    # --- 6. insere os misses no cache global ---
    t0 = time.perf_counter()
    if miss_keys.shape[0] > 0:
        insert_stats = _CACHE.insert_many(miss_keys, miss_values)
        t_stats.cache_inserted = insert_stats.n_inserted
        t_stats.cache_insert_skipped_collision = insert_stats.n_skipped_collision
    t_stats.time_cache_insert = time.perf_counter() - t0

    # --- 7. reconstrução ---
    t0 = time.perf_counter()
    out_valid = result_for_unique[inverse]
    out_flat = out.reshape(-1)
    out_flat[flat_mask] = out_valid
    t_stats.time_reconstruct = time.perf_counter() - t0

    t_stats.time_total = time.perf_counter() - t_start
    return tile, out, t_stats


def iter_tiles(height: int, width: int, tile_h: int, tile_w: int) -> Iterator[Tile]:
    tile_id = 0
    for row in range(0, height, tile_h):
        h = min(tile_h, height - row)
        for col in range(0, width, tile_w):
            w = min(tile_w, width - col)
            yield Tile(tile_id=tile_id, row=row, col=col, h=h, w=w)
            tile_id += 1


def main(
    in_path: str,
    sample_file: str,
    order: int,
    exp_value: float,
    out_path: str,
    n_workers: int = 20,
    tile_h: int = 1024,
    tile_w: int = 1024,
    add_alpha: bool = False,
    shared_cache_mb: float = 0.0,
    cache_key_mode: str = "exact",
    cache_key_decimals: int | None = None,
    hash_strategy_name: str = "blake2b",
    audit_json_path: str | None = None,
    audit_tile_csv_path: str | None = None,
) -> None:
    poly_model = PolyMahalanobis(sample_file, num_levels=order)
    poly_model.makeSpace()
    poly_model.samples = None

    model_bytes = pickle.dumps(poly_model, protocol=pickle.HIGHEST_PROTOCOL)
    n_bands = poly_model.n_bands

    with rasterio.open(in_path) as src:
        band_idx = resolve_band_indices(src)

        if len(band_idx) != n_bands:
            raise ValueError(
                f"Mismatch: {n_bands} colunas no samples.txt, mas {len(band_idx)} bandas "
                f"espectrais no TIFF (excluindo alpha). "
                f"O arquivo de amostras deve ter a mesma quantidade de colunas que bandas "
                f"não-alpha no TIFF."
            )

        effective_dtype, needs_float64_cast = validate_and_resolve_dtype(src, band_idx)

        height, width = src.height, src.width

        transform = src.transform
        if transform is None or transform.e > 0:
            transform = Affine.translation(0, height) * Affine.scale(1, -1)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            count=2 if add_alpha else 1,
            dtype="uint8",
            tiled=True,
            blockxsize=max(16, math.ceil(tile_w / 16) * 16),
            blockysize=max(16, math.ceil(tile_h / 16) * 16),
            compress=None,
            BIGTIFF="YES",
            transform=transform,
            crs=src.crs,
        )

        # Build cache in main process
        cache = build_cache(shared_cache_mb, n_bands, effective_dtype, hash_strategy_name)

        # Pipeline stats
        pipeline_stats = PipelineStats(
            n_bands=n_bands,
            input_dtype=str(effective_dtype),
            tile_size=tile_h,
            workers=n_workers,
            cache_enabled=cache.enabled,
            cache_backend=type(cache).__name__,
            hash_strategy=hash_strategy_name if cache.enabled else "",
            cache_key_mode=cache_key_mode,
            cache_key_decimals=cache_key_decimals,
            requested_cache_mb=shared_cache_mb,
        )
        if cache.enabled:
            info = cache.info()
            pipeline_stats.effective_cache_mb = info["effective_cache_mb"]
            pipeline_stats.effective_cache_bytes = info["effective_cache_bytes"]
            pipeline_stats.slot_size_bytes = info["slot_size_bytes"]
            pipeline_stats.effective_slot_count = info["effective_slot_count"]

        # Prepare cache attach info for workers
        cache_attach_info: dict[str, object] = {}
        if cache.enabled:
            cache_attach_info = {
                "shm_name": cache.shm_name,
                "n_slots": cache.n_slots,
                "n_bands": cache.n_bands,
                "sig_dtype": str(cache.sig_dtype),
                "slot_size": cache.slot_size,
                "effective_bytes": cache.effective_bytes,
                "locks": cache.locks,
            }

        ctx = mp.get_context("spawn")

        with rasterio.open(out_path, "w", **profile) as dst:
            tiles = list(iter_tiles(height, width, tile_h, tile_w))

            with ctx.Pool(
                processes=n_workers,
                initializer=init_worker,
                initargs=(
                    in_path,
                    model_bytes,
                    exp_value,
                    n_bands,
                    band_idx,
                    cache_key_mode,
                    cache_key_decimals,
                    cache.enabled,
                    cache_attach_info,
                    hash_strategy_name,
                ),
                maxtasksperchild=64,
            ) as pool:
                for tile, out, tstats in pool.imap_unordered(
                    process_tile, tiles, chunksize=1
                ):
                    t_write_start = time.perf_counter()
                    dst.write(out, 1, window=Window(tile.col, tile.row, tile.w, tile.h))
                    if add_alpha:
                        mask = src.dataset_mask(
                            window=Window(tile.col, tile.row, tile.w, tile.h)
                        ).astype(np.uint8)
                        dst.write(mask, 2, window=Window(tile.col, tile.row, tile.w, tile.h))
                    tstats.time_write_tile = time.perf_counter() - t_write_start
                    pipeline_stats.accumulate(tstats)

        # Set alpha color interpretation if needed
        if add_alpha:
            with rasterio.open(out_path, "r+") as dst:
                dst.colorinterp = (
                    rasterio.enums.ColorInterp.gray,
                    rasterio.enums.ColorInterp.alpha,
                )

        if cache.enabled:
            pipeline_stats.cache_occupancy_estimated = cache.occupancy()
            cache.close()

    if audit_json_path:
        write_audit_json(pipeline_stats, audit_json_path)
    if audit_tile_csv_path:
        write_tile_csv(pipeline_stats, audit_tile_csv_path)


if __name__ == "__main__":
    args = get_arguments()
    setup_logging(args.log_path)

    tile_size = int(args.tile_size)

    logging.info("Iniciando processamento")
    main(
        args.input_path,
        args.samples_path,
        args.order,
        args.exp_value,
        args.output_path,
        n_workers=args.workers,
        tile_h=tile_size,
        tile_w=tile_size,
        add_alpha=args.alpha,
        shared_cache_mb=args.shared_cache_mb,
        cache_key_mode=args.cache_key_mode,
        cache_key_decimals=args.cache_key_decimals,
        hash_strategy_name=args.hash_strategy,
        audit_json_path=args.audit_json,
        audit_tile_csv_path=args.audit_tile_csv,
    )
    logging.info("Processamento finalizado")

    if args.statistics:
        logging.info("Calculando estatísticas...")
        with rasterio.open(args.output_path) as src_out:
            scale = max(1, max(src_out.width, src_out.height) // 2048)
            data = src_out.read(
                1,
                out_shape=(src_out.height // scale, src_out.width // scale),
                resampling=rasterio.enums.Resampling.nearest,
            )

            if args.alpha:
                mask = src_out.read(
                    2,
                    out_shape=(src_out.height // scale, src_out.width // scale),
                    resampling=rasterio.enums.Resampling.nearest,
                )
                data = data[mask > 0]

        severe_px_count = np.sum(data < int(255 * 0.05))
        severe_percent = severe_px_count / data.size * 100

        moderate_px_count = np.sum((data >= int(255 * 0.05)) & (data < int(255 * 0.25)))
        moderate_percent = moderate_px_count / data.size * 100

        intermediate_px_count = np.sum((data >= int(255 * 0.25)) & (data < int(255 * 0.50)))
        intermediate_percent = intermediate_px_count / data.size * 100

        good_px_count = np.sum((data >= int(255 * 0.50)) & (data < int(255 * 0.75)))
        good_percent = good_px_count / data.size * 100

        excellent_px_count = np.sum(data >= int(255 * 0.75))
        excellent_percent = excellent_px_count / data.size * 100

        logging.info(
            f"Severe: {severe_percent:.2f}%, Moderate: {moderate_percent:.2f}%, "
            f"Intermediate: {intermediate_percent:.2f}%, Good: {good_percent:.2f}%, "
            f"Excellent: {excellent_percent:.2f}%"
        )

        with rasterio.open(args.output_path, "r+") as dst:
            dst.update_tags(
                excellent_percent=f"{excellent_percent:.2f}",
                good_percent=f"{good_percent:.2f}",
                intermediate_percent=f"{intermediate_percent:.2f}",
                moderate_percent=f"{moderate_percent:.2f}",
                severe_percent=f"{severe_percent:.2f}",
            )

    if args.overviews:
        logging.info("Gerando overviews...")
        add_overviews_gdal(args.output_path)
