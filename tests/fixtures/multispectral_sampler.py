#!/usr/bin/env python3
"""
Coletor interativo de samples para rasters multiespectrais com N bandas.

Funcionalidades:
- Abre um raster com rasterio.
- Mostra a lista de bandas não-alpha e permite escolher quais bandas entram em R, G e B.
- Exibe uma prévia RGB normalizada só para visualização.
- Permite desenhar polígonos no mapa.
- Para cada polígono, coleta os pixels internos em todas as bandas não-alpha do raster.
- Salva os samples em TXT, uma linha por pixel, colunas na ordem original das bandas, ignorando color=alpha.

Uso básico:
    python multispectral_sampler.py imagem.tif

Uso sem janela de seleção de bandas:
    python multispectral_sampler.py imagem.tif --rgb 4,3,2 --output samples.txt

Controles dentro da janela do mapa:
    - Clique para criar os vértices do polígono.
    - Pressione Enter para fechar/coletar o polígono.
    - Tecla s: salva samples.txt.
    - Tecla u: desfaz o último polígono coletado.
    - Tecla c: limpa todos os samples desta sessão.
    - Tecla q: fecha a janela.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Button, PolygonSelector
from rasterio.enums import Resampling
from rasterio.windows import Window


DEFAULT_OUTPUT = "samples.txt"
DEFAULT_PRECISION = "%.18e"
DEFAULT_TILE_SIZE = 512


def is_alpha_band(src: rasterio.io.DatasetReader, band_idx: int) -> bool:
    """Retorna True quando a banda está marcada como alpha no color interpretation."""
    if not src.colorinterp:
        return False
    return src.colorinterp[band_idx - 1].name.lower() == "alpha"


def non_alpha_band_indexes(src: rasterio.io.DatasetReader) -> list[int]:
    """Bandas 1-based do raster, em ordem original, excluindo color=alpha."""
    return [band_idx for band_idx in src.indexes if not is_alpha_band(src, band_idx)]


def parse_rgb(value: str, allowed_band_indexes: Sequence[int]) -> tuple[int, int, int]:
    """Lê uma string como '4,3,2' e retorna bandas 1-based para R, G e B."""
    parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use exatamente 3 bandas no formato R,G,B. Exemplo: 4,3,2")

    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("As bandas RGB precisam ser números inteiros. Exemplo: 4,3,2") from exc

    allowed = set(allowed_band_indexes)
    invalid = [b for b in rgb if b not in allowed]
    if invalid:
        allowed_text = ",".join(str(b) for b in allowed_band_indexes)
        raise argparse.ArgumentTypeError(
            f"Bandas inválidas para RGB: {invalid}. Bandas disponíveis, ignorando alpha: {allowed_text}"
        )

    return rgb  # type: ignore[return-value]


def band_label(src: rasterio.io.DatasetReader, band_idx: int) -> str:
    desc = src.descriptions[band_idx - 1] if src.descriptions else None
    dtype = src.dtypes[band_idx - 1] if src.dtypes else "?"
    nodata = src.nodatavals[band_idx - 1] if src.nodatavals else None
    color = src.colorinterp[band_idx - 1].name if src.colorinterp else "undefined"

    pieces = [f"Banda {band_idx}"]
    if desc:
        pieces.append(str(desc))
    pieces.append(f"dtype={dtype}")
    pieces.append(f"color={color}")
    if nodata is not None:
        pieces.append(f"nodata={nodata}")
    return " | ".join(pieces)


def band_labels(src: rasterio.io.DatasetReader, band_indexes: Sequence[int] | None = None) -> list[str]:
    selected = list(band_indexes) if band_indexes is not None else list(src.indexes)
    return [band_label(src, band_idx) for band_idx in selected]


def choose_raster_path_by_dialog() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Selecione o raster multiespectral",
        filetypes=(
            ("GeoTIFF / TIFF", "*.tif *.tiff"),
            ("Todos os arquivos", "*.*"),
        ),
    )
    root.destroy()
    if not filename:
        return None
    return Path(filename)


def choose_rgb_bands_by_dialog(
    labels: Sequence[str],
    band_indexes: Sequence[int],
    default_rgb: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception:
        return None

    result: dict[str, tuple[int, int, int] | None] = {"value": None}
    values = list(labels)
    band_lookup = dict(zip(values, band_indexes))

    root = tk.Tk()
    root.title("Mapeamento RGB das bandas")
    root.resizable(False, False)

    main = ttk.Frame(root, padding=12)
    main.grid(row=0, column=0, sticky="nsew")

    ttk.Label(main, text="Escolha quais bandas do raster serão exibidas como R, G e B.").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
    )
    ttk.Label(main, text="A coleta salva TODAS as bandas, não só estas três.").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(0, 12)
    )

    variables: list[tk.StringVar] = []
    channel_names = ["R", "G", "B"]
    for row_offset, (channel_name, default_band) in enumerate(zip(channel_names, default_rgb), start=2):
        ttk.Label(main, text=f"Canal {channel_name}:").grid(row=row_offset, column=0, sticky="w", padx=(0, 8), pady=4)
        try:
            default_value = values[list(band_indexes).index(default_band)]
        except ValueError:
            default_value = values[0]
        var = tk.StringVar(value=default_value)
        combo = ttk.Combobox(main, textvariable=var, values=values, width=90, state="readonly")
        combo.grid(row=row_offset, column=1, sticky="ew", pady=4)
        variables.append(var)

    def on_ok() -> None:
        selected: list[int] = []
        for var in variables:
            try:
                selected.append(band_lookup[var.get()])
            except KeyError:
                messagebox.showerror("Seleção inválida", "Selecione uma banda válida para R, G e B.")
                return
        result["value"] = (selected[0], selected[1], selected[2])
        root.destroy()

    def on_cancel() -> None:
        result["value"] = None
        root.destroy()

    button_frame = ttk.Frame(main)
    button_frame.grid(row=5, column=0, columnspan=2, sticky="e", pady=(12, 0))
    ttk.Button(button_frame, text="Cancelar", command=on_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(button_frame, text="Abrir mapa", command=on_ok).grid(row=0, column=1)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return result["value"]


def default_rgb_from_colorinterp(src: rasterio.io.DatasetReader, allowed_band_indexes: Sequence[int]) -> tuple[int, int, int]:
    """Tenta usar bandas marcadas como red/green/blue. Caso contrário usa as 3 primeiras não-alpha."""
    color_by_band = {band_idx: src.colorinterp[band_idx - 1].name.lower() for band_idx in allowed_band_indexes}
    wanted = ["red", "green", "blue"]
    found: list[int] = []
    for wanted_name in wanted:
        for band_idx in allowed_band_indexes:
            if color_by_band.get(band_idx) == wanted_name:
                found.append(band_idx)
                break
    if len(found) == 3:
        return found[0], found[1], found[2]

    if len(allowed_band_indexes) < 3:
        raise ValueError(
            "O raster precisa ter pelo menos 3 bandas não-alpha para criar uma visualização RGB."
        )
    return allowed_band_indexes[0], allowed_band_indexes[1], allowed_band_indexes[2]


def normalize_for_display(channel: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    """Normaliza um canal float/int para 0..1 usando percentis robustos."""
    arr = np.ma.filled(channel, np.nan).astype(np.float64, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)

    values = arr[finite]
    lo, hi = np.nanpercentile(values, [2, 98])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))

    if hi <= lo:
        out = np.zeros(arr.shape, dtype=np.float32)
        out[finite] = 0.5
        return out

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def read_rgb_preview(
    src: rasterio.io.DatasetReader,
    rgb_bands: tuple[int, int, int],
    max_preview_size: int,
) -> np.ndarray:
    """Lê só uma prévia RGB. A coleta continua sendo feita no raster original."""
    scale = max(src.width / max_preview_size, src.height / max_preview_size, 1.0)
    preview_width = max(1, int(round(src.width / scale)))
    preview_height = max(1, int(round(src.height / scale)))

    data = src.read(
        list(rgb_bands),
        out_shape=(3, preview_height, preview_width),
        resampling=Resampling.bilinear,
        masked=True,
    )

    rgb = np.dstack([normalize_for_display(data[i]) for i in range(3)])
    return rgb


class MultispectralSampler:
    def __init__(
        self,
        image_path: Path,
        rgb_bands: tuple[int, int, int],
        sample_band_indexes: Sequence[int],
        output_path: Path,
        precision: str = DEFAULT_PRECISION,
        max_preview_size: int = 1600,
        tile_size: int = DEFAULT_TILE_SIZE,
        autosave: bool = True,
        keep_invalid: bool = False,
    ) -> None:
        self.image_path = image_path
        self.rgb_bands = rgb_bands
        self.sample_band_indexes = list(sample_band_indexes)
        self.output_path = output_path
        self.precision = precision
        self.max_preview_size = max_preview_size
        self.tile_size = tile_size
        self.autosave = autosave
        self.keep_invalid = keep_invalid

        self.src: rasterio.io.DatasetReader | None = None
        self.samples_by_polygon: list[np.ndarray] = []
        self.patches: list[MplPolygon] = []

        self.fig = None
        self.ax = None
        self.selector: PolygonSelector | None = None
        self.save_button: Button | None = None
        self.undo_button: Button | None = None
        self.clear_button: Button | None = None

    @property
    def total_samples(self) -> int:
        return sum(samples.shape[0] for samples in self.samples_by_polygon)

    def run(self) -> None:
        with rasterio.open(self.image_path) as src:
            self.src = src
            rgb = read_rgb_preview(src, self.rgb_bands, self.max_preview_size)

            self.fig, self.ax = plt.subplots(figsize=(11, 8))
            plt.subplots_adjust(bottom=0.16)
            self.ax.imshow(rgb, extent=(0, src.width, src.height, 0), interpolation="nearest")
            self.ax.set_xlim(0, src.width)
            self.ax.set_ylim(src.height, 0)
            self.ax.set_xlabel("Coluna do pixel")
            self.ax.set_ylabel("Linha do pixel")
            self._update_title()

            self.selector = PolygonSelector(self.ax, self._on_polygon_selected, useblit=True)
            self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

            ax_save = self.fig.add_axes([0.58, 0.04, 0.12, 0.06])
            ax_undo = self.fig.add_axes([0.72, 0.04, 0.12, 0.06])
            ax_clear = self.fig.add_axes([0.86, 0.04, 0.10, 0.06])

            self.save_button = Button(ax_save, "Salvar")
            self.undo_button = Button(ax_undo, "Desfazer")
            self.clear_button = Button(ax_clear, "Limpar")
            self.save_button.on_clicked(lambda _event: self.save_samples())
            self.undo_button.on_clicked(lambda _event: self.undo_last_polygon())
            self.clear_button.on_clicked(lambda _event: self.clear_samples())

            print("\nMapa aberto.")
            print("Desenhe um polígono clicando no mapa e pressione Enter para coletar.")
            print("Teclas: s salva, u desfaz, c limpa, q sai.\n")
            plt.show()

    def _update_title(self) -> None:
        if self.ax is None:
            return
        r, g, b = self.rgb_bands
        self.ax.set_title(
            f"{self.image_path.name} | RGB = bandas {r},{g},{b} | alpha ignorado | "
            f"{self.total_samples} samples coletados | "
            "Enter fecha polígono, s salva, u desfaz, c limpa"
        )
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _on_key_press(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key == "s":
            self.save_samples()
        elif event.key == "u":
            self.undo_last_polygon()
        elif event.key == "c":
            self.clear_samples()
        elif event.key == "q":
            plt.close(self.fig)

    def _on_polygon_selected(self, vertices: list[tuple[float, float]]) -> None:
        if len(vertices) < 3:
            print("Polígono ignorado: são necessários pelo menos 3 vértices.")
            return

        vertices_array = np.asarray(vertices, dtype=np.float64)
        samples = self.sample_polygon(vertices_array)
        if samples.size == 0:
            print("Polígono sem pixels válidos. Nada foi adicionado.")
            self._clear_active_selector()
            return

        self.samples_by_polygon.append(samples)
        self._draw_polygon(vertices_array)
        self._update_title()
        print(
            f"Polígono coletado: {samples.shape[0]} pixels, {samples.shape[1]} bandas. "
            f"Total: {self.total_samples} samples."
        )

        if self.autosave:
            self.save_samples()

        self._clear_active_selector()

    def _clear_active_selector(self) -> None:
        if self.selector is not None:
            try:
                self.selector.clear()
            except Exception:
                pass
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _draw_polygon(self, vertices: np.ndarray) -> None:
        if self.ax is None:
            return
        patch = MplPolygon(vertices, closed=True, fill=False, linewidth=1.5)
        self.ax.add_patch(patch)
        self.patches.append(patch)
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def sample_polygon(self, vertices: np.ndarray) -> np.ndarray:
        """
        Coleta pixels dentro do polígono.

        Importante: src.read(self.sample_band_indexes, window=...) retorna as bandas
        não-alpha na ordem original do raster. Portanto cada linha do resultado fica assim:
        primeira_banda_nao_alpha segunda_banda_nao_alpha ... ultima_banda_nao_alpha
        """
        if self.src is None:
            raise RuntimeError("Raster ainda não foi aberto.")

        src = self.src
        xs = vertices[:, 0]
        ys = vertices[:, 1]

        col_start = max(0, int(math.floor(float(np.nanmin(xs)))))
        col_stop = min(src.width, int(math.ceil(float(np.nanmax(xs)))))
        row_start = max(0, int(math.floor(float(np.nanmin(ys)))))
        row_stop = min(src.height, int(math.ceil(float(np.nanmax(ys)))))

        if col_stop <= col_start or row_stop <= row_start:
            return np.empty((0, len(self.sample_band_indexes)), dtype=np.float64)

        polygon_path = MplPath(vertices)
        pieces: list[np.ndarray] = []

        for row0 in range(row_start, row_stop, self.tile_size):
            row1 = min(row0 + self.tile_size, row_stop)
            for col0 in range(col_start, col_stop, self.tile_size):
                col1 = min(col0 + self.tile_size, col_stop)
                height = row1 - row0
                width = col1 - col0
                window = Window(col0, row0, width, height)

                data = src.read(self.sample_band_indexes, window=window, masked=True)

                grid_y, grid_x = np.mgrid[row0:row1, col0:col1]
                centers = np.column_stack(((grid_x.ravel() + 0.5), (grid_y.ravel() + 0.5)))
                inside = polygon_path.contains_points(centers)
                if not np.any(inside):
                    continue

                band_count = len(self.sample_band_indexes)
                values = np.ma.filled(data, np.nan).reshape(band_count, -1).T.astype(np.float64, copy=False)

                valid = inside.copy()
                if not self.keep_invalid:
                    mask = np.ma.getmaskarray(data).reshape(band_count, -1).T
                    valid &= ~np.any(mask, axis=1)
                    valid &= np.all(np.isfinite(values), axis=1)

                tile_samples = values[valid]
                if tile_samples.size:
                    pieces.append(tile_samples)

        if not pieces:
            return np.empty((0, len(self.sample_band_indexes)), dtype=np.float64)
        return np.vstack(pieces)

    def all_samples(self) -> np.ndarray:
        band_count = len(self.sample_band_indexes)

        if not self.samples_by_polygon:
            return np.empty((0, band_count), dtype=np.float64)
        return np.vstack(self.samples_by_polygon)

    def save_samples(self) -> None:
        samples = self.all_samples()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if samples.size == 0:
            self.output_path.write_text("", encoding="utf-8")
            print(f"Nenhum sample coletado. Arquivo vazio salvo em: {self.output_path}")
            return

        np.savetxt(self.output_path, samples, fmt=self.precision, delimiter=" ")
        print(
            f"Salvo: {samples.shape[0]} samples x {samples.shape[1]} bandas em {self.output_path} "
            f"com precisão {self.precision}."
        )

    def undo_last_polygon(self) -> None:
        if not self.samples_by_polygon:
            print("Não há polígonos para desfazer.")
            return

        removed = self.samples_by_polygon.pop()
        if self.patches:
            patch = self.patches.pop()
            patch.remove()

        self._update_title()
        if self.autosave:
            self.save_samples()
        print(f"Último polígono removido: {removed.shape[0]} samples.")

    def clear_samples(self) -> None:
        self.samples_by_polygon.clear()
        for patch in self.patches:
            patch.remove()
        self.patches.clear()
        self._update_title()
        if self.autosave:
            self.save_samples()
        print("Samples da sessão atual foram limpos.")


def print_band_summary(src: rasterio.io.DatasetReader) -> None:
    print("\nBandas encontradas no raster:")
    for band_idx in src.indexes:
        label = band_label(src, band_idx)
        suffix = "  [IGNORADA: alpha]" if is_alpha_band(src, band_idx) else ""
        print(f"  {label}{suffix}")

    kept = non_alpha_band_indexes(src)
    print(f"Bandas usadas para mapeamento e samples, ignorando alpha: {kept}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coleta samples float de rasters multiespectrais desenhando polígonos.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Caminho do raster multiespectral. Se omitido, abre um seletor de arquivo.",
    )
    parser.add_argument(
        "--rgb",
        type=str,
        help="Bandas 1-based para exibição RGB no formato R,G,B. Exemplo: --rgb 4,3,2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Arquivo TXT de saída. Padrão: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=DEFAULT_PRECISION,
        help=f"Formato do np.savetxt para preservar floats. Padrão: {DEFAULT_PRECISION}",
    )
    parser.add_argument(
        "--max-preview-size",
        type=int,
        default=1600,
        help="Maior dimensão da prévia RGB. A coleta usa o raster original. Padrão: 1600",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tamanho dos blocos usados na coleta. Padrão: {DEFAULT_TILE_SIZE}",
    )
    parser.add_argument(
        "--no-autosave",
        action="store_true",
        help="Não salva automaticamente após cada polígono.",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Mantém pixels com nodata/NaN no TXT. Por padrão eles são ignorados.",
    )
    return parser


def resolve_image_path(path_from_args: Path | None) -> Path:
    if path_from_args is not None:
        return path_from_args

    path_from_dialog = choose_raster_path_by_dialog()
    if path_from_dialog is not None:
        return path_from_dialog

    typed = input("Caminho do raster multiespectral: ").strip().strip('"')
    if not typed:
        raise SystemExit("Nenhum raster informado.")
    return Path(typed)


def resolve_rgb_bands(src: rasterio.io.DatasetReader, rgb_arg: str | None) -> tuple[int, int, int]:
    allowed_band_indexes = non_alpha_band_indexes(src)
    default_rgb = default_rgb_from_colorinterp(src, allowed_band_indexes)

    if rgb_arg:
        return parse_rgb(rgb_arg, allowed_band_indexes)

    labels = band_labels(src, allowed_band_indexes)
    selected = choose_rgb_bands_by_dialog(labels, allowed_band_indexes, default_rgb)
    if selected is not None:
        return selected

    print_band_summary(src)
    print(f"Sugestão padrão: {default_rgb[0]},{default_rgb[1]},{default_rgb[2]}")
    typed = input("Bandas para RGB no formato R,G,B: ").strip()
    if not typed:
        return default_rgb
    return parse_rgb(typed, allowed_band_indexes)


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    image_path = resolve_image_path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Raster não encontrado: {image_path}")

    with rasterio.open(image_path) as src:
        print_band_summary(src)
        sample_band_indexes = non_alpha_band_indexes(src)
        rgb_bands = resolve_rgb_bands(src, args.rgb)

    sampler = MultispectralSampler(
        image_path=image_path,
        rgb_bands=rgb_bands,
        sample_band_indexes=sample_band_indexes,
        output_path=args.output,
        precision=args.precision,
        max_preview_size=args.max_preview_size,
        tile_size=args.tile_size,
        autosave=not args.no_autosave,
        keep_invalid=args.keep_invalid,
    )
    sampler.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
