import argparse
import csv
import os
import subprocess
import time
import matplotlib.pyplot as plt


DEFAULT_INPUT: str = os.environ.get(
    "BENCHMARK_INPUT",
    "/media/orthomosaic/0/0/0/assets/odm_orthophoto/odm_orthophoto.tif",
)
DEFAULT_CONF: str = os.environ.get("BENCHMARK_CONF", "./ortoconf.txt")
DEFAULT_LOG: str = os.environ.get("BENCHMARK_LOG", "./new.log")
DEFAULT_BENCHMARK_DIR: str = os.environ.get("BENCHMARK_DIR", "./compare")
DEFAULT_ORDER: str = os.environ.get("BENCHMARK_ORDER", "3")
DEFAULT_EXP: str = os.environ.get("BENCHMARK_EXP", "-1.0")
DEFAULT_TILE_SIZE: str = os.environ.get("BENCHMARK_TILE_SIZE", "2000")


def get_arguments() -> argparse.Namespace:
    """Define e faz o parsing dos argumentos do benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark de processamento por quantidade de workers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Caminho da imagem de entrada (.tif)",
    )
    parser.add_argument(
        "--conf",
        type=str,
        default=DEFAULT_CONF,
        help="Arquivo de configuración/amostras (.txt ou .maha)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=DEFAULT_LOG,
        help="Arquivo de log",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=DEFAULT_BENCHMARK_DIR,
        help="Diretório para saída do benchmark (imagem de resultado)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=int(DEFAULT_ORDER),
        help="Ordem do polinômio",
    )
    parser.add_argument(
        "--exp",
        type=float,
        default=float(DEFAULT_EXP),
        help="Expoente negativo para o cálculo",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=int(DEFAULT_TILE_SIZE),
        help="Tamanho do tile em pixels",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Número mínimo de workers para o benchmark",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Número máximo de workers para o benchmark",
    )
    return parser.parse_args()


def build_base_command(args: argparse.Namespace) -> list[str]:
    """Constrói a lista de argumentos base para src/main.py."""
    output_path = os.path.join(args.benchmark_dir, "benchmark_output.tif")
    return [
        "python3",
        "src/main.py",
        args.input,
        output_path,
        args.conf,
        args.log,
        "--order", str(args.order),
        "--exp", str(args.exp),
        "--tile-size", str(args.tile_size),
    ]


# Configurações do Teste (valores por padrão, sobrescritos por CLI)
OUTPUT_CSV_FILE: str = "time_by_worker_qtd.csv"
OUTPUT_GRAPH_FILE: str = "time_by_worker_qtd.png"


if __name__ == "__main__":
    args = get_arguments()
    BASE_COMMAND_PARTS: list[str] = build_base_command(args)
    MIN_WORKERS: int = args.min_workers
    MAX_WORKERS: int = args.max_workers

    print(f"Iniciando benchmarking de {MIN_WORKERS} a {MAX_WORKERS} workers...")
    print(f"Comando base: {' '.join(BASE_COMMAND_PARTS)} --workers [N]")

    # --- Execução e Coleta de Dados ---
    results = []  # Lista para armazenar (worker_qtd, tempo_real)

    for workers_qtd in range(MIN_WORKERS, MAX_WORKERS + 1):
        # Monta o comando final adicionando a flag de workers
        command_to_run = BASE_COMMAND_PARTS + ["--workers", str(workers_qtd)]

        print(f"\n-> Rodando com --workers {workers_qtd}...")

        try:
            start_time = time.time()
            
            # Executa o comando
            # stdout e stderr são capturados para não poluir o terminal, 
            # mas você pode remover os PIPEs se quiser ver o log do script rodando.
            process_result = subprocess.run(
                command_to_run,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            end_time = time.time()
            real_time = end_time - start_time
            
            print(f"   Tempo Real: {real_time:.4f} segundos")
            results.append((workers_qtd, real_time))

        except subprocess.CalledProcessError as e:
            print(f"   ERRO ao executar com {workers_qtd} workers!")
            print(f"   Saída de erro: {e.stderr}")
            results.append((workers_qtd, None))

    # --- Salvando em CSV ---
    if results:
        print(f"\nSalvando CSV em: {OUTPUT_CSV_FILE}")
        with open(OUTPUT_CSV_FILE, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["workers_qtd", "real_time_seconds"])
            for workers_qtd, real_time in results:
                writer.writerow([workers_qtd, real_time if real_time is not None else "ERRO"])

    # --- Geração do Gráfico ---
    plot_data = [(w, t) for w, t in results if t is not None]

    if plot_data:
        print(f"Gerando gráfico em: {OUTPUT_GRAPH_FILE}")
        workers = [item[0] for item in plot_data]
        times = [item[1] for item in plot_data]

        plt.figure(figsize=(12, 6))
        plt.plot(workers, times, marker="o", linestyle="-", color="tab:blue", label="Tempo de Processamento")
        
        # Anota o tempo em cada ponto do gráfico
        for w, t in plot_data:
            plt.annotate(f"{t:.1f}s", (w, t), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8)

        plt.title("Performance por Quantidade de Workers")
        plt.xlabel("Workers (--workers)")
        plt.ylabel("Tempo (segundos)")
        plt.xticks(workers)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        
        plt.savefig(OUTPUT_GRAPH_FILE)
        plt.close()
        print("Sucesso! Gráfico e CSV gerados.")
    else:
        print("Nenhum dado válido para gerar o gráfico.")