import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from pandas import read_csv, DataFrame, concat, isna
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, TaskProgressColumn

from .config import GEOCODING_CONFIG
from .helpers import calculate_metrics
from .nhd import geocode_neighborhood
from .st import geocode_street
from .st_by_nhd import geocode_street_by_neighborhood


# --- Configurações e Constantes ---

# Nomes das colunas nos datasets
NEIGHBORHOOD_COLUMN = "bairro"
STREET_COLUMN = "rua"

# Caminhos dos arquivos de dados
TRAINING_SET_PATH = GEOCODING_CONFIG.SOURCE_DATA_DIRECTORY / "training_set.csv"
TESTING_SET_PATH = GEOCODING_CONFIG.SOURCE_DATA_DIRECTORY / "testing_set.csv"

GEOCODING_PATH = GEOCODING_CONFIG.EXTERNAL_DATA_DIRECTORY / "geocoding"

OUTPUT_NEIGHBORHOODS_GEOCODING_PATH = GEOCODING_PATH / "neighborhoods_geocoding.csv"
OUTPUT_STREETS_GEOCODING_PATH = GEOCODING_PATH / "streets_geocoding.csv"
OUTPUT_STREETS_BY_NEIGHBORHOODS_GEOCODING_PATH = GEOCODING_PATH / "streets_by_neighborhoods_geocoding.csv"


def get_geocoding_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Carrega os dados de bairros e ruas dos conjuntos de treinamento e teste.

    Returns:
        tuple[DataFrame, DataFrame, DataFrame]: Três DataFrames contendo:
            - Pares únicos de bairros e ruas.
            - Bairros únicos.
            - Ruas únicas.
    """
    logging.info("Carregando datasets...")
    training_set_nhds_and_sts: DataFrame = read_csv(TRAINING_SET_PATH, usecols=[NEIGHBORHOOD_COLUMN, STREET_COLUMN])
    testing_set_nhds_and_sts: DataFrame = read_csv(TESTING_SET_PATH, usecols=[NEIGHBORHOOD_COLUMN, STREET_COLUMN])

    all_data = concat([training_set_nhds_and_sts, testing_set_nhds_and_sts], ignore_index=True)

    unique_nhd_st_pairs: DataFrame = all_data.drop_duplicates(subset=[NEIGHBORHOOD_COLUMN, STREET_COLUMN]).dropna()

    unique_neighborhoods: DataFrame = DataFrame(all_data[NEIGHBORHOOD_COLUMN].dropna().unique(), columns=[NEIGHBORHOOD_COLUMN])
    unique_streets: DataFrame = DataFrame(all_data[STREET_COLUMN].dropna().unique(), columns=[STREET_COLUMN])

    logging.info(f"Dados carregados. Pares: {len(unique_nhd_st_pairs)} | Bairros: {len(unique_neighborhoods)} | Ruas: {len(unique_streets)}")

    return unique_nhd_st_pairs, unique_neighborhoods, unique_streets


def _process(
        df: DataFrame,
        output_path: Path,
        geocode_function: Callable,
        item_type_name: str,
        columns: list[str]
) -> None:
    """
    Processa a geocodificação de um DataFrame.
    Carrega dados existentes, completa faltantes e calcula coordenadas métricas.

    Parameters:
        df (DataFrame): DataFrame contendo os itens a serem geocodificados.
        output_path (Path): Caminho para salvar os resultados de geocodificação.
        geocode_function (Callable): Função de geocodificação a ser utilizada.
        item_type_name (str): Nome descritivo do tipo de item (para logs).
        columns (list[str]): Lista de colunas que identificam unicamente cada item.
    """
    required_columns = ["latitude", "longitude", "distance_to_center_m", "coordinate_x_m", "coordinate_y_m"]
    for column in required_columns:
        if column not in df.columns:
            df[column] = float('nan')
            df[column] = df[column].astype(float)

    if os.path.exists(output_path):
        logging.info(f"Carregando dados existentes de: {output_path}")
        existing_df = read_csv(output_path)
        
        merged = df.merge(existing_df, on=columns, how="left", suffixes=("", "_old"))
        
        for column in required_columns:
            old_column = f"{column}_old"
            if old_column in merged.columns:
                merged[column] = merged[column].fillna(merged[old_column])
        
        final_columns = columns + required_columns
        extra_columns = [c for c in df.columns if c not in final_columns] 
        df = merged[final_columns + extra_columns]

        logging.info("Merge concluído. Retomando processamento...")
    
    else:
        logging.info(f"Arquivo não encontrado. Iniciando novo em: {output_path}")

    total_items = len(df)
    
    already_geocoded = df["latitude"].notna().sum()
    
    logging.info(f"Status {item_type_name.upper()}: {already_geocoded} já geocodificados de {total_items}.")

    geocoded_count = 0
    calculated_metrics_count = 0
    failed_count = 0
    start_time = time.time()
    delay = GEOCODING_CONFIG.GEOCODING_DELAY

    progress = Progress(
        TextColumn("[cyan]Progresso[/cyan]:"),
        BarColumn(bar_width=40, style="bright_blue", complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("[green]API: {task.fields[api_calls]}[/green]"),
        TextColumn("[yellow]Métricas: {task.fields[metrics]}[/yellow]"),
        TextColumn("[red]Falhas: {task.fields[fail]}[/red]"),
        expand=True,
    )

    with progress:
        task_id = progress.add_task(
            "geo",
            total=total_items,
            api_calls=0,
            metrics=0,
            fail=0
        )
        
        progress.update(task_id, completed=0) 

        for row in df.itertuples(index=True):
            current_latitude = getattr(row, "latitude")
            current_longitude = getattr(row, "longitude")
            current_distance = getattr(row, "distance_to_center_m")

            has_coords = not isna(current_latitude) and not isna(current_longitude)
            has_metrics = not isna(current_distance)

            if has_coords and not has_metrics:
                metrics = calculate_metrics(current_latitude, current_longitude)
                df.at[row.Index, "distance_to_center_m"] = metrics["distance_to_center_m"]
                df.at[row.Index, "coordinate_x_m"] = metrics["coordinate_x_m"]
                df.at[row.Index, "coordinate_y_m"] = metrics["coordinate_y_m"]
                
                calculated_metrics_count += 1
                progress.update(task_id, advance=1, metrics=calculated_metrics_count)
                continue

            if has_coords and has_metrics:
                progress.update(task_id, advance=1)
                continue

            arguments = [getattr(row, col) for col in columns]
            result = geocode_function(*arguments)

            if result:
                latitude = result["latitude"]
                longitude = result["longitude"]
                
                metrics = calculate_metrics(latitude, longitude)

                df.at[row.Index, "latitude"] = latitude
                df.at[row.Index, "longitude"] = longitude
                df.at[row.Index, "distance_to_center_m"] = metrics["distance_to_center_m"]
                df.at[row.Index, "coordinate_x_m"] = metrics["coordinate_x_m"]
                df.at[row.Index, "coordinate_y_m"] = metrics["coordinate_y_m"]
                
                geocoded_count += 1
                progress.update(task_id, advance=1, api_calls=geocoded_count)
            
            else:
                failed_count += 1
                progress.update(task_id, advance=1, fail=failed_count)

            # Salvamento periódico
            if (geocoded_count + calculated_metrics_count) > 0 and \
               (geocoded_count + calculated_metrics_count + failed_count) % GEOCODING_CONFIG.SAVE_BATCH_SIZE == 0:
                df.to_csv(output_path, index=False)

            time.sleep(delay)

    # Salvamento final
    final = df.copy()
    final.sort_values(by=columns, inplace=True)
    final.to_csv(output_path, index=False)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Concluído {item_type_name}: {geocoded_count} chamadas API, "
        f"{calculated_metrics_count} métricas calculadas, "
        f"{failed_count} falhas. Tempo: {elapsed_time:.2f}s"
    )
    logging.info(f"Arquivo salvo: {output_path}")


def main():
    """
    Função principal.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    unique_nhd_st_pairs, unique_neighborhoods, unique_streets = get_geocoding_data()

    print("\n" + "="*60)
    print(" INICIANDO GEOCODIFICACAO ")
    print("="*60 + "\n")

    _process(
        df=DataFrame(unique_neighborhoods, columns=[NEIGHBORHOOD_COLUMN]),
        output_path=OUTPUT_NEIGHBORHOODS_GEOCODING_PATH,
        geocode_function=geocode_neighborhood,
        item_type_name="Bairros",
        columns=[NEIGHBORHOOD_COLUMN]
    )

    print("-" * 60)

    _process(
        df=DataFrame(unique_streets, columns=[STREET_COLUMN]),
        output_path=OUTPUT_STREETS_GEOCODING_PATH,
        geocode_function=geocode_street,
        item_type_name="Ruas",
        columns=[STREET_COLUMN]
    )

    print("-" * 60)

    _process(
        df=unique_nhd_st_pairs.copy(),
        output_path=OUTPUT_STREETS_BY_NEIGHBORHOODS_GEOCODING_PATH,
        geocode_function=geocode_street_by_neighborhood,
        item_type_name="Ruas por Bairro",
        columns=[STREET_COLUMN, NEIGHBORHOOD_COLUMN]
    )

    print("\n" + "="*60)
    logging.info("Processo finalizado com sucesso.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()