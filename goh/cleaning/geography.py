import logging
import numpy as np
from pandas import read_csv, DataFrame, isna

from ..config import CONFIG
from ..geocoding import calculate_metrics, BOUNDS


logger = logging.getLogger(__name__)


def _is_in_bounds(latitude: float, longitude: float) -> bool:
    """
    Verifica se a latitude e longitude estão dentro dos limites de Goiânia.

    Parameters:
        latitude (float): Latitude a ser verificada.
        longitude (float): Longitude a ser verificada.

    Returns:
        bool: True se estiver dentro dos limites, False caso contrário.
    """
    if isna(latitude) or isna(longitude):
        return False
    
    return (
        BOUNDS["south"] <= latitude <= BOUNDS["north"] and
        BOUNDS["west"] <= longitude <= BOUNDS["east"]
    )


def validate_coordinates(df: DataFrame) -> DataFrame:
    """
    Valida as coordenadas existentes. 
        1. Se estiverem no bound, mantém.
        2. Se não, tenta inverter. Se inverter e entrar no bound, usa invertido.
        3. Se inválido, coloca como NaN para ser preenchido pelo geocoding.

    Parameters:
        df (DataFrame): DataFrame contendo as colunas 'latitude' e 'longitude'.

    Returns:
        DataFrame: DataFrame com as coordenadas validadas.
    """
    logger.info("Validando coordenadas existentes...")
    
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    def check_row(row):
        latitude, longitude = row["latitude"], row["longitude"]
        
        # Cenário 1: Já está correto
        if _is_in_bounds(latitude, longitude):
            return latitude, longitude
        
        # Cenário 2: Tentar inverter
        if _is_in_bounds(longitude, latitude):
            return longitude, latitude
        
        # Cenário 3: Inválido
        return np.nan, np.nan

    coords = df.apply(check_row, axis=1, result_type="expand")
    df["latitude"] = coords[0]
    df["longitude"] = coords[1]
    
    return df


def enrich_with_geocoding(df: DataFrame) -> DataFrame:
    """
    Realiza o merge com os dados de geocodificação na ordem de prioridade:
        1. Rua + Bairro
        2. Rua
        3. Bairro

    Parameters:
        df (DataFrame): DataFrame contendo as colunas 'rua', 'bairro', 'latitude' e 'longitude'.
    
    Returns:
        DataFrame: DataFrame com as coordenadas enriquecidas.
    """
    geo_path = CONFIG.GEOCODING_DATA_DIRECTORY
    
    try:
        st_by_nhd_geo = read_csv(geo_path / "streets_by_neighborhoods_geocoding.csv")
        st_geo = read_csv(geo_path / "streets_geocoding.csv")
        nhd_geo = read_csv(geo_path / "neighborhoods_geocoding.csv")
    
    except FileNotFoundError as e:
        logger.error(f"Arquivos de geocodificação não encontrados. Rode o módulo geocoding antes. Erro: {e}")
        return df

    logger.info("Iniciando merge de geolocalização...")

    original_length = len(df)
    
    geocoding_columns = ["latitude", "longitude"] 
    
    # --- Nível 1: Rua por Bairro ---
    merged_1 = df.merge(
        st_by_nhd_geo[["rua", "bairro"] + geocoding_columns],
        on=["rua", "bairro"],
        how="left",
        suffixes=("", "_new")
    )
    df["latitude"] = df["latitude"].fillna(merged_1["latitude_new"])
    df["longitude"] = df["longitude"].fillna(merged_1["longitude_new"])

    # --- Nível 2: Rua ---
    merged_2 = df.merge(
        st_geo[["rua"] + geocoding_columns],
        on="rua",
        how="left",
        suffixes=("", "_new")
    )
    df["latitude"] = df["latitude"].fillna(merged_2["latitude_new"])
    df["longitude"] = df["longitude"].fillna(merged_2["longitude_new"])

    # --- Nível 3: Bairro ---
    merged_3 = df.merge(
        nhd_geo[["bairro"] + geocoding_columns],
        on="bairro",
        how="left",
        suffixes=("", "_new")
    )
    df["latitude"] = df["latitude"].fillna(merged_3["latitude_new"])
    df["longitude"] = df["longitude"].fillna(merged_3["longitude_new"])

    if len(df) != original_length:
        logger.warning(f"O tamanho do DataFrame mudou durante o merge! {original_length} -> {len(df)}")

    return df


def update_metrics(df: DataFrame) -> DataFrame:
    """
    Recalcula as métricas espaciais baseadas nas coordenadas finais consolidadas.

    Parameters:
        df (DataFrame): DataFrame contendo as colunas 'latitude' e 'longitude'.

    Returns:
        DataFrame: DataFrame com as métricas atualizadas.
    """
    logger.info("Recalculando métricas espaciais...")
    
    def apply_metrics(row):
        metrics = calculate_metrics(row["latitude"], row["longitude"])
        return metrics["distance_to_center_m"], metrics["coordinate_x_m"], metrics["coordinate_y_m"]

    metrics_df = df.apply(apply_metrics, axis=1, result_type="expand")
    
    df["distance_to_center_m"] = metrics_df[0]
    df["coordinate_x_m"] = metrics_df[1]
    df["coordinate_y_m"] = metrics_df[2]
    
    return df