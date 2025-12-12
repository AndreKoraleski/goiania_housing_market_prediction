import logging
import numpy as np
import pandas as pd

from pandas import DataFrame

from ..config import CONFIG
from .geography import validate_coordinates, enrich_with_geocoding, update_metrics
from .inconsistencies import validate_inconsistencies
from .remove import drop_columns


logger = logging.getLogger(__name__)


COLUMNS_TO_REMOVE = [
    "id",
    "sub_tipo_imovel",
    "rua_numero",
    "titulo",
    "descricao",
    "iptu_mensal",
    "andar",
    "andar_unidades",
    "microregiao_id",
    "microregiao_nome",
    "macroregiao_id",
    "macroregiao_nome",
    "grupo_num"
]

def engineer_features(df: DataFrame) -> DataFrame:
    """
    Engenharia de features, principalmente interações e logs.
    
    Parameters:
        df (DataFrame): DataFrame original.

    Returns:
        DataFrame: DataFrame com novas features criadas.
    """
    df = df.copy()
    
    if 'caracteristicas' in df.columns:
        def count_amenities(x):
            if not isinstance(x, str) or x == '{}': 
                return 0
            return len([i for i in x.strip('{}').split(',') if i.strip()])
        df['total_amenities'] = df['caracteristicas'].apply(count_amenities)

    if 'suites' in df.columns and 'quartos' in df.columns:
        df['suites_per_room'] = df['suites'] / df['quartos'].replace(0, 1)
    
    if 'suites' in df.columns and 'area_util' in df.columns:
        df['suites_per_area'] = df['suites'] / df['area_util'].replace(0, 1)

    for column in ['area_total', 'area_util']:
        if column in df.columns:
            df[f'log_{column}'] = np.log1p(df[column].fillna(0))
            
    if 'area_total' in df.columns and 'area_util' in df.columns:
        df['max_area'] = df[['area_total', 'area_util']].max(axis=1)
        df['log_max_area'] = np.log1p(df['max_area'])

    return df


def remove_outliers(df: DataFrame) -> DataFrame:
    """
    Remove outliers (Hard Removal) usando regras e IQR.

    Parameters:
        df (DataFrame): DataFrame original.

    Returns:
        DataFrame: DataFrame sem os outliers.
    """
    df = df.copy()
    initial_length = len(df)
    mask = np.ones(len(df), dtype=bool)
    
    if 'area_total' in df.columns: 
        mask &= (df['area_total'] <= 3000)
    
    if 'area_util' in df.columns: 
        mask &= (df['area_util'] <= 2500)
    
    columns_to_check = ['valor', 'latitude', 'longitude']
    for column in columns_to_check:
        if column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 2.5 * IQR, Q3 + 2.5 * IQR
            mask &= (df[column] >= lower) & (df[column] <= upper)
            
    df_clean = df[mask]
    logger.info(f"Removidas {initial_length - len(df_clean)} linhas (Outliers).")
    return df_clean


def rename_to_pt(df: DataFrame, is_test=False) -> DataFrame:
    """
    Mapeia colunas para Português, incluindo as novas features.

    Parameters:
        df (DataFrame): DataFrame original.
        is_test (bool): Se True, remove a coluna 'valor' do mapeamento.

    Returns:
        DataFrame: DataFrame com colunas renomeadas.
    """
    mapping = {
        # Localização
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'coordinate_x_m': 'Coordenada X (m)',
        'coordinate_y_m': 'Coordenada Y (m)',
        'distance_to_center_m': 'Distância ao Centro (m)',
        'rua': 'Rua',
        'bairro': 'Bairro',
        'grupo_desc': 'Grupo',

        # Dimensões principais
        'area_total': 'Área Total',
        'area_util': 'Área Útil',
        'max_area': 'Área Máxima',

        # Logs de área
        'log_area_total': 'Log - Área Total',
        'log_area_util': 'Log - Área Útil',
        'log_max_area': 'Log - Área Máxima',

        # Composição do imóvel
        'quartos': 'Qtd. Quartos',
        'suites': 'Qtd. Suítes',
        'banheiros': 'Qtd. Banheiros',
        'vagas_garagem': 'Qtd. Vagas de Garagem',

        # Proporções / métricas por área
        'suites_per_room': 'Suítes por Quarto',
        'suites_per_area': 'Suítes por Área',

        # Amenidades 
        'total_amenities': 'Qtd. Comodidades',

        # Custos
        'iptu_anual': 'IPTU Anual',
        'taxa_condominio': 'Taxa de Condomínio',

        # Alvo
        'valor': 'Valor'
    }
    
    if is_test and 'valor' in mapping:
        del mapping['valor']

    columns_to_keep = [c for c in mapping.keys() if c in df.columns]
    return df[columns_to_keep].rename(columns=mapping)


def process_dataset(df: DataFrame, dataset_name: str, is_train: bool = False) -> DataFrame:
    """
    Processa o dataset: limpeza, validação, engenharia de features e remoção de outliers.

    Parameters:
        df (DataFrame): DataFrame original.
        dataset_name (str): Nome do dataset (para logs).
        is_train (bool): Se True, aplica remoção de outliers.
    
    Returns:
        DataFrame: DataFrame processado.
    """
    logger.info(f"--- Processando {dataset_name} ---")
    
    df = drop_columns(df, COLUMNS_TO_REMOVE, ignore_missing=True)
    df = validate_inconsistencies(df)
    
    df = validate_coordinates(df)
    df = enrich_with_geocoding(df)
    df = update_metrics(df)
    
    df = engineer_features(df)
    
    if is_train:
        logger.info("Aplicando remoção de outliers...")
        df = remove_outliers(df)
    
    df = rename_to_pt(df, is_test=not is_train)
    return df


def main():
    training_set = pd.read_csv(CONFIG.SOURCE_DATA_DIRECTORY / "training_set.csv")
    testing_set = pd.read_csv(CONFIG.SOURCE_DATA_DIRECTORY / "testing_set.csv")

    training_set = process_dataset(training_set, "Training Set", is_train=True)
    testing_set = process_dataset(testing_set, "Testing Set", is_train=False)
    
    training_set.to_csv(CONFIG.CLEANED_DATA_DIRECTORY / "training_set.csv", index=False)
    testing_set.to_csv(CONFIG.CLEANED_DATA_DIRECTORY / "testing_set.csv", index=False)
    logger.info(f"Datasets limpos salvos em: {CONFIG.CLEANED_DATA_DIRECTORY}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()