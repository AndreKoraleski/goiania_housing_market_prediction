import logging
import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)

# Definições de limites para validações
MAX_ROOM_COUNT = 30
MAX_AREA = 100000

STRICTLY_POSITIVE_COLS = ['area_total', 'area_util']
NON_NEGATIVE_COLS = ['quartos', 'banheiros', 'suites', 'vagas_garagem', 'iptu_anual', 'taxa_condominio']
CAPPED_ROOM_COLS = ['quartos', 'banheiros', 'vagas_garagem', 'suites']
CAPPED_AREA_COLS = ['area_total', 'area_util']


def _validate_area_relational(df: DataFrame) -> DataFrame:
    """
    Trata a consistência entre Area Útil e Area Total.
    Regra: area_util não pode ser maior que area_total.
    Ação: Tenta inverter (swap) os valores, assumindo erro de cadastro.

    Parameters:
        df (DataFrame): DataFrame a ser validado.

    Returns:
        DataFrame: DataFrame com as correções aplicadas.
    """
    if 'area_util' in df.columns and 'area_total' in df.columns:
        mask_swap = (
            df['area_util'].notna() &
            df['area_total'].notna() &
            (df['area_util'] > df['area_total'])
        )

        if mask_swap.any():
            logger.info(f"Ajustando {mask_swap.sum()} inconsistências entre área útil e área total.")

            maximum_area_values = df.loc[mask_swap, ['area_util', 'area_total']].max(axis=1)

            df.loc[mask_swap, 'area_util'] = maximum_area_values
            df.loc[mask_swap, 'area_total'] = maximum_area_values

    return df


def _validate_suites_relational(df: DataFrame) -> DataFrame:
    """
    Trata a consistência entre Suítes e Quartos.
    Regra: O número de suítes não pode ser maior que o total de quartos.
    Ação: Define 'suites' como NaN.

    Parameters:
        df (DataFrame): DataFrame a ser validado.

    Returns:
        DataFrame: DataFrame com as correções aplicadas.
    """
    if 'suites' in df.columns and 'quartos' in df.columns:
        mask_error = (
            df['suites'].notna() &
            df['quartos'].notna() &
            (df['suites'] > df['quartos'])
        )

        if mask_error.any():
            logger.info(f"Corrigindo {mask_error.sum()} inconsistências entre suítes e quartos.")

            max_suite_quarto_values = df.loc[mask_error, ['suites', 'quartos']].max(axis=1)

            df.loc[mask_error, 'suites'] = max_suite_quarto_values
            df.loc[mask_error, 'quartos'] = max_suite_quarto_values

    return df


def _validate_strictly_positive(df: DataFrame) -> DataFrame:
    """
    Remove valores que deveriam ser estritamente positivos (> 0).
    Aplica-se a: áreas e valores monetários principais.

    Parameters:
        df (DataFrame): DataFrame a ser validado.

    Returns:
        DataFrame: DataFrame com as correções aplicadas.
    """
    for column in STRICTLY_POSITIVE_COLS:
        if column not in df.columns:
            continue

        mask_invalid = df[column].notna() & (df[column] <= 0)

        if mask_invalid.any():
            logger.debug(f"Tornando NaN {mask_invalid.sum()} valores inválidos (<= 0) em '{column}'.")
            df.loc[mask_invalid, column] = np.nan

    return df


def _validate_non_negative(df: DataFrame) -> DataFrame:
    """
    Remove valores negativos em colunas que aceitam zero (>= 0).
    Aplica-se a: contagens de cômodos, vagas e taxas secundárias.

    Parameters:
        df (DataFrame): DataFrame a ser validado.

    Returns:
        DataFrame: DataFrame com as correções aplicadas.
    """
    for column in NON_NEGATIVE_COLS:
        if column not in df.columns:
            continue

        mask_invalid = df[column].notna() & (df[column] < 0)

        if mask_invalid.any():
            logger.debug(f"Tornando NaN {mask_invalid.sum()} valores negativos em '{column}'.")
            df.loc[mask_invalid, column] = np.nan

    return df


def _validate_hard_caps(df: DataFrame) -> DataFrame:
    """
    Remove valores absurdamente altos (Hard Caps) que indicam erro de digitação.
    Aplica-se a: cômodos (> 30) e áreas gigantescas (> 100k).

    Parameters:
        df (DataFrame): DataFrame a ser validado.

    Returns:
        DataFrame: DataFrame com as correções aplicadas.
    """
    for column in CAPPED_ROOM_COLS:
        if column not in df.columns:
            continue

        mask_absurd = df[column].notna() & (df[column] > MAX_ROOM_COUNT)

        if mask_absurd.any():
            logger.info(
                f"Aplicando cap: removendo {mask_absurd.sum()} valores > {MAX_ROOM_COUNT} em '{column}'."
            )
            df.loc[mask_absurd, column] = np.nan

    for column in CAPPED_AREA_COLS:
        if column not in df.columns:
            continue

        mask_huge = df[column].notna() & (df[column] > MAX_AREA)

        if mask_huge.any():
            logger.info(
                f"Aplicando cap: removendo {mask_huge.sum()} valores > {MAX_AREA} em '{column}'."
            )
            df.loc[mask_huge, column] = np.nan

    return df


def validate_inconsistencies(df: DataFrame) -> DataFrame:
    """
    Função orquestradora que aplica todas as validações de consistência e integridade.
    
    Args:
        df (DataFrame): DataFrame com os dados imobiliários brutos.

    Returns:
        DataFrame: DataFrame tratado, com inconsistências substituídas por NaN ou corrigidas.
    """
    df = df.copy()

    # 1. Consistência Relacional
    df = _validate_area_relational(df)
    df = _validate_suites_relational(df)

    # 2. Integridade (Sanity Checks)
    df = _validate_strictly_positive(df)
    df = _validate_non_negative(df)
    df = _validate_hard_caps(df)

    return df
