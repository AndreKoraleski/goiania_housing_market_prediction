import logging

from pandas import DataFrame


logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame, columns: str | list[str], ignore_missing: bool = True) -> DataFrame:
    """
    Remove colunas especificadas de um DataFrame.

    Parameters:
        df (DataFrame): O DataFrame original.
        columns (str | list[str]): Nome da coluna ou lista de colunas a remover.
        ignore_missing (bool): Se True, apenas avisa (warning) caso uma coluna 
                               não exista. Se False, levanta um erro.

    Returns:
        DataFrame: Um novo DataFrame sem as colunas especificadas.
    """
    df = df.copy()
    
    targets = [columns] if isinstance(columns, str) else columns
    
    present_columns = [c for c in targets if c in df.columns]
    missing_columns = [c for c in targets if c not in df.columns]

    if missing_columns:
        message = f"As seguintes colunas não foram encontradas para remoção: {missing_columns}"
        if ignore_missing:
            logger.warning(message)
        
        else:
            raise KeyError(message)

    if not present_columns:
        logger.info("Nenhuma coluna válida identificada para remoção.")
        return df

    initial_shape = df.shape
    df.drop(columns=present_columns, inplace=True)
    
    logger.info(
        f"Colunas removidas: {present_columns}. "
        f"Shape: {initial_shape} -> {df.shape}"
    )

    return df