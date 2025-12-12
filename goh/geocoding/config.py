import os

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Config:
    """
    Configurações estáticas para o módulo de geocodificação.
    Parcialmente duplicada da configuração geral do projeto para facilitar o acesso.

    Attributes:
        DATA_DIRECTORY (Path): Diretório principal de dados.
        SOURCE_DATA_DIRECTORY (Path): Diretório para dados de origem.
        EXTERNAL_DATA_DIRECTORY (Path): Diretório para dados externos.
        GOOGLE_API_KEY (str): Chave da API do Google Maps.
        GEOCODING_DELAY (float): Atraso entre requisições de geocoding.
        SAVE_BATCH_SIZE (int): Número de entradas para salvar de cada vez.
    """

    # --- Caminhos ---
    DATA_DIRECTORY: Path = Path("data")
    SOURCE_DATA_DIRECTORY: Path = DATA_DIRECTORY / "source"
    EXTERNAL_DATA_DIRECTORY: Path = DATA_DIRECTORY / "external"
    GEOCODING_DATA_DIRECTORY: Path = EXTERNAL_DATA_DIRECTORY / "geocoding"


    # --- API do Geocoding---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEOCODING_API_URL: str = "https://maps.googleapis.com/maps/api/geocode/json"
    GEOCODING_API_TIMEOUT: int = 10  # Segundos de timeout para requisições de geocoding
    GEOCODING_DELAY: float = 0.1  # Segundos de atraso entre as requisições
    SAVE_BATCH_SIZE: int = 100  # Número de entradas para salvar de cada vez


    def __post_init__(self) -> None:
        """
        Cria os diretórios necessários se eles não existirem.
        """
        paths = [
            self.DATA_DIRECTORY,
            self.SOURCE_DATA_DIRECTORY,
            self.EXTERNAL_DATA_DIRECTORY,
            self.GEOCODING_DATA_DIRECTORY
        ]

        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

# Instância global de configuração
GEOCODING_CONFIG = Config()