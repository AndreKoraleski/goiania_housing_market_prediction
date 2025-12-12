from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """
    Configuração centralizada dos caminhos do projeto.
    
    Attributes:
        DATA_DIRECTORY (Path): Diretório raiz de dados.
        OUTPUT_DIRECTORY (Path): Diretório raiz de outputs.
    """

    # --- Caminhos de Dados ---
    DATA_DIRECTORY: Path = Path("data")
    SOURCE_DATA_DIRECTORY: Path = DATA_DIRECTORY / "source"
    CLEANED_DATA_DIRECTORY: Path = DATA_DIRECTORY / "cleaned"
    EXTERNAL_DATA_DIRECTORY: Path = DATA_DIRECTORY / "external"
    GEOCODING_DATA_DIRECTORY: Path = EXTERNAL_DATA_DIRECTORY / "geocoding"

    # --- Caminhos de Output (Novos) ---
    OUTPUT_DIRECTORY: Path = Path("output")
    METRICS_DIRECTORY: Path = OUTPUT_DIRECTORY / "metrics"
    SUBMISSIONS_DIRECTORY: Path = OUTPUT_DIRECTORY / "submissions"
    PLOTS_DIRECTORY: Path = OUTPUT_DIRECTORY / "plots"

    def __post_init__(self) -> None:
        """
        Cria os diretórios necessários automaticamente na inicialização.
        """
        paths = [
            self.DATA_DIRECTORY,
            self.SOURCE_DATA_DIRECTORY,
            self.CLEANED_DATA_DIRECTORY,
            self.EXTERNAL_DATA_DIRECTORY,
            self.GEOCODING_DATA_DIRECTORY,
            self.OUTPUT_DIRECTORY,
            self.METRICS_DIRECTORY,
            self.SUBMISSIONS_DIRECTORY,
            self.PLOTS_DIRECTORY
        ]

        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

# Instância global para ser importada
CONFIG = Config()