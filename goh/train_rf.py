import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from .config import CONFIG
from .pipeline import ModelPipeline


def create_rf_model():
    """
    Cria e retorna um RandomForestRegressor com hiperparâmetros definidos.
    Os hiperparâmetros foram escolhidos com base em experimentação prévia
    e em Random Search.

    Inicialmente, única diferença a versões anteriores é a redução de max_depth de
    infinito para 25.

    Returns:
        RandomForestRegressor: Instância do modelo configurado.
    """
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.4,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )


def main():
    df = pd.read_csv(CONFIG.CLEANED_DATA_DIRECTORY / "training_set.csv")
    
    pipeline = ModelPipeline(
        model_factory=create_rf_model, 
        name="RandomForest", 
        use_scaler=False,        
        impute_strategy='mean'
    )
    
    pipeline.run_cross_validation(df, n_splits=10)
    pipeline.train_final_and_create_submission(df)


if __name__ == "__main__":
    main()
