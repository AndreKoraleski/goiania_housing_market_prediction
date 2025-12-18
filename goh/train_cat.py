import pandas as pd
from catboost import CatBoostRegressor

from .config import CONFIG
from .pipeline import ModelPipeline


def create_cat_model():
    """
    Cria e retorna um CatBoostRegressor.
    
    O CatBoost é famoso por seu algoritmo sofisticado de tratamento de 
    variáveis categóricas (Ordered Target Statistics), que geralmente 
    supera o Target Encoding tradicional.
    """
    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='MAE', 
        thread_count=-1,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        cat_features=['Rua', 'Bairro', 'Grupo']
    )


def main():
    df = pd.read_csv(CONFIG.CLEANED_DATA_DIRECTORY / "training_set.csv")
    
    pipeline = ModelPipeline(
        model_factory=create_cat_model, 
        name="CatBoost", 
        use_scaler=False,
        impute_strategy='median', 
        encode_categoricals=False  
    )
    
    pipeline.run_cross_validation(df, n_splits=10)
    pipeline.train_final_and_create_submission(df)


if __name__ == "__main__":
    main()