import pandas as pd
from lightgbm import LGBMRegressor

from .config import CONFIG
from .pipeline import ModelPipeline


def create_lgbm_model():
    """
    Cria e retorna um LGBMRegressor.
    
    O LightGBM lida nativamente com features categóricas e valores nulos,
    geralmente sendo mais rápido e leve que o XGBoost.
    """
    return LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,      
        max_depth=-1,       
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )


def main():
    df = pd.read_csv(CONFIG.CLEANED_DATA_DIRECTORY / "training_set.csv")
    
    pipeline = ModelPipeline(
        model_factory=create_lgbm_model, 
        name="LightGBM", 
        use_scaler=False,        
        impute_strategy='mean',
        encode_categoricals=False  
    )
    
    pipeline.run_cross_validation(df, n_splits=10)
    pipeline.train_final_and_create_submission(df)


if __name__ == "__main__":
    main()