import pandas as pd
from lightgbm import LGBMRegressor

from .config import CONFIG
from .pipeline import ModelPipeline


def create_lgbm_model():
    """
    Retorna um LGBMRegressor configurado com os melhores hiperparâmetros.
    """
    return LGBMRegressor(
        n_estimators=2000,        
        learning_rate=0.05,       
        num_leaves=70,           
        max_depth=20,             
        min_child_samples=20,     
        subsample=0.8,            
        colsample_bytree=1.0,    
        reg_alpha=0.1,           
        reg_lambda=10,           
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