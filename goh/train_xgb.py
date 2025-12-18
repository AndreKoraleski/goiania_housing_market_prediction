import pandas as pd
from xgboost import XGBRegressor

from .config import CONFIG
from .pipeline import ModelPipeline

def create_xgb_model():
    """
    Retorna um modelo XGBRegressor configurado.
    """
    return XGBRegressor(
        n_estimators=1000,        
        learning_rate=0.03,      
        max_depth=9,              
        min_child_weight=3,      
        gamma=0,                  
        subsample=1.0,            
        colsample_bytree=0.8,     
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )

def main():
    df = pd.read_csv(CONFIG.CLEANED_DATA_DIRECTORY / "training_set.csv")
    
    pipeline = ModelPipeline(
        model_factory=create_xgb_model, 
        name="XGBoost", 
        use_scaler=False,        
        impute_strategy='mean',
        encode_categoricals=True
    )
    
    pipeline.run_cross_validation(df, n_splits=10)
    pipeline.train_final_and_create_submission(df)

if __name__ == "__main__":
    main()