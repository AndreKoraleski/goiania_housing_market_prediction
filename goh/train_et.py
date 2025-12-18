import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .config import CONFIG
from .pipeline import ModelPipeline

def create_et_model():
    """
    Retorna um ExtraTreesRegressor configurado com os melhores hiperparâmetros
    encontrados via Random Search.
    """
    return ExtraTreesRegressor(
        n_estimators=300,       
        max_depth=45,           
        min_samples_split=2,   
        min_samples_leaf=1,    
        max_features=0.5,       
        bootstrap=False,       
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

def main():
    df = pd.read_csv(CONFIG.CLEANED_DATA_DIRECTORY / "training_set.csv")
    
    pipeline = ModelPipeline(
        model_factory=create_et_model, 
        name="ExtraTrees", 
        use_scaler=False,        
        impute_strategy='mean',
        encode_categoricals=True # Like RF, it prefers encoded numbers
    )
    
    pipeline.run_cross_validation(df, n_splits=10)
    pipeline.train_final_and_create_submission(df)

if __name__ == "__main__":
    main()