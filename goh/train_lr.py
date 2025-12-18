import warnings
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

from .config import CONFIG
from .pipeline import ModelPipeline

from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def create_grid_search_model():
    """
    Retorna um objeto GridSearchCV configurado.
    """
    elastic_net = ElasticNet(
        max_iter=100000,
        tol=1e-3,
        random_state=42
    )
    
    parameter_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
    }
    
    inner_cross_validation = KFold(n_splits=5, shuffle=True, random_state=42)
    
    return GridSearchCV(
        estimator=elastic_net,
        param_grid=parameter_grid,
        cv=inner_cross_validation,
        scoring='neg_mean_absolute_error', 
        n_jobs=-1,
        verbose=0
    )

def main():
    df = pd.read_csv(CONFIG.CLEANED_DATA_DIRECTORY / "training_set.csv")
    
    pipeline = ModelPipeline(
        model_factory=create_grid_search_model,
        name="ElasticNet_GridSearch",
        use_scaler=True,
        impute_strategy='median',
        encode_categoricals=True
    )
    
    pipeline.run_cross_validation(df, n_splits=10)
    pipeline.train_final_and_create_submission(df)

if __name__ == "__main__":
    main()