import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder

from .config import CONFIG
from .reporting import calculate_regression_metrics, display_summary, check_statistical_significance, save_metrics_and_plots


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelPipeline:
    def __init__(self, model_factory, name, use_scaler=False, impute_strategy='mean'):
        """
        Pipeline genérica para treinamento e validação de modelos.

        Parameters:
            model_factory (callable): Função que retorna uma instância nova do modelo.
            name (str): Nome do modelo para logs e arquivos.
            use_scaler (bool): Se True, aplica StandardScaler (para modelos lineares).
            impute_strategy (str): 'mean' ou 'median' para preencher nulos.
        """
        self.model_factory = model_factory
        self.name = name
        self.use_scaler = use_scaler
        self.impute_strategy = impute_strategy
        
        self.categorical_columns = ['Rua', 'Bairro', 'Grupo']
        self.target_column = 'Valor'


    def _get_X_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separa features e target.

        Parameters:
            df (DataFrame): DataFrame completo com features e target.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Features (X) e target (y).
        """
        return df.drop(columns=[self.target_column]), df[self.target_column]


    def _impute(self, df, stats=None) -> tuple[pd.DataFrame, dict]:
        """
        Preenche valores nulos e retorna estatísticas usadas.

        Parameters:
            df (DataFrame): Dados a serem imputados.
            stats (dict): Estatísticas pré-calculadas para imputação.

        Returns:
            tuple[pd.DataFrame, dict]: Dados com nulos preenchidos e estatísticas usadas.
        """
        df_output = df.copy()
        numeric_columns = [c for c in df_output.columns if c not in self.categorical_columns]
        
        if stats is None:
            stats = {}
            for c in numeric_columns:
                stats[c] = df_output[c].median() if self.impute_strategy == 'median' else df_output[c].mean()
        
        for c in numeric_columns:
            imputed_value = stats.get(c, 0)
            df_output[c] = df_output[c].fillna(imputed_value)
            
        for c in self.categorical_columns:
            if c in df_output.columns:
                df_output[c] = df_output[c].astype(str).fillna('Missing')
            else:
                df_output[c] = 'Missing'
                
        return df_output, stats


    def _transform(self, X_train: pd.DataFrame, y_train_log: pd.Series, X_validation: pd.DataFrame | None = None):
        """
        Aplica TargetEncoder e opcionalmente StandardScaler.
        Retorna dados transformados e os transformers fitados.

        Parameters:
            X_train (DataFrame): Dados de treino.
            y_train_log (Series): Target log-transformado para treino.
            X_val (DataFrame) | None: Dados de validação (opcional).

        Returns:
            tuple: Dados transformados (X_train, X_val), TargetEncoder, StandardScaler (se usado).
        """
        encoder = TargetEncoder(smooth="auto", target_type="continuous", random_state=42)
        
        X_encoded_data = X_train.copy()
        X_encoded_data[self.categorical_columns] = encoder.fit_transform(X_train[self.categorical_columns], y_train_log)
        
        X_encoded_validation_data = None
        if X_validation is not None:
            X_encoded_validation_data = X_validation.copy()
            X_encoded_validation_data[self.categorical_columns] = encoder.transform(X_validation[self.categorical_columns])

        scaler = None
        if self.use_scaler:
            scaler = StandardScaler()
            X_encoded_data = pd.DataFrame(scaler.fit_transform(X_encoded_data), columns=X_encoded_data.columns, index=X_encoded_data.index)
            
            if X_encoded_validation_data is not None:
                X_encoded_validation_data = pd.DataFrame(scaler.transform(X_encoded_validation_data), columns=X_encoded_validation_data.columns, index=X_encoded_validation_data.index)

        return X_encoded_data, X_encoded_validation_data, encoder, scaler


    def run_cross_validation(self, df: pd.DataFrame, n_splits: int = 10):
        """
        Executa Cross-Validation estratificado e reporta métricas.

        Parameters:
            df (DataFrame): Dados completos para CV.
            n_splits (int): Número de folds para StratifiedKFold.
        """
        logger.info(f"--- Iniciando CV: {self.name} ({n_splits} folds) ---")
        X, y = self._get_X_y(df)
        
        y_bins = pd.qcut(y, q=n_splits, labels=False, duplicates='drop')
        stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        out_of_fold_predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(stratified_k_fold.split(X, y_bins), 1):
            X_train_set, X_validation_set = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train_set, y_validation_set = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
            
            X_train_set, stats = self._impute(X_train_set)
            X_validation_set, _ = self._impute(X_validation_set, stats)
            
            y_train_log_transformed = np.log1p(y_train_set)
            X_train_processed, X_validation_processed, _, _ = self._transform(X_train_set, y_train_log_transformed, X_validation_set)
            
            model = self.model_factory()
            model.fit(X_train_processed, y_train_log_transformed)
            
            predicted_values = np.expm1(model.predict(X_validation_processed))
            out_of_fold_predictions[val_idx] = predicted_values
            
            metrics = calculate_regression_metrics(y_validation_set, predicted_values, fold)
            fold_results.append(metrics)
            
            info_extra = ""
            if hasattr(model, 'best_params_'):
                info_extra = f" | Params={model.best_params_}"
            elif hasattr(model, 'alpha_'):
                info_extra = f" | Alpha={model.alpha_:.4f}"
            
            logger.info(f"Fold {fold:02d}: MAPE={metrics['mape']:5.2f}%{info_extra}")

        df_fold_metrics = pd.DataFrame(fold_results)
        display_summary(df_fold_metrics, self.name)
        check_statistical_significance(df_fold_metrics, len(y), len(X.columns))
        save_metrics_and_plots(y, out_of_fold_predictions, fold_results, self.name)


    def train_final_and_create_submission(self, df: pd.DataFrame):
        """
        Treina o modelo final com todos os dados e gera submissão.

        Parameters:
            df (DataFrame): Dados completos para treino final.
        """
        logger.info(f"\nTreinando modelo final: {self.name}...")
        X, y = self._get_X_y(df)
        
        X_full, stats = self._impute(X)
        y_log = np.log1p(y)
        X_processed, _, encoder, scaler = self._transform(X_full, y_log)
        
        model = self.model_factory()
        model.fit(X_processed, y_log)
        
        self._save_importance(model, X_processed.columns)
        
        test_path = CONFIG.CLEANED_DATA_DIRECTORY / "testing_set.csv"
        if not test_path.exists():
            logger.warning("Arquivo de teste não encontrado.")
            return

        df_test = pd.read_csv(test_path)
        X_test = df_test.copy()
        
        X_test, _ = self._impute(X_test, stats)
        
        training_columns = X_processed.columns
        X_test_encoded = X_test.copy()
        X_test_encoded[self.categorical_columns] = encoder.transform(X_test[self.categorical_columns])
        
        for c in training_columns:
            if c not in X_test_encoded.columns:
                X_test_encoded[c] = 0
        X_test_encoded = X_test_encoded[training_columns]
        
        if self.use_scaler and scaler:
            X_test_encoded = pd.DataFrame(scaler.transform(X_test_encoded), columns=training_columns)
            
        predictions = np.expm1(model.predict(X_test_encoded))
        
        submission_file_name = f"submission_{self.name.lower().replace(' ', '_')}.csv"
        df_submission = pd.DataFrame({'id': range(len(predictions)), 'Valor': predictions})
        df_submission.to_csv(CONFIG.SUBMISSIONS_DIRECTORY / submission_file_name, index=False)
        logger.info(f"Submissão gerada: {submission_file_name}")


    def _save_importance(self, model, features):
        try:
            estimator = model
            if hasattr(model, 'best_estimator_'):
                estimator = model.best_estimator_
            
            df_imp = None
            
            if hasattr(estimator, 'feature_importances_'):
                df_imp = pd.DataFrame({'Feature': features, 'Importance': estimator.feature_importances_})
                df_imp = df_imp.sort_values('Importance', ascending=False)
            elif hasattr(estimator, 'coef_'):
                df_imp = pd.DataFrame({'Feature': features, 'Coef': estimator.coef_, 'Abs': np.abs(estimator.coef_)})
                df_imp = df_imp.sort_values('Abs', ascending=False)
            
            if df_imp is not None:
                fname = f"importance_{self.name.lower()}.csv"
                df_imp.to_csv(CONFIG.OUTPUT_DIRECTORY / fname, index=False)
        
        except Exception as e:
            logger.warning(f"Não foi possível salvar importância das features: {e}")