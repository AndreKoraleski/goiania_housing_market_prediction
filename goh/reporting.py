import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

from .config import CONFIG


logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true: np.ndarray | pd.Series, y_predicted: np.ndarray, fold_index: int | None = None) -> dict:
    """
    Calcula as principais métricas de avaliação para regressão.

    Parameters:
        y_true (array-like): Valores reais do target.
        y_predicted (array-like): Valores previstos pelo modelo.
        fold_index (int, optional): Índice do fold atual (para rastreamento).

    Returns:
        dict: Dicionário contendo MAPE, RMSE, MAE, R2 e fold (se fornecido).
    """
    y_true_array = np.array(y_true)
    y_predicted_array = np.array(y_predicted)
    
    metrics = {
        'mape': mean_absolute_percentage_error(y_true_array, y_predicted_array) * 100,
        'rmse': np.sqrt(mean_squared_error(y_true_array, y_predicted_array)),
        'mae': mean_absolute_error(y_true_array, y_predicted_array),
        'r2': r2_score(y_true_array, y_predicted_array)
    }
    
    if fold_index is not None:
        metrics['fold'] = fold_index
        
    return metrics


def display_summary(results_dataframe: pd.DataFrame, model_name: str):
    """
    Imprime uma tabela resumo formatada com média e desvio padrão das métricas.

    Parameters:
        results_dataframe (DataFrame): DataFrame contendo as métricas de todos os folds.
        model_name (str): Nome do modelo para exibição no cabeçalho.
    """
    means = results_dataframe.mean()
    standard_deviations = results_dataframe.std()

    print("\n" + "="*80)
    print(f"RESUMO FINAL: {model_name.upper()}")
    print("="*80)
    print(f"MAPE: {means['mape']:6.2f}% (± {standard_deviations['mape']:.2f}%)")
    print(f"RMSE: R$ {means['rmse']:,.0f} (± {standard_deviations['rmse']:,.0f})")
    print(f"MAE:  R$ {means['mae']:,.0f} (± {standard_deviations['mae']:,.0f})")
    print(f"R²:   {means['r2']:6.4f} (± {standard_deviations['r2']:.4f})")
    print("-" * 80)


def check_statistical_significance(results_dataframe: pd.DataFrame, n_total_samples: int, n_features: int, alpha: float = 0.05):
    """
    Executa o Teste F para avaliar a significância global do modelo.
    Calcula F-Statistic, Valor Crítico e P-Value baseados na média do R².

    Parameters:
        results_dataframe (DataFrame): DataFrame com resultados dos folds (deve conter coluna 'r2').
        n_total_samples (int): Número total de amostras no dataset.
        n_features (int): Número de features utilizadas no modelo.
        alpha (float): Nível de significância (padrão 0.05 para 95% de confiança).
    """
    r2_values = results_dataframe['r2'].values
    n_folds = len(r2_values)
    
    average_samples_per_fold = n_total_samples / n_folds
    
    f_statistics = []
    for r2 in r2_values:
        if 0 < r2 < 1.0:
            numerator = r2 / n_features
            denominator = (1 - r2) / (average_samples_per_fold - n_features - 1)
            f_stat = numerator / denominator
            f_statistics.append(f_stat)
            
    if not f_statistics:
        logger.warning("Não foi possível calcular estatísticas F (R² inválidos ou insuficientes).")
        return

    mean_f_statistic = np.mean(f_statistics)
    
    degrees_of_freedom_1 = n_features
    degrees_of_freedom_2 = int(average_samples_per_fold - n_features - 1)
    
    try:
        f_critical_value = stats.f.ppf(1 - alpha, degrees_of_freedom_1, degrees_of_freedom_2)
        
        # P-Value
        p_value = 1 - stats.f.cdf(mean_f_statistic, degrees_of_freedom_1, degrees_of_freedom_2)
        
        print(f"\nANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA (Teste F, alpha={alpha}):")
        print("-" * 80)
        print(f"  • Graus de Liberdade: df1={degrees_of_freedom_1}, df2={degrees_of_freedom_2}")
        print(f"  • F-Statistic (Média): {mean_f_statistic:.4f}")
        print(f"  • Valor Crítico (F-Crit): {f_critical_value:.4f}")
        print(f"  • P-Value: {p_value:.6e}")
        
        print("-" * 80)
        if mean_f_statistic > f_critical_value:
             print("  ✓ RESULTADO: O modelo é Estatisticamente Significante (Rejeita H0)")
        else:
             print("  ✗ RESULTADO: O modelo NÃO é Estatisticamente Significante (Falha em rejeitar H0)")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Erro ao calcular testes estatísticos: {e}")


def save_metrics_and_plots(y_true: np.ndarray, y_predicted: np.ndarray, fold_results: list, model_name: str):
    """
    Salva as métricas em CSV e gera gráficos de diagnóstico (Scatter e Resíduos).

    Parameters:
        y_true (array-like): Valores reais acumulados de todos os folds.
        y_predicted (array-like): Valores previstos acumulados (OOF predictions).
        fold_results (list): Lista de dicionários com métricas de cada fold.
        model_name (str): Nome do modelo para arquivos e títulos.
    """
    metrics_dataframe = pd.DataFrame(fold_results)
    csv_filename = f"metrics_{model_name.lower().replace(' ', '_')}.csv"
    csv_path = CONFIG.METRICS_DIRECTORY / csv_filename
    
    metrics_dataframe.to_csv(csv_path, index=False)
    logger.info(f"Métricas salvas em: {csv_path}")

    plt.figure(figsize=(16, 7))
    sns.set_theme(style="whitegrid")

    y_t_plot, y_p_plot = np.array(y_true), np.array(y_predicted)
    
    if len(y_t_plot) > 10000:
        indices = np.random.choice(len(y_t_plot), 10000, replace=False)
        y_t_plot, y_p_plot = y_t_plot[indices], y_p_plot[indices]

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_t_plot, y=y_p_plot, alpha=0.5, color='royalblue', edgecolor=None)
    
    min_val = min(y_t_plot.min(), y_p_plot.min())
    max_val = max(y_t_plot.max(), y_p_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (x=y)')
    
    plt.title(f"{model_name}: Real vs Previsto")
    plt.xlabel("Valor Real (R$)")
    plt.ylabel("Valor Previsto (R$)")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    plt.subplot(1, 2, 2)
    residuals = y_t_plot - y_p_plot
    sns.histplot(residuals, kde=True, color='crimson', bins=50)
    
    plt.title(f"{model_name}: Distribuição dos Resíduos")
    plt.xlabel("Erro (Real - Previsto)")
    plt.ylabel("Frequência")
    plt.axvline(x=0, color='k', linestyle='--', lw=1)

    plot_filename = f"plots_{model_name.lower().replace(' ', '_')}.png"
    plot_path = CONFIG.PLOTS_DIRECTORY / plot_filename
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    logger.info(f"Gráficos salvos em: {plot_path}")