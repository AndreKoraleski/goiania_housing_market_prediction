# Predição do Mercado Imobiliário de Goiânia

Projeto de Aprendizado de Máquina Supervisionado para predição de valores de imóveis em Goiânia, GO.

---

## Sumário

1. [Sobre a Competição](#sobre-a-competição)
2. [Visão Geral do Projeto](#visão-geral-do-projeto)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Instalação e Configuração](#instalação-e-configuração)
5. [Como Executar](#como-executar)
6. [Pipeline de Processamento](#pipeline-de-processamento)
7. [Modelos Implementados](#modelos-implementados)
8. [Avaliação](#avaliação)

---

## Sobre a Competição

Esta solução foi desenvolvida para a competição [Predição de preços de imóveis](https://www.kaggle.com/competitions/predicao-de-precos-de-imoveis) hospedada no Kaggle por Lucas Araújo.

**Objetivo:** Utilizar as características do dataset `train.csv` para encontrar um modelo que consiga prever a coluna `valor` e generalize para o dataset `test.csv`.

**Métrica de Avaliação:** Mean Absolute Percentage Error (MAPE)
```
MAPE = (1/n) * Σ|y_true - y_pred| / y_true * 100%
```

**Formato de Submissão (CSV):**
```csv
id,valor
0,5875.8
1,3263.72
2,2195.78
```

---

## Visão Geral do Projeto

Este projeto implementa um pipeline completo (embora não completamente automático) de machine learning para predição de preços de imóveis. O sistema realiza:

- Geocodificação automática de endereços via Google Maps API
- Limpeza e validação de dados imobiliários
- Engenharia de features geográficas e estruturais
- Treinamento e validação de modelos de regressão
- Avaliação estatística com validação cruzada estratificada

**Modelos implementados:**
- Regressão Linear com regularização ElasticNet
- Random Forest Regressor

---

## Estrutura do Projeto

```
goiania-housing-market-prediction/
├── data/
│   ├── source/
│   │   └── .gitkeep         # Coloque aqui train.csv e test.csv
│   ├── cleaned/             # Gerado automaticamente
│   └── external/
│       └── geocoding/       # Cache de geocodificação (incluído)
├── goh/
│   ├── cleaning/
│   │   ├── __main__.py      # Orquestração da limpeza
│   │   ├── geography.py     # Validação e enriquecimento geográfico
│   │   ├── inconsistencies.py  # Validação de consistência
│   │   └── remove.py        # Remoção de colunas
│   ├── geocoding/
│   │   ├── __main__.py      # Processo de geocodificação
│   │   ├── config.py        # Configurações da API
│   │   ├── helpers.py       # Funções utilitárias
│   │   ├── nhd.py           # Geocodificação de bairros
│   │   ├── st.py            # Geocodificação de ruas
│   │   └── st_by_nhd.py     # Geocodificação de ruas por bairro
│   ├── config.py            # Configurações globais
│   ├── pipeline.py          # Pipeline genérico de ML
│   ├── reporting.py         # Métricas e visualizações
│   ├── train_lr.py          # Treinamento - Regressão Linear
│   └── train_rf.py          # Treinamento - Random Forest
├── output/                  # Gerado automaticamente
│   ├── metrics/             # CSVs com métricas por fold
│   ├── plots/               # Gráficos de diagnóstico
│   └── submissions/         # Predições finais para Kaggle
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

**Nota:** Os dados da competição não estão incluídos no repositório conforme as regras de competição do Kaggle. Os dados de geocodificação foram pré-processados e incluídos para conveniência, e podem ser refinados manualmente ou serem regerados com atualizações das informações na fonte (API).

---

## Instalação e Configuração

### 1. Requisitos

- Python 3.10+
- Chave de API do Google Maps (opcional, apenas se quiser reprocessar geocodificação)

### 2. Clonar o Repositório

```bash
git clone <repository-url>
cd goiania-housing-market-prediction
```

### 3. Instalar Dependências

```bash
pip install -e .
```

### 4. Adicionar os Dados

Baixe os dados da competição do Kaggle e coloque-os em `data/source/`:

```
data/source/
├── training_set.csv  # ou train.csv (renomeie se necessário)
└── testing_set.csv   # ou test.csv (renomeie se necessário)
```

**Importante:** Renomeie os arquivos para `training_set.csv` e `testing_set.csv` caso os nomes originais sejam diferentes.

### 5. Configurar API do Google Maps (Opcional)

Apenas necessário se você quiser reprocessar a geocodificação:

```bash
cp .env.example .env
# Edite .env e adicione: GOOGLE_API_KEY=sua_chave_aqui
```

Os dados de geocodificação já estão incluídos em `data/external/geocoding/`, então este passo pode ser pulado.

---

## Como Executar

### Fluxo Básico (Recomendado)

Se você apenas quer treinar os modelos e gerar submissões:

```bash
# 1. Processar e limpar os dados
python -m goh.cleaning

# 2. Treinar Random Forest (modelo recomendado)
python -m goh.train_rf

# 3. Submissão gerada em: output/submissions/submission_randomforest.csv
```

### Fluxo Completo

Para executar todo o pipeline desde a geocodificação:

```bash
# 1. Geocodificar endereços (opcional - requer API key)
python -m goh.geocoding

# 2. Limpar e preparar dados
python -m goh.cleaning

# 3. Treinar Regressão Linear
python -m goh.train_lr

# 4. Treinar Random Forest
python -m goh.train_rf
```

### Saídas Geradas

Após a execução, a estrutura de diretórios será:

```
output/
├── metrics/
│   ├── metrics_elasticnet_gridsearch.csv
│   └── metrics_randomforest.csv
├── plots/
│   ├── plots_elasticnet_gridsearch.png
│   └── plots_randomforest.png
├── submissions/
│   ├── submission_elasticnet_gridsearch.csv
│   └── submission_randomforest.csv
├── importance_elasticnet_gridsearch.csv
└── importance_randomforest.csv
```

**Arquivo para submissão no Kaggle:** `output/submissions/submission_randomforest.csv`

---

## Pipeline de Processamento

### 1. Geocodificação (`goh/geocoding/`)

Enriquece os dados com coordenadas precisas e métricas espaciais.

**Funcionalidades:**
- Geocodificação hierárquica (Rua + Bairro -> Rua -> Bairro)
- Cache persistente (evita chamadas duplicadas à API)
- Validação de coordenadas dentro dos limites de Goiânia
- Cálculo de métricas espaciais:
  - Distância ao centro (Praça Cívica)
  - Coordenadas cartesianas locais (X, Y em metros, com a Praça Cívica sendo o ponto central - vetor nulo)

**Limites de Goiânia aproximados manualmente:**
```python
BOUNDS = {
    "north": -16.5080,
    "south": -16.7740,
    "east": -49.0160,
    "west": -49.3337
}
```

### 2. Limpeza e Feature Engineering (`goh/cleaning/`)

**Validações:**
- Coordenadas: verificação de bounds e inversão automática se inválidas
- Consistência relacional: Área Útil ≤ Área Total, Suítes ≤ Quartos
- Integridade: áreas > 0, contagens ≥ 0
- Hard caps: quartos/banheiros ≤ 30, áreas ≤ 100.000 m²
- Remoção de outliers (apenas conjunto de treino): IQR com limites de 2.5×

**Features Criadas:**
```python
# Métricas de área
max_area = max(area_total, area_util)
log_area_total = log(1 + area_total)
log_area_util = log(1 + area_util)
log_max_area = log(1 + max_area)

# Proporções
suites_per_room = suites / quartos
suites_per_area = suites / area_util

# Contagem
total_amenities = count(caracteristicas)
```

**Colunas Removidas:**
- id, tipo_imovel, sub_tipo_imovel, rua_numero
- titulo, descricao, caracteristicas (após extração da contagem)
- iptu_mensal, andar, andar_unidades
- microregiao_id, microregiao_nome, macroregiao_id, macroregiao_nome, grupo_num

### 3. Pipeline de ML (`goh/pipeline.py`)

**Arquitetura:**
```python
pipeline = ModelPipeline(
    model_factory=create_model,
    name="NomeModelo",
    use_scaler=False,        # True para Regressão Linear
    impute_strategy='mean'   # 'mean' ou 'median'
)
```

**Processamento:**
1. **Imputação:** média/mediana para numéricos, 'Missing' para categóricos
2. **Target Encoding:** encoding com smoothing automático para 'Rua', 'Bairro', 'Grupo'
3. **Padronização:** StandardScaler (apenas para modelos lineares)
4. **Transformação do Target:** `y_train = log(1 + valor)`, `y_pred = exp(log_pred) - 1`

**Validação Cruzada:** Stratified K-Fold (10 folds)

---

## Modelos Implementados

### 1. Regressão Linear com ElasticNet

**Arquivo:** `goh/train_lr.py`

**Configuração:**
```python
ElasticNet(
    max_iter=100000,
    tol=1e-3,
    random_state=42
)
```

**Grid Search (5-fold):**
```python
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
}
```

**Características:**
- Regularização L1 (Lasso) + L2 (Ridge)
- StandardScaler aplicado
- Seleção automática de hiperparâmetros

### 2. Random Forest Regressor

**Arquivo:** `goh/train_rf.py`

**Configuração:**
```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.4,
    random_state=42,
    n_jobs=-1
)
```

**Características:**
- 300 árvores de decisão
- Profundidade máxima de 25 níveis
- 40% das features por split

---

## Avaliação

### Métricas Calculadas

Para cada modelo, o pipeline calcula:
- **MAPE:** Mean Absolute Percentage Error (métrica oficial da competição)
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of Determination

### Teste de Significância Estatística

- **F-Statistic:** Avalia significância global do R²
- **P-value:** Calculado para α = 0.05
- Ambos os modelos apresentaram significância estatística (p < 0.001)

### Visualizações Geradas

Os gráficos salvos em `output/plots/` incluem:
- **Scatter plot:** Valores reais vs previstos (escala log)
- **Histograma:** Distribuição dos resíduos

### Importância das Features

Arquivos CSV salvos em `output/` com ranking de importância:
- `importance_elasticnet_gridsearch.csv`: Coeficientes do modelo linear
- `importance_randomforest.csv`: Feature importances do ensemble

---

## Configuração Avançada

### Modificar Hiperparâmetros

**Random Forest** (`goh/train_rf.py`):
```python
def create_rf_model():
    return RandomForestRegressor(
        n_estimators=500,        # Mais árvores
        max_depth=30,            # Maior profundidade
        max_features=0.5,        # Mais features por split
        # ...
    )
```

**Regressão Linear** (`goh/train_lr.py`):
```python
parameter_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.3, 0.5, 0.7]
}
```

### Ajustar Validação Cruzada

Em `goh/pipeline.py`:
```python
pipeline.run_cross_validation(df, n_splits=5)  # Menos folds
```

### Estratégia de Imputação

```python
pipeline = ModelPipeline(
    # ...
    impute_strategy='median'  # 'mean' ou 'median'
)
```

---

## Dependências

Definidas em `pyproject.toml`:

```toml
dependencies = [
    "pandas>=2.3.3",
    "scikit-learn>=1.8.0",
    "matplotlib>=3.10.8",
    "seaborn>=0.13.2",
    "requests>=2.32.5",
    "python-dotenv>=1.2.1",
    "rich>=14.2.0",
]
```

---

## Licença

MIT License - Copyright © 2025 Andre Koraleski

---

## Autores

Andre Koraleski - [andrekorale@gmail.com](mailto:andrekorale@gmail.com)

---

## Referências

- [Competição no Kaggle](https://www.kaggle.com/competitions/predicao-de-precos-de-imoveis)
- [Documentação scikit-learn - MAPE](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error)