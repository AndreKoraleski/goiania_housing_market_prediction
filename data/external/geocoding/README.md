# Dados de Geocodificação

Esta pasta contém os artefatos produzidos pela etapa de geocodificação do projeto: arquivos CSV com coordenadas geográficas e projetadas para bairros e ruas de Goiânia.

## Arquivos

* **`neighborhoods_geocoding.csv`** — geocodificação por bairro (lista única de bairros extraída dos dados de origem).
* **`streets_geocoding.csv`** — geocodificação por rua (lista única de ruas).
* **`streets_by_neighborhoods_geocoding.csv`** — geocodificação de pares **(rua, bairro)**.

## Colunas presentes nos arquivos

* **`bairro` / `rua`** — nome do bairro ou da rua.
* **`latitude`, `longitude`** — coordenadas geográficas decimais obtidas pelo provedor de geocodificação.
* **`distance_to_center_m`** — distância (em metros) até o ponto de referência usado pelo projeto (Praça Cívica).
* **`coordinate_x_m`, `coordinate_y_m`** — coordenadas projetadas em metros, referentes ao sistema local adotado (origem na Praça Cívica).

## Como os dados foram gerados

1. O processo reúne nomes únicos de bairros e ruas a partir de `data/source/training_set.csv` e `data/source/testing_set.csv`.
2. A etapa de geocodificação utiliza os seguintes módulos presentes na pasta `geocoding/`:

   * `nhd.py` para geocodificação de bairros.
   * `st.py` para geocodificação de ruas.
   * `st_by_nhd.py` para geocodificação de pares rua+bairro.
3. O arquivo `__main__.py` coordena a execução quando o módulo `goh.geocoding` é chamado.
4. O processo registra progresso em lotes para permitir processamentos parciais em caso de *crash*

## Como regenerar os arquivos

Para executar a geocodificação, instale o pacote em modo editável e rode o módulo principal:

```bash
pip install -e .
python -m goh.geocoding  
```

Em caso de dúvidas, veja o módulo principal em [goh\\geocoding\\__main__.py](goh/geocoding/__main__.py)

## Observações

* Intervalos entre requisições, tamanho dos lotes e caminhos de entrada/saída são definidos na configuração do módulo de geocodificação.
