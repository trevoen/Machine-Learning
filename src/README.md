# Código Fonte — Predição de Mutagenicidade Ames

Este diretório contém o código executável do trabalho. Para a entrega no Moodle, inclua também os notebooks do T1 (pasta `notebooks/`) neste subdiretório `src/` dentro do ZIP.

## Pré-requisitos

1. Dataset em `data/raw/ames_mutagenicity_data.csv`
2. Ambiente virtual ativado
3. Dependências: `pip install -r requirements.txt`

## Parte 1 (T1) — Notebooks

Execute na ordem, a partir da pasta `notebooks/`:

```bash
jupyter notebook
```

1. `01_exploratory_analysis.ipynb` — Análise exploratória
2. `02_data_preprocessing.ipynb` — Pré-processamento
3. `03_model_spotchecking.ipynb` — Spot-checking de 8 algoritmos

## Parte 2 (T2) — Otimização e interpretabilidade

Execute a partir da raiz do projeto:

```bash
python src/t2_hyperparameter_optimization.py --trials 100 --cv-folds 5 --importance-repeats 10
```

Ou via notebook:

```bash
jupyter notebook notebooks/04_hyperparameter_optimization_interpretability.ipynb
```

Teste rápido:

```bash
python src/t2_hyperparameter_optimization.py --trials 2 --cv-folds 3 --importance-repeats 2 --importance-top-features 20
```

Busca mais ampla:

```bash
python src/t2_hyperparameter_optimization.py --trials 100 --cv-folds 5 --importance-repeats 10 --importance-top-features 50 --search-profile robust
```

## Saídas geradas (T2)

- `results/metrics/t2/` — métricas, hiperparâmetros e erros
- `results/figures/t2/` — gráficos comparativos, ROC e matrizes de confusão
- `results/models/t2/` — pipelines otimizados (`.joblib`)
- `docs/Imagens/t2_*.png` — figuras usadas no relatório LaTeX
