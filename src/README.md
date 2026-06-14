# Codigo Fonte

## Parte 2

Execute o pipeline final a partir da raiz do projeto:

```bash
python src/t2_hyperparameter_optimization.py --trials 6 --cv-folds 5 --importance-repeats 5 --importance-top-features 30 --search-profile pragmatic
```

O script espera o dataset bruto em `data/raw/ames_mutagenicity_data.csv`.

Para um teste rapido de funcionamento, use menos trials:

```bash
python src/t2_hyperparameter_optimization.py --trials 2 --cv-folds 3 --importance-repeats 2 --importance-top-features 20
```

Para uma busca mais longa, use:

```bash
python src/t2_hyperparameter_optimization.py --trials 100 --cv-folds 5 --importance-repeats 10 --importance-top-features 50 --search-profile robust
```

Os resultados sao salvos em:

- `results/metrics/t2/`
- `results/figures/t2/`
- `results/models/t2/`
