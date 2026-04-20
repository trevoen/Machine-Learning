# Trabalho Prático 1 - INF01017 - Aprendizado de Máquina

## Predição de Mutagenicidade Ames

### Descrição do Problema

Este projeto tem como objetivo desenvolver modelos preditivos baseados em Aprendizado de Máquina para classificar compostos químicos quanto à sua mutagenicidade segundo o teste de Ames. O teste de Ames é amplamente utilizado para avaliar o potencial mutagênico e cancerígeno de substâncias químicas.

**Tarefa:** Classificação binária (mutagênico/não-mutagênico)

**Dataset:** Ames Mutagenicity Dataset  
**Fonte:** https://data.mendeley.com/datasets/ktc6gbfsbh/2  
**Referência:** Martínez et al. (2022) - "Multi-Task Deep Neural Networks for Ames Mutagenicity Prediction"

### Estrutura do Projeto

```
Machine-Learning/
│
├── data/                           # Dados do projeto
│   ├── raw/                        # Dados originais (não modificados)
│   │   └── ames_mutagenicity_data.csv
│   ├── processed/                  # Dados processados
│   │   ├── train.csv              # Conjunto de treino
│   │   ├── test.csv               # Conjunto de teste
│   │   └── validation.csv         # Conjunto de validação (se aplicável)
│   └── README.txt                  # Descrição do dataset
│
├── src/                            # Código fonte
│   ├── 01_data_loading.py         # Carregamento dos dados
│   ├── 02_eda.py                  # Análise Exploratória de Dados
│   ├── 03_preprocessing.py        # Pré-processamento
│   ├── 04_spot_checking.py        # Spot-checking de algoritmos
│   └── utils.py                   # Funções auxiliares
│
├── notebooks/                      # Jupyter Notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_spotchecking.ipynb
│
├── results/                        # Resultados dos experimentos
│   ├── figures/                    # Gráficos e visualizações
│   │   ├── eda/                   # Gráficos da EDA
│   │   └── models/                # Gráficos de desempenho dos modelos
│   ├── metrics/                    # Métricas de desempenho
│   │   └── spot_checking_results.csv
│   └── models/                     # Modelos treinados (se necessário)
│
├── docs/                           # Documentação
│   ├── T1 - Enunciado.pdf
│   ├── INF01017 - Diretrizes para Uso de Ferramentas de IA.pdf
│   └── INF01017 - Citando o uso de Ferramentas de IA.pdf
│
├── requirements.txt                # Dependências do projeto
├── .gitignore                      # Arquivos a serem ignorados pelo git
└── README.md                       # Este arquivo
```

### Objetivo do Trabalho (T1)

Realizar spot-checking de algoritmos de aprendizado supervisionado para classificação de mutagenicidade, incluindo:

1. **Análise Exploratória dos Dados (EDA)**
   - Caracterização do dataset
   - Análise de distribuições
   - Identificação de padrões e anomalias
   - Verificação de desbalanceamento de classes

2. **Pré-processamento dos Dados**
   - Tratamento de valores faltantes (se necessário)
   - Normalização/padronização de features
   - Seleção de features relevantes
   - Balanceamento de classes (se necessário)

3. **Definição da Abordagem**
   - Seleção de algoritmos diversificados (≥5)
   - Definição de métricas de avaliação
   - Escolha da estratégia de validação

4. **Spot-checking de Algoritmos**
   - Treinamento e avaliação dos modelos
   - Comparação de desempenho
   - Identificação dos 2-3 algoritmos mais promissores

### Dataset

O dataset contém informações sobre compostos químicos avaliados pelo teste de Ames:

- **Instâncias:** ~7000 compostos
- **Features:** ~1600 descritores moleculares (calculados com Mordred)
- **Target:** Coluna "Overall" - label de consenso para mutagenicidade Ames
- **Partições:** Train, Internal, External (conforme artigo original)
- **Features adicionais:** TA98, TA100, TA102, TA1535, TA1537 (labels por cepa)

### Instalação e Configuração

#### Requisitos

- Python 3.8+ (recomendado: Python 3.13.5)
- pip

#### Configuração do Ambiente Virtual (Recomendado)

Para isolar as dependências do projeto, crie e ative um ambiente virtual:

**1. Criar ambiente virtual:**
```bash
python -m venv machine-learning-py3.13.5
```

**2. Ativar ambiente virtual:**
- Windows:
```bash
machine-learning-py3.13.5\Scripts\activate
```
- Linux/Mac:
```bash
source machine-learning-py3.13.5/bin/activate
```

#### Instalação de Dependências

Com o ambiente virtual ativado:

```bash
pip install -r requirements.txt
```

### Como Executar

#### Opção 1: Usando Jupyter Notebooks (Recomendado)

1. Inicie o Jupyter:
```bash
jupyter notebook
```

2. Execute os notebooks na ordem:
   - `01_exploratory_analysis.ipynb`
   - `02_data_preprocessing.ipynb`
   - `03_model_spotchecking.ipynb`

#### Opção 2: Usando Scripts Python

```bash
# 1. Carregar dados
python src/01_data_loading.py

# 2. Análise exploratória
python src/02_eda.py

# 3. Pré-processamento
python src/03_preprocessing.py

# 4. Spot-checking
python src/04_spot_checking.py
```

### Algoritmos a Serem Testados

Conforme a proposta de spot-checking, será testado um conjunto diversificado de algoritmos:

1. **Modelos Lineares**
   - Regressão Logística
   - SVM Linear

2. **Modelos Baseados em Árvores**
   - Árvore de Decisão
   - Random Forest

3. **Modelos de Ensemble/Boosting**
   - Gradient Boosting (XGBoost/LightGBM)

4. **Modelos Baseados em Instâncias**
   - K-Nearest Neighbors (KNN)

5. **Outros**
   - Naive Bayes
   - Redes Neurais (MLP)

### Métricas de Avaliação

Para este problema de classificação, serão consideradas:

- **Acurácia** (Accuracy)
- **Precisão** (Precision)
- **Revocação** (Recall)
- **F1-Score**
- **AUC-ROC**

**Métrica principal:** A definir após análise do desbalanceamento das classes.

### Estratégia de Validação

- **K-Fold Cross-Validation** (k=5 ou k=10)
- **Random State fixo** para reprodutibilidade
- **Múltiplas execuções** para análise estatística

### Autores

[Nome dos integrantes do grupo]

### Disciplina

- **Curso:** INF01017 - Aprendizado de Máquina
- **Professora:** Mariana Recamonde Mendoza
- **Instituição:** UFRGS
- **Semestre:** 2026/1

### Prazo de Entrega

**Data:** 10/05/2026, 23:59h

### Licença

Este projeto é desenvolvido para fins acadêmicos.

### Referências

Martínez, M.J., Sabando, M.V., Soto, A.J., Roca, C., Requena-Triguero, C., Campillo, N.E., Páez, J.A., & Ponzoni, I. (2022). Multi-Task Deep Neural Networks for Ames Mutagenicity Prediction. ChemRxiv. DOI: 10.26434/chemrxiv-2022-852tf

Dataset: https://data.mendeley.com/datasets/ktc6gbfsbh/2
