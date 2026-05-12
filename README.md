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
│   │   ├── ames_mutagenicity_data.csv
│   │   └── README (1).txt         # Descrição do dataset original
│   └── processed/                  # Dados processados (gerados pelos notebooks)
│
├── notebooks/                      # Jupyter Notebooks (implementação principal)
│   ├── 01_exploratory_analysis.ipynb       # Análise Exploratória de Dados
│   ├── 02_data_preprocessing.ipynb         # Pré-processamento
│   └── 03_model_spotchecking.ipynb         # Spot-checking de algoritmos
│
├── results/                        # Resultados dos experimentos (gerados automaticamente)
│   ├── figures/                    # Gráficos e visualizações
│   │   ├── eda/                   # Figuras da EDA (figure_1.png, figure_2.png, figure_3.png)
│   │   └── models/                # Figuras de desempenho dos modelos
│   │       ├── models_comparison.png
│   │       ├── confusion_matrix_*.png
│   │       └── ranking_*.png
│   └── metrics/                    # Métricas de desempenho (CSV)
│       ├── eda_summary.txt
│       ├── preprocessing_summary.txt
│       ├── spot_checking_results.csv
│       └── test_results.csv
│
├── src/                            # Código fonte Python (módulos auxiliares, se houver)
│
├── machine-learning-py3.13.5/      # Ambiente virtual (não versionado)
│
├── .gitignore                      # Arquivos ignorados pelo git
├── .vscode/settings.json           # Configurações do VS Code/Cursor
├── requirements.txt                # Dependências do projeto
├── README.md                       # Este arquivo
├── CHECKLIST_T1.md                 # Verificação de requisitos do T1
└── ESTRATEGIAS_T2.md               # Estratégias recomendadas para o T2
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

- **Instâncias:** 5,536 compostos (após pré-processamento)
- **Features:** 1,359 descritores moleculares numéricos (após remoção de metadados e features problemáticas)
- **Target:** Classes mutagênicas
  - **1** (mutagênico): 56.05% das amostras
  - **-1** (não-mutagênico): 39.78% das amostras
  - **0** (indefinido): 4.17% das amostras (removido no pré-processamento)
- **Características:** Dataset de alta dimensionalidade, sem valores faltantes após limpeza, desbalanceamento moderado

### Instalação e Configuração

#### Requisitos

- Python 3.8+ (recomendado: Python 3.13.5)
- pip
- Dataset: `ames_mutagenicity_data.csv`

#### ⚠️ IMPORTANTE: Preparação do Dataset

**Antes de executar os notebooks**, certifique-se de que o dataset está na pasta correta:

1. Baixe o dataset de: https://data.mendeley.com/datasets/ktc6gbfsbh/2
2. Coloque o arquivo `ames_mutagenicity_data.csv` em: `data/raw/`

```
data/
└── raw/
    └── ames_mutagenicity_data.csv  ← O arquivo deve estar aqui!
```

**Nota:** O dataset não está versionado no Git devido ao seu tamanho, mas **está incluído no arquivo ZIP de entrega do trabalho**.

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

#### Pré-requisitos

1. ✅ Dataset `ames_mutagenicity_data.csv` na pasta `data/raw/`
2. ✅ Ambiente virtual ativado
3. ✅ Dependências instaladas (`pip install -r requirements.txt`)

#### Opção 1: Usando Jupyter Notebooks (Recomendado)

1. Certifique-se de que o ambiente virtual está ativado
2. Inicie o Jupyter:
```bash
jupyter notebook
```

3. Selecione o kernel correto:
   - **Kernel:** `machine-learning-py3.13.5` (Python 3.13.5)
   - No menu do Jupyter: Kernel → Change Kernel → machine-learning-py3.13.5

4. Execute os notebooks na ordem:
   - `01_exploratory_analysis.ipynb` - Gera 3 figuras em `results/figures/eda/`
   - `02_data_preprocessing.ipynb` - Gera dados processados e resumo
   - `03_model_spotchecking.ipynb` - Treina 8 modelos, gera métricas e figuras

**Nota:** Todos os notebooks salvam automaticamente os resultados (figuras, métricas, matrizes de confusão) em `results/`.

#### Opção 2: Abrindo Diretamente no Cursor/VS Code

1. Abra a pasta do projeto no Cursor/VS Code
2. Selecione o interpretador Python correto:
   - Ctrl+Shift+P (Windows) ou Cmd+Shift+P (Mac)
   - Digite "Python: Select Interpreter"
   - Escolha: `machine-learning-py3.13.5/Scripts/python.exe`
3. Abra e execute os notebooks diretamente no editor

### Algoritmos Testados (Spot-checking)

Foram testados 8 algoritmos diversificados, conforme a proposta de spot-checking:

1. **Modelos Lineares**
   - Logistic Regression
   - SVM Linear

2. **Modelos Baseados em Árvores**
   - Decision Tree
   - Random Forest

3. **Modelos de Ensemble/Boosting**
   - Gradient Boosting

4. **Modelos Baseados em Instâncias**
   - K-Nearest Neighbors (KNN)

5. **Modelos Probabilísticos**
   - Naive Bayes

6. **Redes Neurais**
   - MLP Neural Network

### Resultados Obtidos (Top 3)

Com base no spot-checking realizado:

1. **Gradient Boosting** - F1: 99.77% (CV), 99.91% (teste)
2. **Decision Tree** - F1: 98.24% (CV), 99.46% (teste)
3. **Random Forest** - F1: 86.43% (CV), 85.94% (teste)

Para detalhes completos sobre a metodologia, resultados e análise, consulte o relatório PDF do trabalho.

### Métricas de Avaliação

Para este problema de classificação multiclasse, foram utilizadas:

- **Accuracy** - Acurácia geral
- **Precision** - Precisão ponderada por classe (weighted average)
- **Recall** - Revocação ponderada por classe (weighted average)
- **F1-Score** - Métrica principal (weighted average)
- **ROC-AUC** - Área sob a curva ROC (One-vs-Rest, weighted average)

**Métrica principal:** F1-Score (balança precisão e revocação)

### Estratégia de Validação

- **Stratified K-Fold Cross-Validation** (k=5)
- **Holdout validation:** 70/30 train/test split (estratificado)
- **Random State fixo** (42) para reprodutibilidade total
- **Multiple runs** via cross-validation para análise estatística robusta

### Autores

- Everton Fritsch de Lima - 00334081
- [Nome do Integrante 2] - [Cartão]
- [Nome do Integrante 3] - [Cartão]

### Disciplina

- **Curso:** INF01017 - Aprendizado de Máquina
- **Professora:** Mariana Recamonde Mendoza
- **Instituição:** UFRGS - Instituto de Informática
- **Semestre:** 2026/1

### Status do Projeto

✅ **T1 Concluído** - Spot-checking de algoritmos finalizado

**Próximos passos (T2):**
- Otimização de hiperparâmetros dos 3 melhores modelos
- Análise de importância de features
- Implementação de técnicas avançadas de ensemble
- Consultar `ESTRATEGIAS_T2.md` para planejamento detalhado

### Licença

Este projeto é desenvolvido para fins acadêmicos.

### Arquivos Importantes

- **`CHECKLIST_T1.md`** - Verificação completa dos 37 requisitos do T1 (100% atendidos)
- **`ESTRATEGIAS_T2.md`** - Estratégias e recomendações para o Trabalho Prático 2
- **`results/`** - Todas as figuras e métricas geradas automaticamente pelos notebooks

### Referências

Martínez, M.J., Sabando, M.V., Soto, A.J., Roca, C., Requena-Triguero, C., Campillo, N.E., Páez, J.A., & Ponzoni, I. (2022). Multi-Task Deep Neural Networks for Ames Mutagenicity Prediction. ChemRxiv. DOI: 10.26434/chemrxiv-2022-852tf

Dataset: https://data.mendeley.com/datasets/ktc6gbfsbh/2
