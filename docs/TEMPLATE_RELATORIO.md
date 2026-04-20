# Trabalho Prático 1 - INF01017
## Modelagem Preditiva com Aprendizado de Máquina
### Classificação de Mutagenicidade Ames

---

**Disciplina:** INF01017 - Aprendizado de Máquina  
**Professor(a):** Mariana Recamonde Mendoza  
**Instituição:** Universidade Federal do Rio Grande do Sul (UFRGS)  
**Semestre:** 2026/1

**Grupo:**
- [Nome do Aluno 1] - [Cartão UFRGS]
- [Nome do Aluno 2] - [Cartão UFRGS]
- [Nome do Aluno 3] - [Cartão UFRGS]

**Data:** [Data de Entrega]

---

## Sumário

1. [Definição do Problema e Coleta de Dados](#1-definição-do-problema-e-coleta-de-dados)
2. [Análise Exploratória e Pré-processamento dos Dados](#2-análise-exploratória-e-pré-processamento-dos-dados)
3. [Definição da Abordagem, Algoritmos e Estratégia de Avaliação](#3-definição-da-abordagem-algoritmos-e-estratégia-de-avaliação)
4. [Spot-checking de Algoritmos](#4-spot-checking-de-algoritmos)
5. [Conclusões](#5-conclusões)
6. [Referências](#6-referências)

---

## 1. Definição do Problema e Coleta de Dados

### 1.1 Contextualização

O teste de Ames é um ensaio amplamente utilizado para avaliar o potencial mutagênico de compostos químicos, sendo um importante indicador de carcinogenicidade. A identificação precoce de compostos tóxicos é essencial no desenvolvimento de fármacos e avaliação de segurança química.

### 1.2 Objetivo do Trabalho

Este trabalho tem como objetivo desenvolver modelos preditivos baseados em Aprendizado de Máquina para classificar compostos químicos quanto à sua mutagenicidade segundo o teste de Ames. Especificamente, o problema consiste em prever se um composto é mutagênico (classe positiva) ou não-mutagênico (classe negativa) a partir de descritores moleculares.

### 1.3 Dataset Utilizado

**Nome:** Ames Mutagenicity Dataset  
**Fonte:** https://data.mendeley.com/datasets/ktc6gbfsbh/2  
**Referência:** Martínez, M.J., Sabando, M.V., Soto, A.J., et al. (2022). Multi-Task Deep Neural Networks for Ames Mutagenicity Prediction.

**Características do dataset:**
- **Número de instâncias:** [PREENCHER] compostos químicos
- **Número de features:** [PREENCHER] descritores moleculares (calculados com Mordred)
- **Variável alvo:** "Overall" - label de consenso para mutagenicidade Ames
  - Classe 0: Não-mutagênico
  - Classe 1: Mutagênico
- **Features adicionais:** Labels individuais para 5 cepas bacterianas (TA98, TA100, TA102, TA1535, TA1537)

[INCLUIR DESCRIÇÃO MAIS DETALHADA DO DATASET]

---

## 2. Análise Exploratória e Pré-processamento dos Dados

### 2.1 Análise Exploratória de Dados (EDA)

#### 2.1.1 Características Gerais

[DESCREVER: tamanho do dataset, tipos de dados, memória utilizada]

#### 2.1.2 Distribuição da Variável Alvo

[INCLUIR GRÁFICO: Distribuição das classes (0 vs 1)]

**Observações:**
- Proporção classe 0 (não-mutagênico): [X]%
- Proporção classe 1 (mutagênico): [Y]%
- Desbalanceamento: [Sim/Não]

[DISCUTIR: implicações do balanceamento/desbalanceamento]

#### 2.1.3 Análise de Valores Faltantes

[INCLUIR TABELA/GRÁFICO: Quantidade de valores faltantes por feature (se houver)]

**Observações:**
[DESCREVER: se há valores faltantes, em quais features, possíveis causas]

#### 2.1.4 Estatísticas Descritivas

[INCLUIR TABELA: Estatísticas descritivas das principais features]

**Observações:**
[DESCREVER: escalas dos dados, variância, outliers identificados]

#### 2.1.5 Análise de Correlação

[INCLUIR GRÁFICO: Top N features mais correlacionadas com o target]
[INCLUIR GRÁFICO: Matriz de correlação entre top features]

**Observações:**
[DESCREVER: features mais relevantes, possíveis redundâncias, multicolinearidade]

#### 2.1.6 Distribuição das Features

[INCLUIR GRÁFICOS: Distribuições de algumas features representativas]

**Observações:**
[DESCREVER: padrões identificados, normalidade das distribuições]

### 2.2 Pré-processamento dos Dados

#### 2.2.1 Remoção de Metadados

Foram removidas as colunas de metadados que não são relevantes para a modelagem:
- Id, Name, CAS, SMILES RDKit, Partition

#### 2.2.2 Tratamento de Valores Faltantes

**Estratégia adotada:**
[DESCREVER: como foram tratados os valores faltantes - remoção, imputação, etc.]

**Justificativa:**
[EXPLICAR: por que esta estratégia foi escolhida]

#### 2.2.3 Remoção de Features com Baixa Variância

[DESCREVER: threshold utilizado, número de features removidas]

**Justificativa:**
Features com variância muito baixa têm pouco poder discriminativo e podem prejudicar alguns algoritmos.

#### 2.2.4 Normalização das Features

**Método:** StandardScaler (padronização z-score)

**Justificativa:**
A padronização é importante porque:
- As features têm escalas diferentes
- Algoritmos baseados em distância (KNN, SVM) são sensíveis à escala
- Melhora a convergência de alguns algoritmos (Redes Neurais, Regressão Logística)

#### 2.2.5 Divisão dos Dados

**Estratégia:**
- Conjunto de treino: [X]%
- Conjunto de validação: [Y]%
- Conjunto de teste: [Z]%

**Método:** Stratified Split (mantém proporção das classes)

**Justificativa:**
[EXPLICAR: importância da divisão estratificada, especialmente se há desbalanceamento]

### 2.3 Resumo do Pré-processamento

[INCLUIR TABELA: Comparação antes/depois do pré-processamento]
- Número de instâncias
- Número de features
- Proporção das classes em cada conjunto

---

## 3. Definição da Abordagem, Algoritmos e Estratégia de Avaliação

### 3.1 Abordagem de Aprendizado

**Tarefa:** Classificação binária supervisionada

**Justificativa:**
A variável alvo é categórica binária (mutagênico/não-mutagênico) e temos dados rotulados disponíveis.

### 3.2 Algoritmos Selecionados

Seguindo a proposta de spot-checking, foram selecionados [N] algoritmos com diferentes vieses indutivos:

1. **Regressão Logística**
   - Tipo: Modelo linear probabilístico
   - Viés: Assume relação linear entre features e log-odds
   - Hiperparâmetros: [padrão ou especificar]

2. **SVM Linear**
   - Tipo: Modelo baseado em margem máxima
   - Viés: Busca hiperplano ótimo de separação
   - Hiperparâmetros: [padrão ou especificar]

3. **Árvore de Decisão**
   - Tipo: Modelo não-paramétrico baseado em regras
   - Viés: Particionamento recursivo do espaço
   - Hiperparâmetros: [padrão ou especificar]

4. **Random Forest**
   - Tipo: Ensemble de árvores
   - Viés: Reduz overfitting por agregação
   - Hiperparâmetros: [padrão ou especificar]

5. **Gradient Boosting**
   - Tipo: Ensemble sequencial
   - Viés: Otimização iterativa do erro
   - Hiperparâmetros: [padrão ou especificar]

6. **K-Nearest Neighbors (KNN)**
   - Tipo: Baseado em instâncias
   - Viés: Assume similaridade local
   - Hiperparâmetros: k=[valor]

7. **Naive Bayes**
   - Tipo: Probabilístico
   - Viés: Assume independência condicional
   - Hiperparâmetros: [padrão]

8. **Rede Neural MLP**
   - Tipo: Não-linear, conexionista
   - Viés: Aprende representações intermediárias
   - Hiperparâmetros: [especificar arquitetura]

[ADICIONAR OUTROS SE NECESSÁRIO]

**Justificativa da diversidade:**
[EXPLICAR: por que estes algoritmos foram escolhidos, como cobrem diferentes tipos de viés indutivo]

### 3.3 Métricas de Avaliação

Foram selecionadas as seguintes métricas:

1. **Accuracy (Acurácia)**
   - Proporção de predições corretas
   - Útil quando classes balanceadas

2. **Precision (Precisão)**
   - Proporção de verdadeiros positivos entre predições positivas
   - Importante quando custo de falso positivo é alto

3. **Recall (Revocação/Sensibilidade)**
   - Proporção de positivos corretamente identificados
   - Importante quando custo de falso negativo é alto

4. **F1-Score**
   - Média harmônica entre precisão e recall
   - Equilibra ambas métricas

5. **ROC-AUC**
   - Área sob curva ROC
   - Avalia capacidade discriminativa do modelo

**Métrica principal:** [DEFINIR: qual métrica será priorizada e POR QUÊ]

### 3.4 Estratégia de Validação

**Método:** K-Fold Cross-Validation (Stratified)

**Parâmetros:**
- Número de folds: k=[5 ou 10]
- Estratificação: Sim (mantém proporção das classes)
- Random state: 42 (reprodutibilidade)

**Justificativa:**
[EXPLICAR: por que CV foi escolhido, vantagens sobre holdout simples]

---

## 4. Spot-checking de Algoritmos

### 4.1 Resultados da Validação Cruzada

[INCLUIR TABELA: Resultados de todos os modelos com média e desvio padrão de cada métrica]

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| ...    | X ± Y    | X ± Y     | X ± Y  | X ± Y    | X ± Y   |

### 4.2 Visualização Comparativa

[INCLUIR GRÁFICOS: Comparação visual dos modelos por métrica]
- Gráficos de barras horizontais para cada métrica
- Box plots (se múltiplas execuções)

### 4.3 Análise de Desempenho

#### 4.3.1 Desempenho Geral

[DISCUTIR: 
- Quais modelos tiveram melhor desempenho geral?
- Houve grande variação entre os modelos?
- Algum modelo claramente superior/inferior?
]

#### 4.3.2 Análise por Métrica

[DISCUTIR:
- Como os modelos se comportaram em cada métrica?
- Trade-offs observados (e.g., precisão vs recall)
- Variabilidade dos resultados (desvio padrão)
]

#### 4.3.3 Ranking dos Modelos

**Top 3 modelos mais promissores (por [métrica principal]):**

1. **[Modelo 1]**
   - Accuracy: [valor]
   - Precision: [valor]
   - Recall: [valor]
   - F1-Score: [valor]
   - ROC-AUC: [valor]
   - **Pontos fortes:** [descrever]
   - **Considerações:** [descrever]

2. **[Modelo 2]**
   - [mesmas informações]

3. **[Modelo 3]**
   - [mesmas informações]

### 4.4 Avaliação no Conjunto de Teste

[OPCIONAL: Se avaliaram os top modelos no teste]

[INCLUIR: Matrizes de confusão, relatórios de classificação]

---

## 5. Conclusões

### 5.1 Principais Achados

[SUMARIZAR:
1. Características do dataset
2. Necessidades de pré-processamento identificadas
3. Algoritmos mais promissores
4. Insights sobre o problema
]

### 5.2 Algoritmos Recomendados para o T2

Com base nos resultados do spot-checking, os seguintes algoritmos são recomendados para otimização de hiperparâmetros no T2:

1. **[Algoritmo 1]**
   - **Justificativa:** [por que este modelo é promissor]
   - **Próximos passos:** [hiperparâmetros a otimizar]

2. **[Algoritmo 2]**
   - **Justificativa:** [...]
   - **Próximos passos:** [...]

3. **[Algoritmo 3]** (opcional)
   - **Justificativa:** [...]
   - **Próximos passos:** [...]

### 5.3 Limitações e Trabalhos Futuros

[DISCUTIR:
- Limitações do trabalho atual
- Melhorias possíveis no pré-processamento
- Feature engineering a explorar
- Outras abordagens a testar
]

---

## 6. Referências

1. Martínez, M.J., Sabando, M.V., Soto, A.J., Roca, C., Requena-Triguero, C., Campillo, N.E., Páez, J.A., & Ponzoni, I. (2022). Multi-Task Deep Neural Networks for Ames Mutagenicity Prediction. ChemRxiv. DOI: 10.26434/chemrxiv-2022-852tf

2. Faceli, K., Lorena, A. C., Gama, J., & de Carvalho, A. C. P. L. F. (2023). Inteligência Artificial: Uma Abordagem de Aprendizado de Máquina (2ª ed.). LTC.

3. [ADICIONAR: outras referências consultadas - bibliotecas, artigos, recursos online]

---

## Anexos

### Anexo A: Código Fonte

O código fonte completo está disponível no diretório `src/` do arquivo .zip entregue.

Estrutura:
- `utils.py` - Funções auxiliares
- `01_data_loading.py` - Carregamento dos dados
- `02_eda.py` - Análise exploratória
- `03_preprocessing.py` - Pré-processamento
- `04_spot_checking.py` - Spot-checking de algoritmos

### Anexo B: Uso de Ferramentas de IA

[SE APLICÁVEL: Descrever ferramentas de IA utilizadas conforme diretrizes da disciplina]

---

**Fim do Relatório**
