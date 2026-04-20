"""
02 - Análise Exploratória de Dados (EDA)
INF01017 - Trabalho Prático 1

Este script realiza uma análise exploratória completa do dataset de mutagenicidade Ames.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_data,
    set_random_seed,
    save_figure,
    get_class_distribution,
    check_class_imbalance
)

# Configurações
RANDOM_SEED = 42
DATA_PATH = "../data/raw/ames_mutagenicity_data.csv"
FIGURES_PATH = "../results/figures/eda/"

set_random_seed(RANDOM_SEED)


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analisa valores faltantes no dataset.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        DataFrame com estatísticas de valores faltantes
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE VALORES FALTANTES")
    print("=" * 60)
    
    missing_stats = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    if len(missing_stats) > 0:
        print(f"\nTotal de colunas com valores faltantes: {len(missing_stats)}")
        print("\nTop 10 colunas com mais valores faltantes:")
        print(missing_stats.head(10))
    else:
        print("\nNenhum valor faltante encontrado no dataset!")
    
    return missing_stats


def analyze_target_distribution(df: pd.DataFrame, target_col: str = 'Overall') -> None:
    """
    Analisa a distribuição da variável alvo.
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DA DISTRIBUIÇÃO DA VARIÁVEL ALVO")
    print("=" * 60)
    
    if target_col not in df.columns:
        print(f"ERRO: Coluna '{target_col}' não encontrada!")
        return
    
    # Distribuição
    distribution = get_class_distribution(df[target_col])
    print("\nDistribuição das classes:")
    print(distribution)
    
    # Verificar desbalanceamento
    is_imbalanced, minority_ratio = check_class_imbalance(df[target_col])
    print(f"\nRatio da classe minoritária: {minority_ratio:.2%}")
    print(f"Dataset desbalanceado: {'Sim' if is_imbalanced else 'Não'}")
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de barras
    distribution['Count'].plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Distribuição das Classes - Contagem', fontsize=14)
    axes[0].set_xlabel('Classe')
    axes[0].set_ylabel('Contagem')
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de pizza
    axes[1].pie(distribution['Count'], labels=distribution.index, autopct='%1.1f%%',
                startangle=90, colors=['steelblue', 'coral'])
    axes[1].set_title('Distribuição das Classes - Proporção', fontsize=14)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_PATH, 'target_distribution.png'))
    plt.close()


def analyze_feature_statistics(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas das features numéricas.
    
    Args:
        df: DataFrame com os dados
        feature_cols: Lista de colunas de features
        
    Returns:
        DataFrame com estatísticas descritivas
    """
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DESCRITIVAS DAS FEATURES")
    print("=" * 60)
    
    stats = df[feature_cols].describe()
    
    print(f"\nNúmero de features analisadas: {len(feature_cols)}")
    print("\nEstatísticas resumidas (primeiras 5 features):")
    print(stats.iloc[:, :5])
    
    return stats


def analyze_feature_variance(df: pd.DataFrame, feature_cols: list, threshold: float = 0.01) -> list:
    """
    Identifica features com baixa variância.
    
    Args:
        df: DataFrame com os dados
        feature_cols: Lista de colunas de features
        threshold: Limiar de variância
        
    Returns:
        Lista de features com baixa variância
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE VARIÂNCIA DAS FEATURES")
    print("=" * 60)
    
    variances = df[feature_cols].var()
    low_variance_features = variances[variances < threshold].index.tolist()
    
    print(f"\nFeatures com variância < {threshold}: {len(low_variance_features)}")
    
    if len(low_variance_features) > 0:
        print("\nPrimeiras 10 features com baixa variância:")
        print(variances[low_variance_features].head(10))
    
    return low_variance_features


def analyze_feature_correlation(df: pd.DataFrame, feature_cols: list, 
                                target_col: str = 'Overall', n_features: int = 20) -> None:
    """
    Analisa correlação entre features e com a variável alvo.
    
    Args:
        df: DataFrame com os dados
        feature_cols: Lista de colunas de features
        target_col: Nome da coluna alvo
        n_features: Número de features mais correlacionadas para visualizar
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE CORRELAÇÃO")
    print("=" * 60)
    
    # Correlação com o target
    correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    print(f"\nTop {n_features} features mais correlacionadas com o target:")
    print(correlations.head(n_features))
    
    # Visualização
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = correlations.head(n_features)
    top_features.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Top {n_features} Features Mais Correlacionadas com o Target', fontsize=14)
    ax.set_xlabel('Correlação Absoluta')
    ax.set_ylabel('Feature')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_PATH, 'feature_correlation.png'))
    plt.close()
    
    # Matriz de correlação das top features
    top_feature_names = correlations.head(n_features).index.tolist()
    corr_matrix = df[top_feature_names].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
    ax.set_title(f'Matriz de Correlação - Top {n_features} Features', fontsize=14)
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_PATH, 'correlation_matrix.png'))
    plt.close()


def analyze_feature_distributions(df: pd.DataFrame, feature_cols: list, n_features: int = 9) -> None:
    """
    Visualiza distribuições de algumas features.
    
    Args:
        df: DataFrame com os dados
        feature_cols: Lista de colunas de features
        n_features: Número de features para visualizar
    """
    print("\n" + "=" * 60)
    print("VISUALIZAÇÃO DE DISTRIBUIÇÕES")
    print("=" * 60)
    
    # Selecionar features aleatórias
    selected_features = np.random.choice(feature_cols, size=min(n_features, len(feature_cols)), 
                                        replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(selected_features):
        axes[idx].hist(df[feature].dropna(), bins=50, color='steelblue', alpha=0.7)
        axes[idx].set_title(f'{feature}', fontsize=10)
        axes[idx].set_xlabel('Valor')
        axes[idx].set_ylabel('Frequência')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_PATH, 'feature_distributions.png'))
    plt.close()
    
    print(f"\nDistribuições de {len(selected_features)} features salvas.")


def check_duplicates(df: pd.DataFrame) -> None:
    """
    Verifica duplicatas no dataset.
    
    Args:
        df: DataFrame com os dados
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE DUPLICATAS")
    print("=" * 60)
    
    n_duplicates = df.duplicated().sum()
    print(f"\nNúmero de linhas duplicadas: {n_duplicates}")
    
    if n_duplicates > 0:
        print("\nPrimeiras linhas duplicadas:")
        print(df[df.duplicated()].head())


def main():
    """Função principal."""
    print("=" * 60)
    print("ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("=" * 60)
    
    # Criar diretório de figuras
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    # Carregar dados
    df = load_data(DATA_PATH)
    
    # Identificar colunas
    metadata_cols = ['Id', 'Name', 'CAS', 'SMILES RDKit', 'Partition']
    strain_cols = ['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']
    target_col = 'Overall'
    
    feature_cols = [col for col in df.columns 
                   if col not in metadata_cols + strain_cols + [target_col]]
    
    print(f"\nNúmero de features (descritores moleculares): {len(feature_cols)}")
    
    # Análises
    analyze_missing_values(df)
    check_duplicates(df)
    analyze_target_distribution(df, target_col)
    analyze_feature_statistics(df, feature_cols)
    analyze_feature_variance(df, feature_cols)
    analyze_feature_correlation(df, feature_cols, target_col)
    analyze_feature_distributions(df, feature_cols)
    
    print("\n" + "=" * 60)
    print("EDA CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    print(f"Figuras salvas em: {FIGURES_PATH}")
    print("\nPróximo passo: Realizar Pré-processamento dos Dados")


if __name__ == "__main__":
    main()
