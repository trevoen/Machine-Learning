"""
01 - Carregamento dos Dados
INF01017 - Trabalho Prático 1

Este script realiza o carregamento inicial do dataset de mutagenicidade Ames.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import (
    load_data, 
    print_dataset_summary, 
    set_random_seed,
    get_class_distribution
)

# Configurações
RANDOM_SEED = 42
DATA_PATH = "../data/raw/ames_mutagenicity_data.csv"

set_random_seed(RANDOM_SEED)


def load_ames_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Carrega o dataset de mutagenicidade Ames.
    
    Args:
        filepath: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados carregados
    """
    df = load_data(filepath)
    return df


def explore_target_variable(df: pd.DataFrame, target_col: str = 'Overall') -> None:
    """
    Explora a variável alvo (target).
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
    """
    print("\n" + "=" * 60)
    print(f"ANÁLISE DA VARIÁVEL ALVO: {target_col}")
    print("=" * 60)
    
    if target_col in df.columns:
        print(f"\nValores únicos: {df[target_col].unique()}")
        print(f"Número de classes: {df[target_col].nunique()}")
        
        print("\nDistribuição das classes:")
        distribution = get_class_distribution(df[target_col])
        print(distribution)
    else:
        print(f"AVISO: Coluna '{target_col}' não encontrada no dataset!")


def explore_partition_column(df: pd.DataFrame, partition_col: str = 'Partition') -> None:
    """
    Explora a coluna de partição (Train/Internal/External).
    
    Args:
        df: DataFrame com os dados
        partition_col: Nome da coluna de partição
    """
    print("\n" + "=" * 60)
    print(f"ANÁLISE DA COLUNA DE PARTIÇÃO: {partition_col}")
    print("=" * 60)
    
    if partition_col in df.columns:
        print("\nDistribuição das partições:")
        partition_dist = pd.DataFrame({
            'Count': df[partition_col].value_counts(),
            'Percentage': df[partition_col].value_counts(normalize=True) * 100
        })
        print(partition_dist)
    else:
        print(f"AVISO: Coluna '{partition_col}' não encontrada no dataset!")


def check_strain_columns(df: pd.DataFrame) -> None:
    """
    Verifica as colunas de cepas específicas do teste Ames.
    
    Args:
        df: DataFrame com os dados
    """
    strain_cols = ['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']
    
    print("\n" + "=" * 60)
    print("ANÁLISE DAS COLUNAS DE CEPAS (STRAINS)")
    print("=" * 60)
    
    available_strains = [col for col in strain_cols if col in df.columns]
    
    if available_strains:
        print(f"\nCepas disponíveis: {available_strains}")
        print("\nValores únicos por cepa:")
        for strain in available_strains:
            unique_vals = df[strain].unique()
            print(f"  {strain}: {unique_vals}")
    else:
        print("AVISO: Nenhuma coluna de cepa encontrada!")


def identify_feature_columns(df: pd.DataFrame) -> dict:
    """
    Identifica diferentes tipos de colunas no dataset.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        Dicionário com os diferentes tipos de colunas
    """
    metadata_cols = ['Id', 'Name', 'CAS', 'SMILES RDKit', 'Partition']
    strain_cols = ['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']
    target_col = ['Overall']
    
    feature_cols = [col for col in df.columns 
                   if col not in metadata_cols + strain_cols + target_col]
    
    column_info = {
        'metadata': [col for col in metadata_cols if col in df.columns],
        'strains': [col for col in strain_cols if col in df.columns],
        'target': [col for col in target_col if col in df.columns],
        'features': feature_cols
    }
    
    print("\n" + "=" * 60)
    print("IDENTIFICAÇÃO DAS COLUNAS")
    print("=" * 60)
    print(f"Colunas de metadados: {len(column_info['metadata'])}")
    print(f"Colunas de cepas: {len(column_info['strains'])}")
    print(f"Coluna alvo: {len(column_info['target'])}")
    print(f"Colunas de features (descritores moleculares): {len(column_info['features'])}")
    
    return column_info


def main():
    """Função principal."""
    print("=" * 60)
    print("CARREGAMENTO DO DATASET - MUTAGENICIDADE AMES")
    print("=" * 60)
    
    # Carregar dados
    df = load_ames_dataset()
    
    # Informações gerais
    print_dataset_summary(df)
    
    # Primeiras linhas
    print("\nPrimeiras 3 linhas do dataset:")
    print(df.head(3))
    
    # Identificar colunas
    column_info = identify_feature_columns(df)
    
    # Analisar variável alvo
    explore_target_variable(df)
    
    # Analisar partições
    explore_partition_column(df)
    
    # Analisar cepas
    check_strain_columns(df)
    
    # Verificar tipos de dados
    print("\n" + "=" * 60)
    print("TIPOS DE DADOS")
    print("=" * 60)
    print(df.dtypes.value_counts())
    
    print("\n" + "=" * 60)
    print("CARREGAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print("\nPróximo passo: Realizar Análise Exploratória dos Dados (EDA)")


if __name__ == "__main__":
    main()
