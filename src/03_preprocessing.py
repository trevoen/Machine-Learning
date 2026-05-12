"""
03 - Pré-processamento dos Dados
INF01017 - Trabalho Prático 1

Este script realiza o pré-processamento do dataset de mutagenicidade Ames.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from utils import (
    load_data,
    save_dataframe,
    set_random_seed,
    print_dataset_summary
)

# Configurações
RANDOM_SEED = 42
DATA_PATH = "data/raw/ames_mutagenicity_data.csv"
PROCESSED_PATH = "data/processed/"

set_random_seed(RANDOM_SEED)


def remove_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas de metadados não necessárias para modelagem.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame sem colunas de metadados
    """
    print("\n" + "=" * 60)
    print("REMOÇÃO DE COLUNAS DE METADADOS")
    print("=" * 60)
    
    metadata_cols = ['Id', 'Name', 'CAS', 'SMILES RDKit']
    
    cols_to_remove = [col for col in metadata_cols if col in df.columns]
    
    if cols_to_remove:
        df_clean = df.drop(columns=cols_to_remove)
        print(f"\nColunas removidas: {cols_to_remove}")
        print(f"Shape anterior: {df.shape}")
        print(f"Shape atual: {df_clean.shape}")
    else:
        df_clean = df.copy()
        print("\nNenhuma coluna de metadados encontrada para remover.")
    
    return df_clean


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Trata valores faltantes no dataset.
    
    Args:
        df: DataFrame com dados
        strategy: Estratégia para lidar com missing values ('drop', 'mean', 'median')
        
    Returns:
        DataFrame com valores faltantes tratados
    """
    print("\n" + "=" * 60)
    print("TRATAMENTO DE VALORES FALTANTES")
    print("=" * 60)
    
    n_missing_before = df.isnull().sum().sum()
    print(f"\nValores faltantes antes do tratamento: {n_missing_before}")
    
    if n_missing_before == 0:
        print("Nenhum valor faltante encontrado!")
        return df
    
    df_clean = df.copy()
    
    if strategy == 'drop':
        # Remover colunas com muitos missing values (>50%)
        threshold = 0.5
        missing_ratio = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            print(f"Colunas removidas (>{threshold*100}% missing): {len(cols_to_drop)}")
        
        # Remover linhas com missing values restantes
        df_clean = df_clean.dropna()
        
    elif strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
    elif strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    n_missing_after = df_clean.isnull().sum().sum()
    print(f"Valores faltantes após tratamento: {n_missing_after}")
    print(f"Shape anterior: {df.shape}")
    print(f"Shape atual: {df_clean.shape}")
    
    return df_clean


def remove_low_variance_features(df: pd.DataFrame, feature_cols: list, 
                                 threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove features com baixa variância.
    
    Args:
        df: DataFrame com dados
        feature_cols: Lista de colunas de features
        threshold: Limiar de variância
        
    Returns:
        DataFrame sem features de baixa variância
    """
    print("\n" + "=" * 60)
    print("REMOÇÃO DE FEATURES COM BAIXA VARIÂNCIA")
    print("=" * 60)
    
    variances = df[feature_cols].var()
    low_variance_features = variances[variances < threshold].index.tolist()
    
    print(f"\nFeatures com variância < {threshold}: {len(low_variance_features)}")
    
    if low_variance_features:
        df_clean = df.drop(columns=low_variance_features)
        print(f"Features removidas: {len(low_variance_features)}")
        print(f"Shape anterior: {df.shape}")
        print(f"Shape atual: {df_clean.shape}")
    else:
        df_clean = df.copy()
        print("Nenhuma feature com baixa variância encontrada.")
    
    return df_clean


def normalize_features(df: pd.DataFrame, feature_cols: list, 
                       method: str = 'standard') -> tuple:
    """
    Normaliza features numéricas.
    
    Args:
        df: DataFrame com dados
        feature_cols: Lista de colunas de features
        method: Método de normalização ('standard' ou 'minmax')
        
    Returns:
        Tupla (DataFrame normalizado, scaler)
    """
    print("\n" + "=" * 60)
    print(f"NORMALIZAÇÃO DE FEATURES - Método: {method.upper()}")
    print("=" * 60)
    
    df_normalized = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Método '{method}' não reconhecido. Use 'standard' ou 'minmax'.")
    
    # Normalizar apenas features numéricas
    numeric_features = [col for col in feature_cols if col in df.columns 
                       and df[col].dtype in [np.float64, np.int64]]
    
    df_normalized[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    print(f"\nFeatures normalizadas: {len(numeric_features)}")
    print("\nEstatísticas após normalização (primeiras 5 features):")
    print(df_normalized[numeric_features[:5]].describe())
    
    return df_normalized, scaler


def split_data(df: pd.DataFrame, target_col: str = 'Overall', 
               test_size: float = 0.2, val_size: float = 0.1) -> dict:
    """
    Divide dados em treino, validação e teste.
    
    Args:
        df: DataFrame com dados
        target_col: Nome da coluna alvo
        test_size: Proporção do conjunto de teste
        val_size: Proporção do conjunto de validação
        
    Returns:
        Dicionário com os conjuntos divididos
    """
    print("\n" + "=" * 60)
    print("DIVISÃO DOS DADOS")
    print("=" * 60)
    
    # Separar features e target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split treino + validação / teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    
    # Split treino / validação
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"\nTamanho do conjunto de treino: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Tamanho do conjunto de validação: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Tamanho do conjunto de teste: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
    
    print("\nDistribuição das classes:")
    print(f"Treino: {y_train.value_counts().to_dict()}")
    print(f"Validação: {y_val.value_counts().to_dict()}")
    print(f"Teste: {y_test.value_counts().to_dict()}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def save_processed_data(data_splits: dict, output_path: str = PROCESSED_PATH) -> None:
    """
    Salva dados processados em arquivos CSV.
    
    Args:
        data_splits: Dicionário com os conjuntos de dados
        output_path: Diretório de saída
    """
    print("\n" + "=" * 60)
    print("SALVANDO DADOS PROCESSADOS")
    print("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Criar DataFrames completos
    train_df = data_splits['X_train'].copy()
    train_df['Overall'] = data_splits['y_train']
    
    val_df = data_splits['X_val'].copy()
    val_df['Overall'] = data_splits['y_val']
    
    test_df = data_splits['X_test'].copy()
    test_df['Overall'] = data_splits['y_test']
    
    # Salvar
    save_dataframe(train_df, os.path.join(output_path, 'train.csv'))
    save_dataframe(val_df, os.path.join(output_path, 'validation.csv'))
    save_dataframe(test_df, os.path.join(output_path, 'test.csv'))
    
    print(f"\nDados salvos em: {output_path}")


def main():
    """Função principal."""
    print("=" * 60)
    print("PRÉ-PROCESSAMENTO DOS DADOS")
    print("=" * 60)
    
    # Carregar dados
    df = load_data(DATA_PATH)
    print_dataset_summary(df)
    
    # Remover metadados
    df = remove_metadata_columns(df)
    
    # Identificar colunas
    target_col = 'Overall'
    strain_cols = ['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']
    partition_col = 'Partition'
    
    # Opcionalmente, remover colunas de cepas individuais (usar apenas Overall)
    # df = df.drop(columns=strain_cols, errors='ignore')
    
    # Remover coluna de partição (não será usada neste trabalho)
    if partition_col in df.columns:
        df = df.drop(columns=[partition_col])
    
    feature_cols = [col for col in df.columns if col not in [target_col] + strain_cols]
    
    # Tratar valores faltantes
    df = handle_missing_values(df, strategy='drop')
    
    # Atualizar lista de features após possível remoção de colunas
    feature_cols = [col for col in df.columns if col not in [target_col] + strain_cols]
    
    # Remover features com baixa variância
    df = remove_low_variance_features(df, feature_cols, threshold=0.01)
    
    # Atualizar lista de features novamente
    feature_cols = [col for col in df.columns if col not in [target_col] + strain_cols]
    
    # Normalizar features
    df, scaler = normalize_features(df, feature_cols, method='standard')
    
    # Dividir dados
    data_splits = split_data(df, target_col)
    
    # Salvar dados processados
    save_processed_data(data_splits)
    
    print("\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print("\nPróximo passo: Realizar Spot-checking de Algoritmos")


if __name__ == "__main__":
    main()
