"""
Funções utilitárias para o projeto de Machine Learning
INF01017 - Trabalho Prático 1

Este módulo contém funções auxiliares reutilizáveis em todo o projeto.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Configurações globais para visualizações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def set_random_seed(seed: int = 42) -> None:
    """
    Define seed para reprodutibilidade dos experimentos.
    
    Args:
        seed: Valor da seed (default: 42)
    """
    np.random.seed(seed)
    import random
    random.seed(seed)


def create_directories(directories: List[str]) -> None:
    """
    Cria diretórios se não existirem.
    
    Args:
        directories: Lista de caminhos de diretórios a serem criados
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Diretório verificado/criado: {directory}")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV.
    
    Args:
        filepath: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados carregados
    """
    print(f"Carregando dados de: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Salva DataFrame em arquivo CSV.
    
    Args:
        df: DataFrame a ser salvo
        filepath: Caminho do arquivo de destino
    """
    df.to_csv(filepath, index=False)
    print(f"DataFrame salvo em: {filepath}")


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Retorna informações básicas sobre o dataset.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        Dicionário com informações do dataset
    """
    info = {
        'n_samples': df.shape[0],
        'n_features': df.shape[1],
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'duplicated_rows': df.duplicated().sum(),
        'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    return info


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Imprime um resumo das características do dataset.
    
    Args:
        df: DataFrame com os dados
    """
    info = get_dataset_info(df)
    
    print("=" * 60)
    print("RESUMO DO DATASET")
    print("=" * 60)
    print(f"Número de amostras: {info['n_samples']:,}")
    print(f"Número de features: {info['n_features']}")
    print(f"Uso de memória: {info['memory_usage_mb']:.2f} MB")
    print(f"Valores faltantes: {info['missing_values']:,}")
    print(f"Linhas duplicadas: {info['duplicated_rows']}")
    print(f"Features numéricas: {len(info['numeric_features'])}")
    print(f"Features categóricas: {len(info['categorical_features'])}")
    print("=" * 60)


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
    """
    Salva figura em arquivo.
    
    Args:
        fig: Figura do matplotlib
        filepath: Caminho do arquivo de destino
        dpi: Resolução da imagem (default: 300)
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figura salva em: {filepath}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formata valor como porcentagem.
    
    Args:
        value: Valor a ser formatado (0-1)
        decimals: Número de casas decimais
        
    Returns:
        String formatada como porcentagem
    """
    return f"{value * 100:.{decimals}f}%"


def get_class_distribution(y: pd.Series) -> pd.DataFrame:
    """
    Calcula distribuição das classes.
    
    Args:
        y: Serie com os labels
        
    Returns:
        DataFrame com contagem e porcentagem de cada classe
    """
    distribution = pd.DataFrame({
        'Count': y.value_counts(),
        'Percentage': y.value_counts(normalize=True) * 100
    })
    distribution.index.name = 'Class'
    return distribution.sort_index()


def check_class_imbalance(y: pd.Series, threshold: float = 0.4) -> Tuple[bool, float]:
    """
    Verifica se há desbalanceamento de classes.
    
    Args:
        y: Serie com os labels
        threshold: Limiar para considerar desbalanceamento (default: 0.4)
        
    Returns:
        Tupla (is_imbalanced, ratio) onde ratio é a proporção da classe minoritária
    """
    class_counts = y.value_counts()
    minority_ratio = class_counts.min() / class_counts.sum()
    is_imbalanced = minority_ratio < threshold
    
    return is_imbalanced, minority_ratio


if __name__ == "__main__":
    print("Módulo de utilidades carregado com sucesso!")
    print("Este módulo contém funções auxiliares para o projeto.")
