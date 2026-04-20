"""
04 - Spot-checking de Algoritmos
INF01017 - Trabalho Prático 1

Este script realiza o spot-checking de algoritmos de aprendizado supervisionado.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Algoritmos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from utils import load_data, set_random_seed, save_figure

# Configurações
RANDOM_SEED = 42
TRAIN_DATA_PATH = "../data/processed/train.csv"
TEST_DATA_PATH = "../data/processed/test.csv"
RESULTS_PATH = "../results/"

set_random_seed(RANDOM_SEED)


def load_processed_data():
    """
    Carrega dados processados.
    
    Returns:
        Tupla (X_train, y_train, X_test, y_test)
    """
    print("\n" + "=" * 60)
    print("CARREGANDO DADOS PROCESSADOS")
    print("=" * 60)
    
    train_df = load_data(TRAIN_DATA_PATH)
    test_df = load_data(TEST_DATA_PATH)
    
    # Separar features e target
    X_train = train_df.drop(columns=['Overall'])
    y_train = train_df['Overall']
    X_test = test_df.drop(columns=['Overall'])
    y_test = test_df['Overall']
    
    print(f"\nShape X_train: {X_train.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    print(f"Shape y_test: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def define_models():
    """
    Define os modelos a serem testados no spot-checking.
    
    Returns:
        Dicionário com os modelos
    """
    print("\n" + "=" * 60)
    print("DEFINIÇÃO DOS MODELOS")
    print("=" * 60)
    
    models = {
        # Modelos Lineares
        'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        'SVM Linear': SVC(kernel='linear', random_state=RANDOM_SEED, probability=True),
        
        # Modelos Baseados em Árvores
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100),
        
        # Boosting
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=100),
        
        # Baseado em Instâncias
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        
        # Probabilístico
        'Naive Bayes': GaussianNB(),
        
        # Redes Neurais
        'MLP Neural Network': MLPClassifier(random_state=RANDOM_SEED, max_iter=500, 
                                            hidden_layer_sizes=(100, 50))
    }
    
    print(f"\nNúmero de modelos definidos: {len(models)}")
    print("Modelos:")
    for i, name in enumerate(models.keys(), 1):
        print(f"  {i}. {name}")
    
    return models


def evaluate_model_cv(model, X, y, cv_folds: int = 5):
    """
    Avalia modelo usando cross-validation.
    
    Args:
        model: Modelo scikit-learn
        X: Features
        y: Target
        cv_folds: Número de folds
        
    Returns:
        Dicionário com métricas
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Calcular métricas com cross-validation
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    results = {
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'precision_mean': precision_scores.mean(),
        'precision_std': precision_scores.std(),
        'recall_mean': recall_scores.mean(),
        'recall_std': recall_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std(),
        'roc_auc_mean': roc_auc_scores.mean(),
        'roc_auc_std': roc_auc_scores.std()
    }
    
    return results


def spot_check_models(models: dict, X_train, y_train, cv_folds: int = 5):
    """
    Realiza spot-checking dos modelos.
    
    Args:
        models: Dicionário com os modelos
        X_train: Features de treino
        y_train: Target de treino
        cv_folds: Número de folds para cross-validation
        
    Returns:
        DataFrame com resultados
    """
    print("\n" + "=" * 60)
    print("SPOT-CHECKING DE ALGORITMOS")
    print("=" * 60)
    print(f"\nCross-validation: {cv_folds} folds")
    print("Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC")
    print("\nIniciando avaliação dos modelos...")
    
    results = []
    
    for name, model in models.items():
        print(f"\n  Avaliando: {name}...")
        try:
            metrics = evaluate_model_cv(model, X_train, y_train, cv_folds)
            metrics['model'] = name
            results.append(metrics)
            print(f"    Accuracy: {metrics['accuracy_mean']:.4f} (±{metrics['accuracy_std']:.4f})")
            print(f"    F1-Score: {metrics['f1_mean']:.4f} (±{metrics['f1_std']:.4f})")
        except Exception as e:
            print(f"    ERRO ao avaliar {name}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    # Reordenar colunas
    cols = ['model', 'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
            'recall_mean', 'recall_std', 'f1_mean', 'f1_std', 'roc_auc_mean', 'roc_auc_std']
    results_df = results_df[cols]
    
    return results_df


def visualize_results(results_df: pd.DataFrame):
    """
    Visualiza resultados do spot-checking.
    
    Args:
        results_df: DataFrame com resultados
    """
    print("\n" + "=" * 60)
    print("VISUALIZAÇÃO DOS RESULTADOS")
    print("=" * 60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Gráfico de barras comparativo
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        results_sorted = results_df.sort_values(mean_col, ascending=False)
        
        axes[idx].barh(results_sorted['model'], results_sorted[mean_col], 
                      xerr=results_sorted[std_col], color='steelblue', alpha=0.7)
        axes[idx].set_xlabel('Score')
        axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')
        axes[idx].set_xlim(0, 1)
    
    # Remover subplot extra
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    save_figure(fig, os.path.join(RESULTS_PATH, 'figures/models/spot_checking_comparison.png'))
    plt.close()
    
    # Box plot para F1-Score (requer múltiplas execuções)
    print("\nGráficos salvos em: results/figures/models/")


def rank_models(results_df: pd.DataFrame, metric: str = 'f1_mean'):
    """
    Ranqueia modelos por métrica específica.
    
    Args:
        results_df: DataFrame com resultados
        metric: Métrica para ranqueamento
        
    Returns:
        DataFrame ranqueado
    """
    print("\n" + "=" * 60)
    print(f"RANKING DOS MODELOS - Métrica: {metric.upper()}")
    print("=" * 60)
    
    ranked = results_df.sort_values(metric, ascending=False).reset_index(drop=True)
    ranked['rank'] = ranked.index + 1
    
    print("\nTop 3 modelos mais promissores:")
    for idx, row in ranked.head(3).iterrows():
        print(f"\n{idx+1}º lugar: {row['model']}")
        print(f"   Accuracy: {row['accuracy_mean']:.4f} (±{row['accuracy_std']:.4f})")
        print(f"   Precision: {row['precision_mean']:.4f} (±{row['precision_std']:.4f})")
        print(f"   Recall: {row['recall_mean']:.4f} (±{row['recall_std']:.4f})")
        print(f"   F1-Score: {row['f1_mean']:.4f} (±{row['f1_std']:.4f})")
        print(f"   ROC-AUC: {row['roc_auc_mean']:.4f} (±{row['roc_auc_std']:.4f})")
    
    return ranked


def save_results(results_df: pd.DataFrame, output_path: str = RESULTS_PATH):
    """
    Salva resultados em arquivo CSV.
    
    Args:
        results_df: DataFrame com resultados
        output_path: Diretório de saída
    """
    filepath = os.path.join(output_path, 'metrics/spot_checking_results.csv')
    results_df.to_csv(filepath, index=False)
    print(f"\nResultados salvos em: {filepath}")


def main():
    """Função principal."""
    print("=" * 60)
    print("SPOT-CHECKING DE ALGORITMOS DE MACHINE LEARNING")
    print("=" * 60)
    
    # Criar diretórios
    os.makedirs(os.path.join(RESULTS_PATH, 'figures/models'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_PATH, 'metrics'), exist_ok=True)
    
    # Carregar dados
    X_train, y_train, X_test, y_test = load_processed_data()
    
    # Definir modelos
    models = define_models()
    
    # Realizar spot-checking
    results_df = spot_check_models(models, X_train, y_train, cv_folds=5)
    
    # Visualizar resultados
    visualize_results(results_df)
    
    # Ranquear modelos
    ranked_df = rank_models(results_df, metric='f1_mean')
    
    # Salvar resultados
    save_results(ranked_df)
    
    print("\n" + "=" * 60)
    print("SPOT-CHECKING CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print("\nResultados e visualizações disponíveis em: results/")
    print("\nRecomendação: Os 2-3 algoritmos mais promissores identificados")
    print("devem ser selecionados para otimização de hiperparâmetros no T2.")


if __name__ == "__main__":
    main()
