"""
Script de verificação do ambiente e estrutura do projeto
Predição de Mutagenicidade Ames - Aprendizado de Máquina

Execute este script para verificar se tudo está configurado corretamente.
"""

import os
import sys
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def print_header(text):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def check_python_version():
    """Verifica versão do Python."""
    print_header("VERIFICAÇÃO DA VERSÃO DO PYTHON")
    version = sys.version_info
    print(f"Versão do Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Versão adequada (Python 3.8+)")
        return True
    else:
        print("✗ AVISO: Recomendado Python 3.8 ou superior")
        return False


def check_dependencies():
    """Verifica se as bibliotecas necessárias estão instaladas."""
    print_header("VERIFICAÇÃO DE DEPENDÊNCIAS")
    
    required_packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'jupyter',
        'optuna',
        'joblib'
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            installed_packages.append(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - NÃO INSTALADO")
    
    print(f"\nInstalados: {len(installed_packages)}/{len(required_packages)}")
    
    if missing_packages:
        print("\nPacotes faltantes:", ", ".join(missing_packages))
        print("\nPara instalar, execute:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✓ Todas as dependências estão instaladas!")
        return True


def check_directory_structure():
    """Verifica estrutura de diretórios."""
    print_header("VERIFICAÇÃO DA ESTRUTURA DE DIRETÓRIOS")
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'results',
        'results/figures',
        'results/figures/eda',
        'results/figures/models',
        'results/figures/t2',
        'results/metrics',
        'results/metrics/t2',
        'results/models/t2',
        'docs/Imagens'
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for directory in required_dirs:
        if os.path.isdir(directory):
            existing_dirs.append(directory)
            print(f"✓ {directory}/")
        else:
            missing_dirs.append(directory)
            print(f"✗ {directory}/ - NÃO ENCONTRADO")
    
    print(f"\nDiretórios existentes: {len(existing_dirs)}/{len(required_dirs)}")
    
    if missing_dirs:
        print("\nDiretórios faltantes:", ", ".join(missing_dirs))
        return False
    else:
        print("\n✓ Estrutura de diretórios correta!")
        return True


def check_required_files():
    """Verifica arquivos essenciais."""
    print_header("VERIFICAÇÃO DE ARQUIVOS ESSENCIAIS")
    
    required_files = [
        'requirements.txt',
        'README.md',
        '.gitignore',
        'notebooks/01_exploratory_analysis.ipynb',
        'notebooks/02_data_preprocessing.ipynb',
        'notebooks/03_model_spotchecking.ipynb',
        'notebooks/04_hyperparameter_optimization_interpretability.ipynb',
        'src/t2_hyperparameter_optimization.py',
        'src/README.md',
        'docs/main.tex'
    ]
    
    missing_files = []
    existing_files = []
    
    for filepath in required_files:
        if os.path.isfile(filepath):
            existing_files.append(filepath)
            print(f"✓ {filepath}")
        else:
            missing_files.append(filepath)
            print(f"✗ {filepath} - NÃO ENCONTRADO")
    
    print(f"\nArquivos existentes: {len(existing_files)}/{len(required_files)}")
    
    if missing_files:
        print("\nArquivos faltantes:", ", ".join(missing_files))
        return False
    else:
        print("\n✓ Todos os arquivos essenciais presentes!")
        return True


def check_dataset():
    """Verifica se o dataset está presente."""
    print_header("VERIFICAÇÃO DO DATASET")
    
    dataset_path = 'data/raw/ames_mutagenicity_data.csv'
    
    if os.path.isfile(dataset_path):
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"✓ Dataset encontrado: {dataset_path}")
        print(f"  Tamanho: {size_mb:.2f} MB")
        
        # Tentar verificar conteúdo
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path, nrows=5)
            print(f"  Colunas: {len(df.columns)}")
            print(f"  Primeiras linhas carregadas com sucesso")
            return True
        except Exception as e:
            print(f"  ⚠ Aviso: Erro ao ler dataset: {str(e)}")
            return False
    else:
        print(f"✗ Dataset NÃO encontrado: {dataset_path}")
        print("\nCertifique-se de baixar o dataset de:")
        print("https://data.mendeley.com/datasets/ktc6gbfsbh/2")
        print("E colocar o arquivo na pasta data/raw/")
        return False


def print_summary(checks):
    """Imprime resumo das verificações."""
    print_header("RESUMO")
    
    total = len(checks)
    passed = sum(checks.values())
    
    print(f"\nVerificações passadas: {passed}/{total}")
    
    if passed == total:
        print("\n✓✓✓ TUDO OK! O projeto está configurado corretamente.")
        print("\nPróximos passos:")
        print("1. Ative o ambiente virtual:")
        print("   machine-learning-py3.13.5\\Scripts\\activate")
        print("2. Inicie o Jupyter Notebook:")
        print("   jupyter notebook")
        print("3. Execute os notebooks/scripts na ordem:")
        print("   - 01_exploratory_analysis.ipynb")
        print("   - 02_data_preprocessing.ipynb")
        print("   - 03_model_spotchecking.ipynb")
        print("   - 04_hyperparameter_optimization_interpretability.ipynb")
        print("   ou: python src/t2_hyperparameter_optimization.py --trials 100 --cv-folds 5")
    else:
        print("\n⚠ ATENÇÃO: Algumas verificações falharam.")
        print("\nResolva os problemas indicados acima antes de prosseguir.")
        print("Consulte README.md para ajuda.")
    
    print("\n" + "=" * 60)


def main():
    """Função principal."""
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DO AMBIENTE")
    print("Predição de Mutagenicidade Ames")
    print("=" * 60)
    
    checks = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Directory Structure': check_directory_structure(),
        'Required Files': check_required_files(),
        'Dataset': check_dataset()
    }
    
    print_summary(checks)


if __name__ == "__main__":
    main()
