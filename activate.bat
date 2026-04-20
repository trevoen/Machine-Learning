@echo off
REM Script de ativação do ambiente virtual
REM Execute com: activate.bat

echo ============================================
echo T1 - INF01017 - Ativando Ambiente Virtual
echo ============================================
echo.

if exist "machine-learning-py3.13.5\Scripts\activate.bat" (
    echo Ativando ambiente: machine-learning-py3.13.5
    call machine-learning-py3.13.5\Scripts\activate.bat
    
    echo.
    echo ✓ Ambiente ativado com sucesso!
    echo.
    
    echo Informacoes do ambiente:
    python --version
    echo.
    
    echo Comandos uteis:
    echo   python check_setup.py       - Verificar configuracao
    echo   jupyter notebook           - Iniciar Jupyter
    echo   python src\01_data_loading.py - Executar analises
    echo   deactivate                 - Desativar ambiente
    echo.
    
) else (
    echo ✗ Ambiente virtual nao encontrado!
    echo.
    echo Crie o ambiente com:
    echo   python -m venv machine-learning-py3.13.5
    echo   pip install -r requirements.txt
    echo.
)
