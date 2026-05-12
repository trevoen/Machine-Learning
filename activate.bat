@echo off
REM Script de ativação do ambiente virtual
REM Execute com: activate.bat

echo ============================================
echo Ativando Ambiente Virtual
echo Predição de Mutagenicidade Ames
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
