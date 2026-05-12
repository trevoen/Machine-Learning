# Script de ativação do ambiente virtual
# Execute com: .\activate.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Ativando Ambiente Virtual" -ForegroundColor Cyan
Write-Host "Predição de Mutagenicidade Ames" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$envPath = "machine-learning-py3.13.5\Scripts\Activate.ps1"

if (Test-Path $envPath) {
    Write-Host "Ativando ambiente: machine-learning-py3.13.5" -ForegroundColor Green
    & $envPath
    
    Write-Host ""
    Write-Host "Ambiente ativado com sucesso!" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Informacoes do ambiente:" -ForegroundColor Yellow
    python --version
    Write-Host ""
    
    Write-Host "Comandos uteis:" -ForegroundColor Cyan
    Write-Host "  python check_setup.py       - Verificar configuracao" -ForegroundColor White
    Write-Host "  jupyter notebook            - Iniciar Jupyter" -ForegroundColor White
    Write-Host "  deactivate                  - Desativar ambiente" -ForegroundColor White
    Write-Host ""
    
} else {
    Write-Host "Ambiente virtual nao encontrado!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Crie o ambiente com:" -ForegroundColor Yellow
    Write-Host "  python -m venv machine-learning-py3.13.5" -ForegroundColor White
    Write-Host "  pip install -r requirements.txt" -ForegroundColor White
    Write-Host ""
}
