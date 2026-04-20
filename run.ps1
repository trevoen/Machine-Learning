# Script de automação para o projeto T1 - INF01017
# Execute comandos comuns de forma simplificada

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "T1 - INF01017 - Automation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Função para executar com confirmação
function Run-Step {
    param(
        [string]$Description,
        [string]$Command
    )
    
    Write-Host ""
    Write-Host ">>> $Description" -ForegroundColor Yellow
    $response = Read-Host "Executar? (s/n)"
    
    if ($response -eq 's' -or $response -eq 'S' -or $response -eq '') {
        Write-Host "Executando..." -ForegroundColor Green
        Invoke-Expression $Command
        Write-Host "Concluido!" -ForegroundColor Green
    } else {
        Write-Host "Pulado." -ForegroundColor Gray
    }
}

# Menu
Write-Host "Escolha uma opcao:" -ForegroundColor Cyan
Write-Host "1. Verificar ambiente (check_setup.py)" -ForegroundColor White
Write-Host "2. Instalar dependencias (pip install)" -ForegroundColor White
Write-Host "3. Executar pipeline completo (scripts)" -ForegroundColor White
Write-Host "4. Executar apenas EDA" -ForegroundColor White
Write-Host "5. Executar apenas preprocessing" -ForegroundColor White
Write-Host "6. Executar apenas spot-checking" -ForegroundColor White
Write-Host "7. Abrir Jupyter Notebook" -ForegroundColor White
Write-Host "8. Limpar dados processados" -ForegroundColor White
Write-Host "9. Sair" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Digite o numero da opcao"

switch ($choice) {
    "1" {
        Write-Host "`nVerificando ambiente..." -ForegroundColor Green
        python check_setup.py
    }
    
    "2" {
        Write-Host "`nInstalando dependencias..." -ForegroundColor Green
        pip install -r requirements.txt
        Write-Host "`nInstalacao concluida!" -ForegroundColor Green
    }
    
    "3" {
        Write-Host "`nExecutando pipeline completo..." -ForegroundColor Green
        Set-Location src
        
        Run-Step "1. Carregamento de dados" "python 01_data_loading.py"
        Run-Step "2. Analise exploratoria (EDA)" "python 02_eda.py"
        Run-Step "3. Pre-processamento" "python 03_preprocessing.py"
        Run-Step "4. Spot-checking" "python 04_spot_checking.py"
        
        Set-Location ..
        Write-Host "`nPipeline completo concluido!" -ForegroundColor Green
    }
    
    "4" {
        Write-Host "`nExecutando analise exploratoria..." -ForegroundColor Green
        Set-Location src
        python 01_data_loading.py
        python 02_eda.py
        Set-Location ..
        Write-Host "`nEDA concluida!" -ForegroundColor Green
    }
    
    "5" {
        Write-Host "`nExecutando pre-processamento..." -ForegroundColor Green
        Set-Location src
        python 03_preprocessing.py
        Set-Location ..
        Write-Host "`nPre-processamento concluido!" -ForegroundColor Green
    }
    
    "6" {
        Write-Host "`nExecutando spot-checking..." -ForegroundColor Green
        Set-Location src
        python 04_spot_checking.py
        Set-Location ..
        Write-Host "`nSpot-checking concluido!" -ForegroundColor Green
    }
    
    "7" {
        Write-Host "`nAbrindo Jupyter Notebook..." -ForegroundColor Green
        jupyter notebook
    }
    
    "8" {
        Write-Host "`nLimpando dados processados..." -ForegroundColor Yellow
        $confirm = Read-Host "Tem certeza? Isso removera todos os arquivos em data/processed/ (s/n)"
        
        if ($confirm -eq 's' -or $confirm -eq 'S') {
            Remove-Item -Path "data\processed\*" -Force -ErrorAction SilentlyContinue
            Write-Host "Dados processados removidos!" -ForegroundColor Green
        } else {
            Write-Host "Operacao cancelada." -ForegroundColor Gray
        }
    }
    
    "9" {
        Write-Host "`nSaindo..." -ForegroundColor Gray
        exit
    }
    
    default {
        Write-Host "`nOpcao invalida!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pressione qualquer tecla para sair..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
