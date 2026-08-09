$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = 'C:\Users\29465\AppData\Local\Programs\Python\Python310\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = 'python'
}

Write-Host 'Preparing PhysCrop-Risk company demo data...'
& $python scripts\prepare_company_demo.py
if ($LASTEXITCODE -ne 0) {
    throw 'Demo preparation failed'
}

Write-Host 'Starting PhysCrop-Risk alerting workspace...'
Write-Host 'Browser will open http://127.0.0.1:8765 shortly'
Write-Host 'Press Ctrl+C in this window to stop the server'

Start-Job -ScriptBlock { Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:8765' } | Out-Null
& $python scripts\run_physcrop_agent_v2_server.py --demo

Get-Job | Remove-Job -Force
