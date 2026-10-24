# Start (or, after a restart / shutdown, resume) the PPO tuning study with 4 workers.
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File experiments\tune\run_ppo_v1.ps1
# Interrupted trials are detected and re-run automatically. Logs: experiments\tune\_logs\
$study = 'ppo_v1'
$workers = 4
$cmd = "-u tune.py run --algo ppo --study $study --trials 40 --threads 3"
New-Item -ItemType Directory -Force experiments\tune\_logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($i in 1..$workers) {
    Start-Process python -ArgumentList $cmd -WindowStyle Hidden `
        -RedirectStandardOutput "experiments\tune\_logs\${study}_${stamp}_w$i.out" `
        -RedirectStandardError "experiments\tune\_logs\${study}_${stamp}_w$i.err"
    Start-Sleep 3
}
Write-Host "started $workers workers for $study; progress: python tune.py status --study $study"
