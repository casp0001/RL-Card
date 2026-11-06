# Start (or, after a restart / shutdown, resume) the NFSP tuning study nfsp_v3 with 4 workers.
# Adam + gradient clipping + divergence guard (nfsp_v1/v2 Q-networks diverged to NaN); experiments\tune\nfsp_v3_space.json
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File experiments\tune\run_nfsp_v3.ps1
# Interrupted trials are detected and re-run automatically. Logs: experiments\tune\_logs\
$study = 'nfsp_v3'
$workers = 4
$cmd = "-u tune.py run --algo nfsp --study $study --trials 50 --threads 3 --budget 2000000 --space experiments\tune\nfsp_v3_space.json --startup_trials 15 --set warmup_episodes=400000 --set warmup_bot=honest"
New-Item -ItemType Directory -Force experiments\tune\_logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($i in 1..$workers) {
    Start-Process python -ArgumentList $cmd -WindowStyle Hidden `
        -RedirectStandardOutput "experiments\tune\_logs\${study}_${stamp}_w$i.out" `
        -RedirectStandardError "experiments\tune\_logs\${study}_${stamp}_w$i.err"
    Start-Sleep 3
}
Write-Host "started $workers workers for $study; progress: python tune.py status --study $study"
