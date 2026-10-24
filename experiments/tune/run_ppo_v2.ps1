# Start (or, after a restart / shutdown, resume) the PPO tuning study ppo_v2 with 4 workers:
# the ppo_v1 search space with the self-play variant chosen by the ablation (warmup_honest).
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File experiments\tune\run_ppo_v2.ps1
# Interrupted trials are detected and re-run automatically. Logs: experiments\tune\_logs\
$study = 'ppo_v2'
$workers = 4
$cmd = "-u tune.py run --algo ppo --study $study --trials 50 --threads 3 --set warmup_episodes=200000 --set warmup_bot=honest"
New-Item -ItemType Directory -Force experiments\tune\_logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($i in 1..$workers) {
    Start-Process python -ArgumentList $cmd -WindowStyle Hidden `
        -RedirectStandardOutput "experiments\tune\_logs\${study}_${stamp}_w$i.out" `
        -RedirectStandardError "experiments\tune\_logs\${study}_${stamp}_w$i.err"
    Start-Sleep 3
}
Write-Host "started $workers workers for $study; progress: python tune.py status --study $study"
