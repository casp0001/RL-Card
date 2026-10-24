# Start (or, after a restart / shutdown, resume) the DQN tuning study dqn_v1 with 4 workers.
# Written by experiments/after_ablation.py; self-play setup: {'warmup_episodes': 200000, 'warmup_bot': 'honest'}
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File experiments\tune\run_dqn_v1.ps1
# Interrupted trials are detected and re-run automatically. Logs: experiments\tune\_logs\
$study = 'dqn_v1'
$workers = 4
$cmd = "-u tune.py run --algo dqn --study $study --trials 50 --threads 3 --startup_trials 15 --set warmup_episodes=200000 --set warmup_bot=honest"
New-Item -ItemType Directory -Force experiments\tune\_logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($i in 1..$workers) {
    Start-Process python -ArgumentList $cmd -WindowStyle Hidden `
        -RedirectStandardOutput "experiments\tune\_logs\${study}_${stamp}_w$i.out" `
        -RedirectStandardError "experiments\tune\_logs\${study}_${stamp}_w$i.err"
    Start-Sleep 3
}
Write-Host "started $workers workers for $study; progress: python tune.py status --study $study"
