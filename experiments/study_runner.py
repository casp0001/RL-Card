''' Helpers for the unattended experiment scripts: start / resume a tuning
study with 4 hidden workers (writing experiments/tune/run_<study>.ps1, the
exact command to resume it after a restart) and wait until it has finished. '''
import os
import sys
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import tune  # noqa: E402


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M'), msg, flush=True)


def study_state(study):
    if not os.path.isfile(tune.config_path(study)):
        return None, 0, 0
    s = tune.load_study(study)
    return s, tune.count_finished(s), tune.count_running(s)


def count_processes(pattern):
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like "
                          f"'{pattern}' }}).Count"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    return int(out or 0)


def workers_alive(study):
    return count_processes(f'*tune.py run*--study {study} *')


def start_study(study, algo, trials, extra='', sets=None, workers=4, threads=3, note=''):
    sets = sets or {}
    set_args = ' '.join(f'--set {k}={v}' for k, v in sets.items())
    script = os.path.join(REPO, 'experiments', 'tune', f'run_{study}.ps1')
    with open(script, 'w', encoding='utf-8') as f:
        f.write(f"""# Start (or, after a restart / shutdown, resume) the {algo.upper()} tuning study {study} with {workers} workers.
# {note}
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File experiments\\tune\\run_{study}.ps1
# Interrupted trials are detected and re-run automatically. Logs: experiments\\tune\\_logs\\
$study = '{study}'
$workers = {workers}
$cmd = "-u tune.py run --algo {algo} --study $study --trials {trials} --threads {threads} {extra} {set_args}"
New-Item -ItemType Directory -Force experiments\\tune\\_logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($i in 1..$workers) {{
    Start-Process python -ArgumentList $cmd -WindowStyle Hidden `
        -RedirectStandardOutput "experiments\\tune\\_logs\\${{study}}_${{stamp}}_w$i.out" `
        -RedirectStandardError "experiments\\tune\\_logs\\${{study}}_${{stamp}}_w$i.err"
    Start-Sleep 3
}}
Write-Host "started $workers workers for $study; progress: python tune.py status --study $study"
""")
    log(f'starting {study}: {algo}, {trials} trials, {set_args or "plain self-play"} {extra}')
    subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script], check=False, cwd=REPO)


def run_study_to_end(study, algo, trials, **start_kwargs):
    ''' Start (or resume) a study and wait until it has finished. '''
    _, finished, running = study_state(study)
    if finished >= trials and running == 0:
        log(f'{study} already finished')
        return
    if workers_alive(study) == 0:
        start_study(study, algo, trials, **start_kwargs)
    while True:
        time.sleep(300)
        _, finished, running = study_state(study)
        log(f'{study}: {finished}/{trials} finished, {running} running')
        if finished >= trials and running == 0:
            log(f'{study} finished')
            return
        if workers_alive(study) == 0:
            log(f'{study}: no workers left at {finished}/{trials}; starting them again')
            start_study(study, algo, trials, **start_kwargs)
