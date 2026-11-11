# Pause or resume every python process whose command line mentions a study
# (e.g. the finalists workers of leduc_dqn_v1). A paused process keeps its
# memory and continues exactly where it stopped; a reboot loses it (then
# restart `python experiments\run_pipeline.py`: finished runs are kept).
#   powershell -ExecutionPolicy Bypass -File experiments\suspend.ps1 leduc_dqn_v1           # pause
#   powershell -ExecutionPolicy Bypass -File experiments\suspend.ps1 leduc_dqn_v1 -Resume   # resume
param([Parameter(Mandatory)][string]$Study, [switch]$Resume)
Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h); [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "--study $Study\b" }
foreach ($p in $procs) {
    $h = (Get-Process -Id $p.ProcessId).Handle
    $r = if ($Resume) { [W.P]::NtResumeProcess($h) } else { [W.P]::NtSuspendProcess($h) }
    "$(if ($Resume) { 'resumed' } else { 'paused' }) $($p.ProcessId) (status $r)"
}
if (-not $procs) { "no process of $Study found" }
