param([ValidateSet('freeze','pilot','run','export')][string]$Stage='export')
$ErrorActionPreference='Stop'
$challengeRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$challengePython=Join-Path $challengeRoot '.venv\Scripts\python.exe'
$challengeEntry=Join-Path $PSScriptRoot 'run.py'
if ($Stage -notin @('pilot','run')) {
    & $challengePython $challengeEntry $Stage
    exit $LASTEXITCODE
}
$challengeSecret=ConvertTo-SecureString (Get-Content -Raw -LiteralPath (Join-Path $challengeRoot '.cloudru-key.dpapi')).Trim()
$challengePreviousKey=$env:API_KEY
$challengePreviousEncoding=$env:PYTHONIOENCODING
try {
    $env:API_KEY=([pscredential]::new('cloud',$challengeSecret)).GetNetworkCredential().Password
    $env:PYTHONIOENCODING='utf-8'
    & $challengePython -u $challengeEntry $Stage
    $challengeExit=$LASTEXITCODE
} finally {
    if ($null -eq $challengePreviousKey) { Remove-Item Env:API_KEY -ErrorAction SilentlyContinue } else { $env:API_KEY=$challengePreviousKey }
    if ($null -eq $challengePreviousEncoding) { Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue } else { $env:PYTHONIOENCODING=$challengePreviousEncoding }
    $challengeSecret.Dispose()
}
exit $challengeExit
