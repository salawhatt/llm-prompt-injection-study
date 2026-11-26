$ErrorActionPreference='Stop'
$challengeRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$challengePython=Join-Path $challengeRoot '.venv\Scripts\python.exe'
Push-Location $challengeRoot
try {
    & $challengePython -X utf8 (Join-Path $PSScriptRoot 'continuation.py') export
    if ($LASTEXITCODE -ne 0) { throw 'Ledger export failed' }
    foreach ($challengePart in @('','supplement','robustness')) {
        $challengeResult=Join-Path $challengeRoot ('results/hard-challenge-2026-11-26/'+$challengePart)
        if ($challengePart -eq 'robustness') {
            # First-response figures are already present in the main directory.
            & $challengePython -X utf8 (Join-Path $PSScriptRoot 'analyze.py') --directory $challengeResult --no-plots
        } else {
            & $challengePython -X utf8 (Join-Path $PSScriptRoot 'analyze.py') --directory $challengeResult
        }
        if ($LASTEXITCODE -ne 0) { throw ('Analysis failed: '+$challengePart) }
    }
    & $challengePython -X utf8 (Join-Path $PSScriptRoot 'post_analysis.py')
    if ($LASTEXITCODE -ne 0) { throw 'Synthesis failed' }
    & $challengePython -X utf8 (Join-Path $PSScriptRoot 'validate_artifacts.py')
    if ($LASTEXITCODE -ne 0) { throw 'Artifact validation failed' }
} finally {
    Pop-Location
}
