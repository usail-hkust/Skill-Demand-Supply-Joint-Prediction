$ErrorActionPreference = "Stop"

$dir = $PSScriptRoot

Invoke-Expression "$dir\setup-build-utilities.ps1"

powershell.exe -ExecutionPolicy Bypass -Command "$dir\build.ps1" $args
if ($LastExitCode -ne 0) {
    exit $LastExitCode
}
