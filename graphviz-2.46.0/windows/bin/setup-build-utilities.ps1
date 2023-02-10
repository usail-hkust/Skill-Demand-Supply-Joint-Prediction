# This script tries to find specific build utilities in the user's
# path and if it doesn't find one falls back to set up the path to the
# utility in the Graphviz build utilities submodule at
# windows\dependencies\graphviz-build-utilities.
#
# It should be sourced from a PowerShell script like so:
#
# Set-ExecutionPolicy Bypass -Force -Scope Process
# setup-build-utilities.ps1
#

$ErrorActionPreference = "Stop"

$GRAPHVIZ_WINDOWS_BIN = $PSScriptRoot
$GRAPHVIZ_WINDOWS = Split-Path $GRAPHVIZ_WINDOWS_BIN -Parent
$GRAPHVIZ_ROOT = Split-Path $GRAPHVIZ_WINDOWS -Parent

$VS_BUILD_TOOLS = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
$CMAKE_BIN = "$VS_BUILD_TOOLS\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$MSBUILD_BIN = "$VS_BUILD_TOOLS\MSBuild\Current\Bin"

$exit_status = 0

function find_or_fallback($programs, $fallback_path) {

    $script:all_programs = "$script:all_programs $programs"

    $not_found = ""
    $programs.Split(" ") | ForEach {
        $program = $_
        if ($exe = (& Invoke-Expression "Get-Command -EA Continue $program.exe" 2>$null).Source) {
            echo "Found $program at $exe"
            $found = "$found $program"
        } else {
            if ("$not_found" -ne "") {
                $not_found = "$not_found, $program"
            } else {
                $not_found = $program
            }
        }
    }

    if ("$not_found" -ne "") {
        echo "Fallback needed for: $not_found"
        echo "Setting up fallback path for: $programs to $fallback_path"
        $env:Path = $fallback_path + ";" + $env:Path
    }
}

$build_utilities_path = "$GRAPHVIZ_ROOT\windows\dependencies\graphviz-build-utilities"

find_or_fallback "awk sed swig" "$build_utilities_path"
find_or_fallback "win_bison win_flex" "$build_utilities_path\winflexbison"
find_or_fallback "makensis" "$build_utilities_path\NSIS\Bin"
find_or_fallback "perl" "$build_utilities_path\Perl64\Bin"
find_or_fallback "python3" "$build_utilities_path\Python38-32"
find_or_fallback "pytest" "$build_utilities_path\Python38-32\Scripts"
find_or_fallback "diff grep" "$build_utilities_path\GnuWin\bin"
find_or_fallback "cmake cpack" "$CMAKE_BIN"
find_or_fallback "msbuild" "$MSBUILD_BIN"

if (-NOT (cpack.exe --help | grep 'CPACK_GENERATOR')) {
    echo "Moving $CMAKE_BIN to front of PATH in order to find CMake's cpack"
    $CMAKE_BIN_ESCAPED = echo $CMAKE_BIN | sed 's#\\#\\\\#g'
    $path = (Invoke-Expression 'echo $Env:Path' | sed "s#;$CMAKE_BIN_ESCAPED##")
    $Env:Path="$CMAKE_BIN;$path"
}

$ErrorActionPreference = "Continue"
if (-NOT (sort.exe /? 2>$null | grep "SORT")) {
    $ErrorActionPreference = "Stop"
    echo "Moving C:\WINDOWS\system32 to front of PATH in order to find Windows' sort"
    $path = (Invoke-Expression 'echo $Env:Path' | sed 's#;C:\\WINDOWS\\system32;#;#')
    $Env:Path="C:\WINDOWS\system32;$path"
}
$ErrorActionPreference = "Stop"

$script:all_programs += " sort"


echo "Final check where all utilites are found:"

$script:all_programs.Trim().Split(" ") | ForEach {
    $program = $_
    if ($result = (Invoke-Expression "Get-Command -EA Continue $program.exe" 2>$null).Source) {
        echo $result
    } else {
        Write-Error -EA Continue "Fatal error: $program still not found"
        $exit_status=1
    }
}

# Special checks

if (-NOT (cpack.exe --help | grep 'CPACK_GENERATOR')) {
    $exe = (Get-Command cpack.exe 2>$null).Source
    Write-Error -EA Continue "Found an unknown cpack at $exe"
    $exit_status = 1
}

$ErrorActionPreference = "Continue"
if (-NOT (sort.exe /? 2>$null | grep "SORT")) {
    $ErrorActionPreference = "Stop"
    $exe = (Get-Command sort.exe 2>$null).Source
    Write-Error -EA Continue "Found an unknown sort at $exe"
    $exit_status = 1
}
$ErrorActionPreference = "Stop"

if ($exit_status -eq 0) {
    echo "All utilities have been found. Happy building!"
} else {
    Write-Error -EA Continue "Some utilities were not found"
}

exit $exit_status
