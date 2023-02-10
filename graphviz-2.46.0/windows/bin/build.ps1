Param (
    [Parameter(Mandatory=$true)]
    [ValidateSet("cmake", "msbuild")]
    [string]
    $buildsystem
    ,
    [Parameter(Mandatory=$true)]
    [ValidateSet("Release", "Debug")]
    [string]
    $configuration
    ,
    [Parameter(Mandatory=$true)]
    [ValidateSet("Win32", "x64")]
    [string]
    $platform
)

$ErrorActionPreference = "Stop"

if ($buildsystem -eq "cmake") {
    rm -Recurse -Force -ErrorAction SilentlyContinue build
    mkdir build
    cd build

    cmake -G "Visual Studio 16 2019" -A $platform ..
    if ($LastExitCode -ne 0) {
        exit $LastExitCode
    }
    cmake --build . --config $configuration
    if ($LastExitCode -ne 0) {
        exit $LastExitCode
    }

    cpack -C $configuration
    if ($LastExitCode -ne 0) {
        exit $LastExitCode
    }
} else {
    MSBuild.exe -p:Configuration=$configuration -p:Platform=$platform graphviz.sln
    if ($LastExitCode -ne 0) {
        exit $LastExitCode
    }
    if ($configuration -eq "Release") {
          rm Release\Graphviz\bin\*.lastcodeanalysissucceeded
          rm Release\Graphviz\bin\*.iobj
          rm Release\Graphviz\bin\*.ipdb
          rm Release\Graphviz\bin\*.ilk
    }
}
