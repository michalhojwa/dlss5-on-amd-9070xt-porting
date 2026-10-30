# Runs hip\build-modules.ps1 for one target. Needed because the .ps1 by PATH is exec-policy
# blocked AND cmd.exe silently drops everything after an unquoted &, so the caller only ever
# pipes this file's text into Invoke-Expression. Target comes from RTC_TARGET, optional module
# subset from RTC_ONLY (comma-separated).
$ErrorActionPolicy = 'Stop'
# This file is piped into Invoke-Expression, so $PSScriptRoot is empty here. Locate the repo
# root by walking up from the caller's CWD looking for hip\build-modules.ps1; RTC_ROOT overrides.
$root = $env:RTC_ROOT
if (-not $root) {
    $d = (Get-Location).Path
    while ($d -and -not (Test-Path (Join-Path $d 'hip\build-modules.ps1'))) {
        $p = Split-Path -Parent $d
        if ($p -eq $d) { break }
        $d = $p
    }
    $root = $d
}
if (-not (Test-Path (Join-Path $root 'hip\build-modules.ps1'))) {
    throw 'cannot locate hip\build-modules.ps1 from CWD - run from the repo root or set RTC_ROOT'
}
if (-not $env:RTC_TARGET) { throw 'set RTC_TARGET (gfx1150/gfx1151/gfx1152/...)' }
$src = Get-Content -LiteralPath (Join-Path $root 'hip\build-modules.ps1') -Raw
$env:RTC_NO_ASM = '1'   # probe_comgr.py --compile would otherwise dump ~0.5 GB of assembly
$only = @()
if ($env:RTC_ONLY) { $only = @($env:RTC_ONLY -split ',') }
$rtc = [ScriptBlock]::Create($src)
& $rtc -Targets $env:RTC_TARGET -SkipOnError -ExtraDefines 'CW_PACK_MODE_MASK 0' `
   -Only $only `
   -OutputDir (Join-Path (Join-Path $root 'hip\modules') $env:RTC_TARGET) `
   -SourceDir (Join-Path $root 'hip') `
   -Compiler (Join-Path $root 'xbox\rtc_compile_cmd.cmd')
$built = @(Get-ChildItem -LiteralPath (Join-Path (Join-Path $root 'hip\modules') $env:RTC_TARGET) -Filter '*.hsaco' -ErrorAction SilentlyContinue)
Write-Output "BUILT $($built.Count) .hsaco for $($env:RTC_TARGET)"
