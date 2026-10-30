# Runs hip\build-modules.ps1 for one target. Needed because the .ps1 by PATH is exec-policy
# blocked AND cmd.exe silently drops everything after an unquoted &, so the caller only ever
# pipes this file's text into Invoke-Expression. Target comes from RTC_TARGET, optional module
# subset from RTC_ONLY (comma-separated).
$ErrorActionPolicy = 'Stop'
$root = 'C:\kien\dlss5-on-amd-9070xt-porting'
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
