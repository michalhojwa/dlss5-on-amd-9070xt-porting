# Bitwise C512 fusion experiment: current direct-IO host, baseline flat-A,
# candidate modules-<Set>. 12-frame RGB per-frame hashes must match; timing ABBA 1000 frames (slots 0/3 A, 1/2 candidate).
param([string]$Set='P',[int]$Adaptive=0,[string]$BenchName='benchmark-base.exe',[string]$CandidateBenchName='benchmark-fused.exe',[string]$BaseAssets='D:\DLSSNR-Lab\zero-copy-io-20260928\assets',[string]$CandidateAssets='',[string]$Batch='correct',[string]$Base='A',[string]$Arch='gfx1201',[string[]]$Only=@(),[switch]$GameFlags,[switch]$SameSet,[switch]$CorrectnessOnly,[switch]$TimingOnly,[int]$TimingSequence=0,[int]$TimingFrames=1000,[int[]]$Heights=@(900,1080))
$ErrorActionPreference='Stop';$r='D:\DLSSNR-Lab\hip-backend';$root=Split-Path -Parent $MyInvocation.MyCommand.Path;$d="$root\runtime-regression-$Set-$Batch";$a='D:\DLSSNR-Lab\Magpie-DLSS5-AMD-0.23\DLSS5-AMD'
$p=$root
function Idle { if(Get-Process -ErrorAction SilentlyContinue | Where-Object {$_.ProcessName -match 'Shipping|^re9$|^Onimusha|^SandFall|^benchmark|^rt_bench|^microbench|^runtime-smoke|^jobbench|^Magpie'}){throw 'Game running'} }
Idle
function One($tag,$height,$seq,$candidate,$frames,$temporal=0){
 Idle;$dir="$d\$tag";New-Item -ItemType Directory -Force $dir|Out-Null
 $adaptiveLog=if($frames -eq 12){"$dir\adaptive.csv"}else{''}
 $flagSource=if($GameFlags){"$root\base-flags.txt"}else{"$a\native-game-flags.txt"}
 $flags=@(Get-Content $flagSource)+@('DLSS5_HIP_MH_FEATURE_BYTE=1','DLSS5_HIP_MH_PROJ_DIAG_FB=1','DLSS5_HIP_MH_BYTE_STREAM=1','DLSS5_HIP_DECODER_BYTE=1','DLSS5_HIP_VIT_BYTE_STREAM=0','DLSS5_HIP_VIT_STREAM=3','DLSS5_HIP_MH_FFN_FRAG256=1','DLSS5_DIRECT_IO=3','DLSS5_BENCH_PLAIN=1','DLSS5_HIP_GRAPH=0','DLSS5_SHOW_FPS=0','DLSS5_PRE_UPSCALE=0',"DLSS5_NETWORK_HEIGHT=$height","DLSS5_VIT_ADAPTIVE=$Adaptive","DLSS5_VIT_ADAPTIVE_LOG=$adaptiveLog","DLSS5_RESIDUAL_SEQUENCE=$seq","DLSS5_RESIDUAL_RGB=$(if($frames -eq 12){1}else{0})",'DLSS5_HIP_PDL=1','DLSS5_HIP_WAVE_OWNED=1','DLSS5_HIP_C512_M32=1','DLSS5_HIP_VIT_PROJ_N64=1')
 $flags += @('DLSS5_HIP_SWIN_RUN=1','DLSS5_MAKE_RESIDENT_EVERY=60','DLSS5_VIT_ADAPTIVE_IDLE_MS=1000000000')
 [IO.File]::WriteAllLines("$dir\flags.txt",$flags)
 $runner=if($candidate -and !$SameSet){"$p\$CandidateBenchName"}else{"$p\$BenchName"}
 $mods=if($candidate -and !$SameSet){"$root\flat-$Set"}else{"$root\flat-$Base"}
 if(@(Get-ChildItem $mods -Filter '*.hsaco').Count -ne 31){throw 'Incomplete flat module set'}
 Write-Output "MODULES $mods C32=$((Get-FileHash "$mods\c32-wave1.hsaco").Hash) C64=$((Get-FileHash "$mods\c64-wave2.hsaco").Hash)"

 $assets=if($candidate -and $CandidateAssets){$CandidateAssets}else{$BaseAssets}
 & $runner $assets "$dir\flags.txt" "$r\live-menu-before.f16" "$dir\rgb" $frames $temporal $mods 0 $(if($frames -eq 12){0}else{1}) 0 0 > "$dir\run.log"
 if($LASTEXITCODE){throw "Replay failed $tag"};Idle
 $rows=@(Import-Csv "$dir\rgb.csv");if($rows.Count -ne $frames -or @($rows|Where-Object{$_.checked -eq '1' -and [int]$_.invalid -ne 0}).Count){throw 'Invalid frames'}
 $mean=($rows|Where-Object{[int]$_.frame -ge $(if($frames -eq 12){1}elseif($frames -ge 1000){200}else{32})}|Measure-Object wall_ms -Average).Average
 [pscustomobject]@{tag=$tag;mean_ms=$mean;sha=(Get-FileHash "$dir\rgb.f16").Hash}|ConvertTo-Json -Compress
}
function Same($b,$t){$files=@(Get-ChildItem "$d\$b" -Filter '*frame-*.f16');if($files.Count -ne 12){throw 'Missing RGB frames'};$different=0;foreach($f in $files){if((Get-FileHash $f.FullName).Hash -ne (Get-FileHash "$d\$t\$($f.Name)").Hash){$different++}};if($different){throw "Output changed $t ($different/12)"};"SAME $b $t"}

if(!$TimingOnly){
 foreach($case in @(@{n='900-static';h=900;s=0;t=0},@{n='900-motion';h=900;s=1;t=0},@{n='1080-static';h=1080;s=0;t=0},@{n='1080-motion';h=1080;s=1;t=0},@{n='720-motion';h=720;s=1;t=0},@{n='900-history';h=900;s=5;t=1},@{n='1080-history';h=1080;s=5;t=1})){
  if($Only.Count -and $case.n -notin $Only){continue}
  foreach($c in $false,$true){One "$($case.n)-$c" $case.h $case.s $c 12 $case.t}
  Same "$($case.n)-False" "$($case.n)-True"
 }
}
if(!$CorrectnessOnly){foreach($h in $Heights){foreach($slot in 0..3){One "time-$h-$slot" $h $TimingSequence ($slot -in 1,2) $TimingFrames}}}












