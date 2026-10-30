【已过时 2026-10-03】不用再换模块切档了。快速数值路径现在是运行时选项：
  在游戏的 DLSS5-AMD\native-game-flags.txt 里加一行 DLSS5_FAST_NUMERIC=1（删掉或写 0 就是逐位档）；
  想要以前 fast 档的 1088 行，再加一行 DLSS5_NETWORK_1080_ROWS=1088。改完重开游戏。
  to-fast.ps1 现在会拒绝执行；to-exact.ps1 / status.ps1 仍可用（用来退出旧的 fast 档）。

DLSS5-AMD fast 档试玩说明（2026-10-01）

现在游戏里装的是：逐位档（和以前一样）。

【怎么切】先关掉游戏，然后在 PowerShell 里运行：
  切到 fast 档：  powershell -ExecutionPolicy Bypass -File D:\DLSSNR-Lab\fast-tier\to-fast.ps1
  切回逐位档：    powershell -ExecutionPolicy Bypass -File D:\DLSSNR-Lab\fast-tier\to-exact.ps1
  只看当前档位：  powershell -ExecutionPolicy Bypass -File D:\DLSSNR-Lab\fast-tier\status.ps1
剑星和鬼武者一起切。最后一行会打印"当前档位: FAST / EXACT"。
游戏开着时脚本会拒绝执行。

【fast 档是什么】
网络里少做几次精度舍入，1080 档改跑 1088 行（和 mochizuki/Daniel 一样）。
离线测试：900 档快约 0.11ms，1080 档快约 0.6ms；画面和逐位档相差约 52~55dB（肉眼应该看不出）。

【看什么】同一个场景两档各玩一会儿：
  1. 运动时有没有闪烁，或者细碎的噪点跳动（草、头发、栅栏、远处的细线）
  2. 快速转镜头时有没有拖影或者重影
  3. 细节：文字、纹理边缘是否变糊；画面最下方几行有没有异常
  4. 帧率：左上角 FPS，或者 DLSS5-AMD\logs\frame-stats.txt（只有剑星开了）

【出问题怎么办】
关掉游戏，运行 to-exact.ps1，看到"当前档位: EXACT"就是原样了。
每次切换都会自动备份到 D:\DLSSNR-Lab\fast-tier\backups\。
