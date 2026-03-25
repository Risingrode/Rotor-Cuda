# Rotor-Cuda v2.0

这是 KeyHunt v1.7 的一个修改版本……
衷心感谢所有其代码被用于此项目的开发者。
同时，也感谢那些允许在其服务器上构建和测试 Rotor-Cuda 第 2 版的朋友们。

Telegram  **https://t.me/CryptoCrackersUK**

## 变更：
- 默认随机范围：95% 为 (252-256) bit，5% 为 (248-252) bit
- 可在指定 bit 范围内随机扫描 (1-256)
- 可在给定 bit 区间之间随机扫描：`-n ? -z ?`
- 根据指定的命令参数自动创建 `Rotor-Cuda_START.bat`
- 许多细小的可视化改进

### GPU 扫描说明
- **-n ?** 每隔 ? 分钟保存一次检查点。(1-10000)
- 如果不指定 `-n ?`，则不会启用断点续扫（一次性搜索）。
- 当 `Rotor-Cuda_Continue.bat` 文件出现后，你可以从上一个检查点继续。
- 若要正确续扫，请不要修改该文件中的参数。
- **如果你不需要继续，请删除 `Rotor-Cuda_Continue.bat`！！！**
---
### 随机模式使用 `-r 5`（GPU）
- **-r ?** 输入 `-r 5` 表示最多允许扫描 50 亿个密钥，之后会改为扫描随机选取的区段。
- **-n ?** `(1-256) bit`。如果不指定 `-n`，默认使用 95% 的 `(252-256) bit` + 5% 的 `(248-252) bit`
- **-z ?** （随机范围结束值必须大于 `-n` 的值）例如：`-n 252 -z 256`
- 用于搜索 [puzzle](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx) 71 的随机示例：

- `./Rotor -g --gpui 0 --gpux 256,256 -m address --coin BTC -r 5 --range 400000000000000000:7fffffffffffffffff 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU`
- 如果你的 GPU 性能低于 RTX 1080，或者驱动发生崩溃，请从命令中移除 **--gpux 256,256**，网格大小将自动分配。
---
### GPU Bitcoin 多地址模式：
- 范围扫描：`./Rotor -g --gpui 0 --gpux 256,256 -m addresses --coin BTC --range 400000000:7ffffffff -i Btc-h160.bin`
- 随机扫描：`./Rotor -g --gpui 0 --gpux 256,256 -m addresses --coin BTC -r 5 -i Btc-h160.bin`
---
### GPU Bitcoin 单地址模式：
- 范围扫描：`./Rotor -g --gpui 0 --gpux 256,256 -m address --coin BTC --range 400000000:7ffffffff 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb`
- 随机扫描：`./Rotor -g --gpui 0 --gpux 256,256 -m address --coin BTC --range 400000000:7ffffffff -r 5 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb`
---
### GPU ETHEREUM 多地址模式：
- 范围扫描：`./Rotor -g --gpui 0 --gpux 256,256 -m addresses --coin eth --range 4000000:ffffffff -i eth.bin`
- 随机扫描：`./Rotor -g --gpui 0 --gpux 256,256 -m addresses --coin eth --rang 4000000:ffffffff -r 5 -i eth.bin`
---
### GPU ETHEREUM 单地址模式：
- 范围扫描：`./Rotor -g --gpui 0 --gpux 256,256 -m address --coin eth --range 8000000:fffffff 0xfda5c442e76a95f96c09782f1a15d3b58e32404f`
- 随机扫描：`./Rotor -g --gpui 0 --gpux 256,256 -m address --coin eth --range 8000000:fffffff -r 5 0xfda5c442e76a95f96c09782f1a15d3b58e32404f`

****🕹不要使用非标准的 Grid Size 值，必须是 32 的倍数****

   |      GPU 型号      |      扫描速度      |    Grid Size    |
   |-------------------|:------------------:|:---------------:|
   |    Tesla T4       |     600 Mkeys      |     128×256     |
   |    RTX 3090       |     1.4 Gkeys      |     256×256     |
   |    RTX 4090       |     3.1 Gkeys      |     256×512     |
   |    RTX 5090       |     6.2 Gkeys      |     512×512     |
   |    RTX 60xx       |     x.x Gkeys      |     xxx.xxx     |


---
### 顺序范围扫描：
```
$./Rotor -g --gpui 0 --gpux 256,256 -m address --coin BTC --range 1000000000:1fffffffff 14iXhn8bGajVWegZHJ18vJLHhntcpL4dex

  Rotor-Cuda v2.0  Mehdi256

  COMP MODE    : COMPRESSED
  COIN TYPE    : BITCOIN
  SEARCH MODE  : Single Address
  DEVICE       : GPU
  GPU IDS      : 0
  GPU SIZE     : 256x256
  SSE          : YES
  BTC ADDRESS  : 14iXhn8bGajVWegZHJ18vJLHhntcpL4dex
  OUTPUT FILE  : Found.txt

  Start Time   : Sun Sep 21 06:51:02 2025

  Global start : 1000000000 (37 bit)
  Global end   : 1FFFFFFFFF (37 bit)
  Global range : FFFFFFFFF (36 bit)

  GPU Mode     : GPU #0 NVIDIA GeForce RTX 4090 (128x0 cores) Grid (256x256)

  Rotor info   : Divide the range FFFFFFFFF (278 bit) into GPU 65536 threads

  Thread 00000 : 1000000000 -> 10000FFFFF
  Thread 00001 : 10000FFFFF -> 10001FFFFE
  Thread 00002 : 10001FFFFE -> 10002FFFFD
  Thread 00003 : 10002FFFFD -> 10003FFFFC
  Thread 65534 : 1FFFDF0002 -> 1FFFEF0001
  Thread 65535 : 1FFFEF0001 -> 1FFFFF0000
  Thread 65536 : 1FFFFF0000 -> 20000EFFFF

  [00:00:07] [15D98444CC] [F: 0] [00:01:07] [C: 36.132813 %] [GPU: 3.60 Gk/s] [T: 24,830,279,680]
  ================================================================================================
  PubAddress: 14iXhn8bGajVWegZHJ18vJLHhntck8EEPa
  Priv (WIF): KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9NRuiZFAX5a6P5M
  Priv (HEX): 1757756A93
  PubK (HEX): 027D2C03C3EF0AEC70F2C7E1E75454A5DFDD0E1ADEA670C1B3A4643C48AD0F1255

  [00:00:07] [16008246F6] [F: 1] [00:01:06] [C: 37.109375 %] [GPU: 3.61 Gk/s] [T: 25,501,368,320]

```
### 随机范围扫描：
```
$./Rotor -g --gpui 0 --gpux 256,256 -m address --coin BTC --range 1000000000:1fffffffff -r 5 14iXhn8bGajVWegZHJ18vJLHhntcpL4dex

  Rotor-Cuda v2.0  Mehdi256

  COMP MODE    : COMPRESSED
  COIN TYPE    : BITCOIN
  SEARCH MODE  : Single Address
  DEVICE       : GPU
  GPU IDS      : 0
  GPU SIZE     : 256x256
  SSE          : YES
  BTC ADDRESS  : 14iXhn8bGajVWegZHJ18vJLHhntcpL4dex
  OUTPUT FILE  : Found.txt

  Start Time   : Sun Sep 21 07:05:02 2025

  GPU Mode     : GPU #0 NVIDIA GeForce RTX 4090 (128x0 cores) Grid (256x256)
  Base Key     : Randomly changes 65536 start Private keys every 5,000,000,000 on the counter
  ROTOR Random : Min 37 (bit) 1000000000
  ROTOR Random : Max 37 (bit) 1FFFFFFFFF

  |00:00:27| R : 17 | 17E005E2C7 | F : 0 | GPU: 3.57 Gk/s | T: 95,428,804,608 |
  ================================================================================================
  PubAddress: 14iXhn8bGajVWegZHJ18vJLHhntck8EEPa
  Priv (WIF): KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9NRuiZFAX5a6P5M
  Priv (HEX): 1757756A93
  PubK (HEX): 027D2C03C3EF0AEC70F2C7E1E75454A5DFDD0E1ADEA670C1B3A4643C48AD0F1255

  |00:00:27| R : 17 | 17E00605C5 | F : 1 | GPU: 3.57 Gk/s | T: 96,099,893,248 |

```
### 多地址范围扫描：
```
$./Rotor -g --gpui 0 --gpux 256,256 -m addresses --coin BTC --range 1000000000:1fffffffff -r 5 -i Puzzles_h160.bin

  Rotor-Cuda v2.0  Mehdi256

  COMP MODE    : COMPRESSED
  COIN TYPE    : BITCOIN
  SEARCH MODE  : Multi Address
  DEVICE       : GPU
  GPU IDS      : 0
  GPU SIZE     : 256x256
  SSE          : YES
  BTC HASH160s : Puzzles_h160.bin
  OUTPUT FILE  : Found.txt

  Loading      : 160 %
  Loaded       : 160 Bitcoin addresses

  Bloom at     : 0x5b675c9509a0
  Version      : 2.1
  Entries      : 322
  Error        : 0.0000010000
  Bits         : 9259
  Bits/Elem    : 28.755175
  Bytes        : 1158 (0 MB)
  Hash funcs   : 20

  Site         : https://github.com/Mehdi256/Rotor-Cuda
  Donate       : bc1qdfaj5zyvfkr7wtzaa72vqxzztpl2tz7g5zk5ug

  Start Time   : Sun Sep 21 18:06:29 2025

  GPU Mode     : GPU #0 NVIDIA GeForce RTX 4090 (128x0 cores) Grid (256x256)
  Base Key     : Randomly changes 65536 start Private keys every 5,000,000,000 on the counter
  ROTOR Random : Min 37 (bit) 1000000000
  ROTOR Random : Max 37 (bit) 1FFFFFFFFF

  |00:00:18| R : 15 | 167ABCD68F | F : 0 | GPU: 3.53 Gk/s | T: 81,604,378,624 |
  ================================================================================================
  PubAddress: 14iXhn8bGajVWegZHJ18vJLHhntcnHyduN
  Priv (WIF): KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9NRuiZFAX5a6P5M
  Priv (HEX): 1757756A93
  PubK (HEX): 027D2C03C3EF0AEC70F2C7E1E75454A5DFDD0E1ADEA670C1B3A4643C48AD0F1255

  |00:00:18| R : 15 | 16F55BF6A1 | F : 1 | GPU: 3.53 Gk/s | T: 82,328,530,944 |

```
### 使用多 GPU 扫描 Puzzle 71：
```
$./Rotor -g --gpui 0,1,2 --gpux 256,256,256,256,256,256 -m address --coin BTC --range 400000000000000000:7fffffffffffffffff -r 15 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU

  Rotor-Cuda v2.0  Mehdi256

  COMP MODE    : COMPRESSED
  COIN TYPE    : BITCOIN
  SEARCH MODE  : Single Address
  DEVICE       : GPU
  GPU IDS      : 0, 1, 2
  GPU SIZE     : 256x256, 256x256, 256x256
  SSE          : YES
  BTC ADDRESS  : 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
  OUTPUT FILE  : Found.txt

  Start Time   : Tue Sep 21 16:58:42 2025

  GPU Mode     : GPU #0 NVIDIA GeForce RTX 4090 (128x0 cores) Grid (256x256)
  Base Key     : Randomly changes 262144 start Private keys every 15,000,000,000 on the counter
  ROTOR Random : Min 71 (bit) 400000000000000000
  ROTOR Random : Max 71 (bit) 7FFFFFFFFFFFFFFFFF

  GPU Mode     : GPU #1 NVIDIA GeForce RTX 4090 (128x0 cores) Grid (256x256)
  Base Key     : Randomly changes 262144 start Private keys every 15,000,000,000 on the counter
  ROTOR Random : Min 71 (bit) 400000000000000000
  ROTOR Random : Max 71 (bit) 7FFFFFFFFFFFFFFFFF

  GPU Mode     : GPU #2 NVIDIA GeForce RTX 4090 (128x0 cores) Grid (256x256)
  Base Key     : Randomly changes 262144 start Private keys every 15,000,000,000 on the counter
  ROTOR Random : Min 71 (bit) 400000000000000000
  ROTOR Random : Max 71 (bit) 7FFFFFFFFFFFFFFFFF

  |00:24:09| R : 1250 | 61AE654C8F21375303 | F : 0 | GPU: 10.96 Gk/s | T: 18,874,233,782,272 |

```
# Linux
- **🕹注意：此版本不要编辑 `Makefile`（不要修改 `Makefile`）**

- 更新并安装 `libgmp`：
  - `sudo apt update`
  - `sudo apt install -y libgmp-dev`

- CUDA = `/usr/local/cuda-10.0 ~ 12.xx`（暂时不要使用 CUDA 13，只支持 10_12.xx 版本）

- CXXCUDA = `/usr/bin/g++`

- 构建仅 CPU 版本：

  ```sh
  make all
  ```
- 使用 CUDA（GPU）构建：

- 查看各种 Nvidia GPU 的 CCAP 值信息：

  https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards

  ```sh
   cd Rotor-Cuda

   make gpu=1 CCAP=75 all     [SM_75]

   make gpu=1 CCAP=89 all     [SM_89]

   make gpu=1 CCAP=100 all    [SM_100]
  ```

## 许可证
- Rotor-Cuda 采用 GPLv3.0 许可证

## 捐赠
- BTC: bc1qdfaj5zyvfkr7wtzaa72vqxzztpl2tz7g5zk5ug
