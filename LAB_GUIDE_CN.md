# 本机准备 → GitHub → 实验室

当前准备范围是 **Part I**。官方来源：<https://github.com/CUHKSZ-Course/DDA5001-26Fall>，基准 commit 见 `upstream-manifest.json`。以课程手册、Blackboard 的最新要求为准。

## 可以提前完成的部分

| 内容 | 本机完成 | 实验室完成 |
| --- | --- | --- |
| model/train/plot 中 7 个 TODO | 补全并运行原始检查器 | 同一检查器复核 |
| 数据、分词器、路径 | 检查官方文件完整性，使用相对路径 | 下载后再次校验 |
| 三种优化器、保存模型、绘图、采样 | 官方小模型 CPU smoke | 小模型 CUDA smoke |
| 正式实验配置 | 准备三个默认配置，其他参数保持一致 | 查看默认曲线后确定三个其他学习率 |
| 六组正式结果、结论 | 先准备流程与报告提纲 | 正式 GPU 训练、作图、分析真实结果 |

`lab.py` 是新增的外部辅助入口；课程核心代码只填写 TODO，不改检查器、评价方法或训练预算。CPU smoke 不属于六组正式实验，也不能拿来当报告结果。

## 1. 本机验证（Windows PowerShell）

在这个仓库根目录打开终端。若 `.venv` 已存在：

```powershell
.\.venv\Scripts\python.exe lab.py verify
```

若在另一台 Windows 电脑重新准备，使用 Python 3.12 创建独立环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r p1/requirements.txt
.\.venv\Scripts\python.exe lab.py verify
```

若 CPU wheel 下载地址停滞，可改用 `python -m pip install torch==2.9.1 --index-url https://pypi.org/simple`（使用同一虚拟环境中的 Python），再运行验证。

Linux/WSL 个人电脑也可按 `p1/setup/README.md` 安装。不要把 Windows 的 `.venv` 复制到 Ubuntu。已验证版本会记录在 `LOCAL_VALIDATION.txt`；该文件与真实实验结果分开。

## 2. 个人 GitHub 私有仓库

仓库地址：<https://github.com/199823823/dda5001-final-project>。

`.gitignore` 排除虚拟环境、运行结果、checkpoint、日志和凭据。官方自带的数据随代码保留，clone 后即可校验；最终提交作业的 ZIP 仍需排除数据与 checkpoint。后续实验结果需要单独备份，不能仅依赖代码仓库。

## 3. 实验室下载后立即验证

代码无需替换 Windows 盘符：运行路径由脚本所在目录推导，子进程固定在 `p1` 内执行，输出使用相对路径。Git 已设置 `.py/.sh/.md` 等文本文件为 LF 换行，数据文件保持原始字节。Windows 虚拟环境不会上传；Linux 使用课程已安装的环境。

进入 **Ubuntu**。使用自己预约时段的账户。原项目在 `~/work/project1`，建议另建目录下载，保留原模板供对照。课程环境已安装，不执行安装脚本，也不升级 PyTorch。

```bash
source /opt/dda5001/venv/bin/activate
cd ~/work
git clone https://github.com/199823823/dda5001-final-project.git dda5001-final
cd dda5001-final
python lab.py verify --gpu
```

也可以用新增的 Linux 入口自动激活环境：在仓库根目录运行 `bash lab-linux.sh` 即执行 GPU 验证；运行 `bash lab-linux.sh train default` 即串行开始默认实验。用 `bash` 启动即可，不依赖文件执行权限。

私有仓库按 GitHub 提示认证；不要把 token 写进 URL 或脚本。也可以在已登录的 GitHub 页面 Download ZIP，解压后在含 `lab.py` 的目录运行最后一条命令。验证包括：数据/检查器哈希、原始 `check_solution.py`、真实 CUDA 运算、三种优化器的小模型训练、绘图和从 `last.pt` 采样。成功后才开始正式实验。

## 4. 六组正式实验

```bash
python lab.py train default
```

该命令串行运行三个默认配置：AdamW `0.0003`、Muon `0.01`、SGDM `0.1`，每组均从随机权重开始。查看 `p1/runs/*-default/loss_vs_tokens.png`：下降缓慢且平滑可考虑增大学习率，明显震荡或升高可考虑减小。

下面三条只是**翻倍方案的命令示例**；根据实际曲线修改数值，并把 reason 改成你的真实观察：

```bash
python lab.py choose adamw 0.0006 --reason "REPLACE_WITH_YOUR_ACTUAL_CURVE_OBSERVATION"
python lab.py choose muon 0.02 --reason "REPLACE_WITH_YOUR_ACTUAL_CURVE_OBSERVATION"
python lab.py choose sgdm 0.2 --reason "REPLACE_WITH_YOUR_ACTUAL_CURVE_OBSERVATION"
python lab.py train other
python lab.py collect
```

`choose` 会从原始正式配置复制，仅替换优化器、对应学习率、输出目录。默认和其他学习率的六次训练都保留 gpt2-medium、上下文 256、batch 72、梯度累积 1、92,160,000 tokens（5,000 次更新）、种子、调度器等固定设置。Muon 的辅助 AdamW 学习率保持原值。不要用 `lr_multiplier` 整体改变 Muon 的辅助优化器。

`train` 发现同配置已完整完成会跳过训练；发现不完整目录会报错，不会覆盖。中途失败时先保留日志，按官方 `train.py --resume` 机制处理，或将失败目录移到 `runs` 外后从头重跑；本工具不自动续训。正式试验只运行一个任务，不能同时开多个终端训练。

`collect` 校验六组 token 预算，生成各自训练/验证曲线、六组训练与验证汇总图、官方验证对比图、`p1/runs/endpoint-table.csv`（终点验证 loss、纯训练时间、墙钟时间）、AdamW 默认组终点文本和 `p1/outputs/` 中的结果 ZIP。比较终点用 `last.pt`，不要用可能位于较早时刻的 `best.pt`。

## 5. 离开前备份并核对

工作站重启会清空个人目录，**GitHub 中仅有代码不等于实验结果已备份**。将轻量结果 ZIP 复制到 U 盘；再单独复制所需 checkpoint。下面的 `/media/...` 是占位目录，先替换成你自己的真实 U 盘挂载路径：

```bash
# 在仓库根目录；确认所有训练任务已结束
tar -cf /media/YOUR_USER/YOUR_USB/dda5001-runs-backup.tar -C p1 runs
sha256sum /media/YOUR_USER/YOUR_USB/dda5001-runs-backup.tar
tar -tf /media/YOUR_USER/YOUR_USB/dda5001-runs-backup.tar | tail
```

`runs` 完整备份包含 checkpoint，文件可能较大；预留空间和传输时间。将备份复制回个人电脑后重新计算 SHA256，与实验室记录比较；结果 ZIP 还应解压测试。提交 ZIP 不含 checkpoint，备份 TAR 不作为作业提交。离开前退出个人服务账号。

## 6. 报告待填内容

主内容最多 1 页；参考文献及附录合计最多额外 2 页。只填写真实实验观察：

- 六组训练、验证曲线；六组终点验证损失、训练时间（注明口径：train_seconds 累计训练更新阶段耗时，可能包含首次编译，不含评价及 checkpoint 保存；wall_seconds 记录总运行耗时）。
- 每种优化器的学习率效果；如果已有曲线发生交叉则讨论，没有交叉不必额外训练寻找。
- 一段终点模型生成的短文本，注明优化器和学习率。
- 主要困难、解决方法、每位组员贡献。
- 将最终报告 PDF 加入结果 ZIP；每组提交一次，保留 checkpoint 供抽查。

手册标注 Part I 截止时间为 11 月 27 日 23:59；整个项目后续部分尚未包含在本次准备中。手册估计六组约 3 小时，实际耗时随硬件变化；预约时段 5 小时，须给验证和备份留时间。
