# Lecture 10 与 Part I 实现对照

参考：Xiao Li，DDA5001《Optimizer: Adam and Muon & Introductory Deep Learning》，用户提供的 `slides10_annotated_L01.pdf`。下文以 PDF 物理页码定位，另列幻灯片页码；手写补充页导致两种页码不同。课件用于理解和分析；正式实验遵循项目手册、代码 TODO 和固定配置。课件 PDF 没有上传到此仓库。

## 实现与公式

| 课件位置 | 要点 | 项目对应 |
| --- | --- | --- |
| PDF 第 4 页，slide 4 | 随机梯度、动量、学习率调度 | `p1/model.py` 原生 `torch.optim.SGD`；`p1/train.py` 保留课程给定的 warmup/cosine 调度 |
| PDF 第 8–11 页，slide 8–10，含手写展开 | Adam 的一阶/二阶矩指数移动平均、偏差修正、逐坐标更新 | 使用原生 `torch.optim.AdamW`，由 PyTorch 管理状态与偏差修正 |
| PDF 第 13–14 页，slide 11–12，含批注 | 向梯度加入 L2 项与解耦 weight decay 的区别 | 保留 AdamW 的 `weight_decay` 参数及参数分组，不额外把正则项加进 CE loss |
| PDF 第 19–23 页，slide 17–21 | Muon 对矩阵动量做近似正交化，采用矩阵形状缩放；实际配方使用 Nesterov 风格方向 | 隐藏层二维矩阵交给原生 Muon，保留 `adjust_lr_fn='original'`；其余参数交给辅助 AdamW；列表顺序为 `[auxiliary, muon]` |
| PDF 第 12 页，手写存储分析 | Adam 需存参数、梯度、一阶矩、二阶矩，另加激活存储 | 使用 trainer 已记录的 `peak_memory_bytes`，在汇总中换算 GiB；CPU smoke 无 GPU 显存值时留空 |
| PDF 第 34–35 页，slide 32–33 | 泛化、正则化、验证损失和 early stopping | 展示训练与验证曲线、泛化差距以及最佳记录点与终点的区别；正式比较仍使用固定预算的 `last.pt` |

### 几个容易误用的地方

1. 课件给出 Adam 常见 `beta2=0.999`，项目固定为 `beta2=0.99`。三个优化器共六组实验要使用课程配置，不能直接替换成课件的通用建议。
2. 项目固定 `weight_decay=0.0`。AdamW 的类和分组仍应正确保留；这次六组实验没有通过改变 weight decay 来比较正则化效果，报告不能把差异归因于它。
3. 课件给出 Muon 的 NS 迭代是帮助理解。作业明确使用原生 `torch.optim.Muon`，不另写 Newton–Schulz、SVD 更新或自定义优化器，也不重复施加形状缩放。保留原生调用的默认 Nesterov 与 NS 参数，课程检查器负责验收调用约束。
4. Muon 的“其他学习率”只调整 `muon_learning_rate`，不能同时调整辅助 AdamW 的 `learning_rate`。因此不用全局 `lr_multiplier` 给它们一起翻倍。
5. 课件 SGDM 的表达式把学习率写进动量递推；PyTorch 的实现累积梯度动量后用当前学习率乘更新。在学习率变化时，两种记法的递推细节不能简单视为完全等价。本项目按给定 PyTorch 实现和调度器运行。
6. 课件介绍 early stopping，但本项目要比较相同 token 预算的终点。不能提前停止其中一组，或拿它的 `best.pt` 与另一组 `last.pt` 比较。
7. 课件中的 Adam 显存估算以各张量相同存储精度为前提。实际 GPU 峰值还受激活、精度、临时缓冲、编译和优化器实现影响；汇总的实测峰值不是该简化公式的精确预测。

## 已加入的代码检查

```bash
python lab.py audit
python -m unittest discover -s tests -v
```

`audit` 核对官方参考配置的哈希，然后检查三个默认配置以及已经创建的其他学习率配置。除优化器名称、对应的学习率、独立输出目录外，其他设置必须一致；重复默认学习率和无效学习率会被拒绝。`verify`、`train`、`choose`、`collect` 也会自动执行这项检查。

`collect` 进一步核对保存的配置、终点 step、累计 token 和 CSV 记录的一致性。它不会用尚未完成的试验、不同预算或改过其他超参数的试验生成正式汇总。

## 报告可以读取的指标

六组正式实验完成后运行 `python lab.py collect`，读取 `p1/runs/endpoint-table.csv`：

- `train_loss`、`val_loss`：固定预算终点的训练、验证交叉熵。
- `generalization_gap = val_loss - train_loss`：两种估计损失的差，不能仅凭这个数就断言过拟合；应结合验证曲线走势。训练 loss 也是抽样评价值，不是全部训练集的精确均值。
- `best_val_loss`、`best_recorded_step`：所有记录的验证评价点中最低损失及其位置，不表示未记录的每一步都被评价。
- `endpoint_minus_best`：终点验证损失减去最佳记录损失；正值可提示终点表现回退，需要结合整条曲线分析。
- `train_seconds`：训练更新阶段累计时间，可能包含首次编译；评价和保存等开销另反映在 `wall_seconds` 中。
- `peak_memory_gib`：原 trainer 记录的 PyTorch GPU 峰值已分配显存，单位 GiB，不是 `nvidia-smi` 的整机显存占用。

六组训练/验证汇总图中，同一优化器使用同一种颜色；默认学习率为实线，其他学习率为虚线。实际哪个优化器更好、是否发生交叉、是否存在过拟合，必须等正式运行后根据真实数据说明。
