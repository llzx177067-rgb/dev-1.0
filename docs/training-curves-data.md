# 训练曲线数据与复现

这些图来自 dev-1.0 首轮训练的真实事件日志，包含全部 **2,448 次更新**与 **11 个开发评估点**。没有补造缺失点，也没有重新训练。

- `training-history.json`：脱敏后的原始数值、源日志与训练数据 SHA256。
- `training-history.csv`：每次更新的 step、soft-label CE、learning rate、样本数与 input tokens，共 2,448 行。
- `development-history.csv`：11 次开发评估的 accuracy、calibrated NLL/ECE/Brier、保存温度与 best 更新记录。
- `loss-curves.png` / `.svg`：训练 CE 全部原值及固定 50 步后向均线；开发 NLL 的 11 个原值点。
- `training-metrics.png` / `.svg`：开发 accuracy、ECE、Brier 与温度。
- `training-plot-manifest.json`：输入、脚本、图与 CSV 的 SHA256 和绘图设置。

在安装 Matplotlib 3.10.0 的独立绘图环境中，从仓库根目录运行：

```bash
python docs/plot_training_curves.py
```

绘图不加载模型，不执行 GPU 推理，也不改变训练环境。PNG 与 SVG 同时生成，GitHub 可以直接嵌入 PNG。

训练 CE 是每次有效 batch 的实际软标签损失，不是整轮平均值；最后一次更新只有 114 条，其余更新为 256 条。开发 NLL 使用开发硬标签，二者不是同一损失总体。淡色原值线展示全部训练更新，深色线仅为 50 个真实更新的后向算术均值，从第 50 步开始，不填充或外推。

开发曲线只在真实评估步骤 0、250、500、750、1000、1250、1500、1750、2000、2250、2448 标记测量值，连接线仅辅助阅读，不代表新增测量。开发集含 1,497 条记录，温度在单独的 983 条记录上拟合；开发集参与 checkpoint 选择。因此，这些曲线不能作为独立测试效果或 BFCL 训练曲线。

数据源日志 SHA256：`7cc393b6ab0dfb158d4d4c5c2acc5ba91e90c6adca2b82fa801b520d0784bed2`。导出的数据与绘图标题不包含机器、账号或服务器路径。
