# dev-1.0

**面向 Agent 的轻量级工具选择与概率决策模型。**

dev-1.0 基于 Qwen3.5-0.8B-Base，将用户请求、上下文和候选选项转换为选择结果及概率分布，可接入 Agent 的工具选择、候选路由和分类流程。在同一套 BFCL 工具名选择评估中，整题准确率达到 **77.39%**，较官方后训练版 Qwen3.5-0.8B 的 **39.37%** 提高 **38.02 个百分点**。

[模型下载](https://modelscope.cn/models/lzx1770670957/dev-1.0) · [数据集](https://modelscope.cn/datasets/lzx1770670957/dev-data-1.0) · [GitHub](https://github.com/llzx177067-rgb/dev-1.0)

## 能力亮点

- **多候选工具选择。** 在 BFCL `live_multiple` 子集中，整题准确率从 **0.96% 提高到 65.28%**，改善多个候选工具之间的选择。
- **减少无关工具选择。** 在本次 BFCL 对照中，不必要工具选择率从 **97.30% 降至 15.06%**，必要工具召回率为 **96.75%**。
- **直接输出决策概率。** 返回候选概率、选中答案和置信度，无需生成和解析解释文本，便于接入现有 Agent 流程。
- **适配有限候选任务。** 支持最多 **255 个候选**和 **8,192 token** 总输入，可用于工具选择、路由和分类；处理文本与可选图像条件。
- **0.8B 骨干与量化运行。** 使用 NF4 4-bit 语言骨干、LoRA adapter 和独立 FP32 决策头；推理通过配套 CUDA 加载器完成。

## BFCL 评估

评估采用 **Decision Index 0.2.1 的 BFCL 工具名选择投影**，共 1,694 题、4,768 个判断字段。一题中全部候选工具判断正确才得分，两模型均完成全部请求。

| 指标 | 官方 Qwen3.5-0.8B | dev-1.0 |
| --- | ---: | ---: |
| 整题准确率 | 39.37% | **77.39%** |
| 多候选工具子集准确率 | 0.96% | **65.28%** |
| 预测调用精确率 | 36.14% | **77.97%** |
| 不必要工具选择率，越低越好 | 97.30% | **15.06%** |
| 必要工具召回率 | **99.94%** | 96.75% |

本结果衡量工具名选择，不包含参数生成、调用顺序和执行。对照使用官方后训练版 Qwen3.5-0.8B；dev-1.0 从 Base 训练，使用专用决策头和训练模板，因此分差包含模型起点与推理接口差异。单工具子集 `simple` 和 `live_simple` 的准确率分别为 99.75% 和 91.47%，低于对照的 100%；完整条件与逐子集数据见[评估报告](docs/bfcl-report.md)。

## 使用方式

输入由状态、问题和显式候选组成，输出为结构化答案。例如，用户询问加法而候选工具是天气查询时，发布模型的玩具样例输出如下：

```json
{
  "type": "choice",
  "choice": "no",
  "probabilities": {
    "yes": 0.004156,
    "no": 0.995844
  }
}
```

这是随仓库提供的单条示例输出，数值按显示精度四舍五入。自定义候选及完整输入见 [`examples/decision.json`](https://github.com/llzx177067-rgb/dev-1.0/blob/main/examples/decision.json)。

下载模型、固定版本的 `Qwen/Qwen3.5-0.8B-Base` 并安装配套环境后，在源码仓库运行：

```bash
python tools/predict.py \
  --model-dir /path/to/dev-1.0 \
  --base-model-path /path/to/Qwen3.5-0.8B-Base \
  --input examples/decision.json \
  --output result.json
```

发布格式为 **QLoRA adapter + 决策头 + processor**，需使用仓库的自定义加载器和固定 Base；输出用于候选决策，工具参数生成与执行交由 Agent 完成。加载器会恢复保存的温度并应用一次。

## 框架与训练

模型采用专用 255 路决策 readout，在 Qwen3.5 的 6 层 full attention 中开放非因果上下文，保留 18 层 GatedDeltaNet 的原始递推行为。NF4 QLoRA 与独立 FP32 readout 共更新 **5,672,448 个参数**，在 626,546 条公开来源重建记录上完成 1 epoch、2,448 次优化更新。

训练使用软标签监督、候选置换和独立温度校准。工程实现包括精确长度分批、padded token 预算、保存恢复与数值核验；单张 H200 NVL 在 GPU 280W 上限下完成，含评估约 7 小时 7 分钟。专用决策接口及相关训练方法继承 [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) 的公开实现，本项目完成 0.8B 混合骨干与 QLoRA 适配。

## 文档与开放资源

- [训练过程与开发集曲线](docs/training-process.md)
- [加载、环境与训练复现](docs/reproduction.md)
- [BFCL 详细评估](docs/bfcl-report.md)
- [已发表预测的 CPU 复算](https://github.com/llzx177067-rgb/dev-1.0/tree/main/evaluation)
- [dev-data-1.0 数据与获取配方](https://modelscope.cn/datasets/lzx1770670957/dev-data-1.0)

代码、模型文件及上游归属声明随发布保留。数据按各自来源条款提供：公开训练原文 601,579 条，其余 24,967 条提供身份、指纹与获取配方；完整实际训练身份为 626,546 条。模型及源码的 Apache 2.0 声明不扩张数据使用权，使用前请阅读数据仓库的逐来源许可。
