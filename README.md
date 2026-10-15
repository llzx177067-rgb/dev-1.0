# dev-1.0 训练与评估源码

dev-1.0 将 Qwen3.5-0.8B-Base 适配为结构化决策模型，用 **NF4 QLoRA 与独立 FP32 决策头**训练 626,546 条公开来源重建记录。发布 checkpoint 完成 1 epoch、2,448 次更新，共有 5,672,448 个可训练参数。

在 Decision Index 0.2.1 的 BFCL 工具名选择投影中，dev-1.0 整题准确率 **77.39%**；官方后训练版 Qwen3.5-0.8B 对照为 **39.37%**，相差 **38.02 个百分点**。该结果只衡量工具名选择，不涵盖参数生成、调用顺序和执行；模型起点与推理接口差异见[评估说明](docs/bfcl-report.md)。

模型发布：[`lzx1770670957/dev-1.0`](https://www.modelscope.cn/models/lzx1770670957/dev-1.0)。数据与重建材料：[`lzx1770670957/dev-data-1.0`](https://www.modelscope.cn/datasets/lzx1770670957/dev-data-1.0)。

## 实现重点

专用答案头、最后位置 readout、软标签交叉熵、候选置换、非因果 full attention 和温度拟合继承 [pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) 的公开方法，固定源码 release revision 为 `db124e5056059be396ec5ba69b97a33dd6adb815`。本项目围绕 Qwen3.5-0.8B 的混合骨干、低显存训练、批量吞吐与运行审计做适配。

- 专用 255 路决策输出，以候选概率替代自由文本生成与解析；保留软标签训练、候选置换增强和独立温度校准。
- 适配 Qwen3.5 的 6 层 full attention 与 18 层 GatedDeltaNet；只取消 full attention 的因果限制，保留 padding 和 DeltaNet 原递推方向。
- NF4、double quantization、BF16 compute，语言 all-linear LoRA rank 8；冻结视觉塔与 embedding，单独训练 FP32 readout。
- CPU 单次精确编码、有效 batch 内稳定长度排序、按真实 padded tokens 拆分 microbatch；不 packing、不截断。
- checkpoint 保存 optimizer、RNG、数据游标与文件指纹，验证图文、8K 输入和重载后更新。

训练采用监督式软标签交叉熵，没有强化学习阶段。上述工程优化未逐项做准确率消融，不能分别归因 BFCL 的总分提升。

## 复现材料

发布保留训练器、运行时依赖源码、Docker 环境定义、数据准备与校验脚本、冻结训练配置，以及本次对照所需的推理与计分代码。Base 权重由官方仓库按固定 revision 单独下载；训练数据以 `dev-data-1.0` 中的许可分发范围和重建配方为准。

| 材料 | 作用 |
| --- | --- |
| `training/qwen35-0.8b/qwen35_qlora.py` | QLoRA 决策模型、监督训练、校准、保存和恢复 |
| `training/qwen35-0.8b/exact_batch.py` | 精确编码及 microbatch 组织 |
| `training/qwen35-0.8b/runtime-source/` | 固定版本上游 Decider 运行时代码及兼容补丁 |
| `training/qwen35-0.8b/Dockerfile` | 固定版本 GPU 环境 |
| `training/qwen35-0.8b/prepare_data.py` | Base processor 编码、长度与隔离核验 |
| `configs/training-first-run.json` | 发布 checkpoint 对应的脱敏冻结训练配置 |
| `provenance/` | 实际依赖环境、推理所需文件身份和冻结源码 SHA256 |
| `tools/predict.py` | 自定义 checkpoint 文件验证与推理入口 |
| `docs/bfcl-report.md` | BFCL 计分定义、混淆矩阵和对照边界 |
| `docs/bfcl-summary.json` | 精确指标及来源指纹 |
| `docs/training-process.md` | 训练阶段、11 次开发评估、校准与功耗记录 |

`prepare_data.py` 和 `verify_prepared_data.py` 保留历史数据准备与审计逻辑；它们依赖原始候选、holdout、Tasksource parquet、上游 prompt source 和原审计元数据。仅 `git clone` 本仓库不能直接执行这条历史重建链。下面的训练主路径使用数据仓库提供的最终 JSONL，或按上游条款取得受限记录后精确恢复的完整 JSONL。

## 加载发布模型

先取得固定 revision 的 `Qwen/Qwen3.5-0.8B-Base`、ModelScope 上的 `dev-1.0` 文件，以及本仓库依赖环境。发布物是 adapter、readout 和 processor，不是标准聊天模型的整套生成权重。可在允许联网的下载环境执行：

```bash
python scripts/download_qwen35_base.py \
  --local-dir /path/to/Qwen3.5-0.8B-Base
```

下载器固定 Base revision 并生成原训练器要求的 `download-verification-manifest.json`。遇到代理大文件限制时可增加 `--range-download-large`。加载器校验全部 12 个 Base 文件，不能换成官方后训练版、其他 revision 或缺少元数据的快照。随后在仓库根目录执行：

```bash
python tools/predict.py \
  --model-dir /path/to/dev-1.0 \
  --base-model-path /path/to/Qwen3.5-0.8B-Base \
  --input examples/decision.json \
  --output result.json
```

输入是 UTF-8 JSON，支持单条决策记录或其数组，包含 `state` 和 `question`；输出 envelope 的 `predictions` 是 Answer 列表，包含候选概率与选中答案。推理需要 CUDA，使用 `CUDA_VISIBLE_DEVICES` 选择设备。加载器保留模型保存的温度，只应用一次；`--verify-only` 只做 CPU 文件与配置校验，不执行推理。

也可在 Python 中调用 `from tools.predict import load_decider, predict_rows`，通过 `load_decider(model_dir, base_model_path)` 加载，再用 `predict_rows(model, rows, batch_size=1)` 获取 Answer 列表。

## 复现监督训练

先按数据仓库的分发文件与获取配方恢复数据，并核对完整训练集 SHA256：`2c3a16403c75ef3df2d936048a38c686e22dd0520871a33a94946494bac8c20d`。直接分发的训练原文为 601,579 条，其他 24,967 条提供 ID、指纹与获取配方；完整实际训练量是 626,546 条。仅使用可直接下载子集重新训练时，应以新的数据身份报告结果。

冻结环境中，从仓库根目录运行：

```bash
export PYTHONPATH="$PWD/training/qwen35-0.8b/runtime-source/src"
python training/qwen35-0.8b/qwen35_qlora.py \
  --model-path /models/qwen35-0.8b-base \
  --output runs/dev-1.0-reproduction \
  --train data/qwen35-final/train.jsonl \
  --development data/qwen35-final/dev.jsonl \
  --temperature data/qwen35-final/temperature.jsonl \
  --attention-mode noncausal_full_attention \
  --rank 8 --lora-targets all-linear \
  --batch-size 32 --effective-batch-size 256 \
  --max-length 8192 --token-budget 32768 \
  --lr 1e-4 --adam-eps 1e-6 --warmup-ratio 0.05 \
  --epochs 1 --eval-every 250 --save-every 10 \
  --cpu-threads 4 --seed 20261012 --deterministic --length-bucket
```

上例假设 GPU 功耗限制已按部署环境设置，训练器不会自行修改硬件功耗。发布训练使用硬件 280W 上限和独立监控；在其他设备上应保留相同数据及算法设置，并重新验证容量和数值行为。

实际运行使用 Python 3.12、PyTorch 2.14.0/CUDA 13.0、Transformers 5.17.0、PEFT 0.21.2、bitsandbytes 0.50.2、tokenizers 0.23.2 和 flash-linear-attention 0.5.2。Dockerfile 固定依赖；其他版本需重新验证 Qwen3.5 量化、DeltaNet、注意力与恢复路径。

训练设置为 effective batch 256、microbatch 上限 32、padded token budget 32,768、max length 8,192、LR `1e-4`、5% warmup + cosine、AdamW epsilon `1e-6`、weight decay `0.01`、gradient clip norm `1.0`、seed 20261012、1 epoch。单张 H200 NVL 在 GPU 280W 上限下完成，含初始及定期评估约 7 小时 7 分钟；性能与耗时会随硬件和长度分布改变。

复现时先核验固定 Base、数据、源码和镜像身份，再执行 CPU 检查、图文 GPU 冒烟和容量验证，最后启动训练。开发集和温度集独立于梯度更新；BFCL 测试集不参与校准、选模或规则调整。恢复须使用同一 run 最近完整 checkpoint，不应更改冻结参数后声称是相同训练轨迹。

训练起点的未更新 adapter 与初始化 readout 在原开发集上的准确率为 37.34%，训练完成后为 85.91%；校准 ECE 从 18.48% 降至 1.65%。这是 1,497 条原开发集上的监测指标，不是 BFCL 分数，也不是官方后训练版 Qwen 的成绩；[训练过程与曲线](docs/training-process.md)列出全部 11 个评估点及功耗边界。

## 评估结果

| 模型 | 整题准确率 | 必要工具召回率 | 不必要工具选择率 |
| --- | ---: | ---: | ---: |
| 官方 Qwen3.5-0.8B | 39.37% | 99.94% | 97.30% |
| **dev-1.0** | **77.39%** | 96.75% | **15.06%** |

两者使用同一 1,694 题、4,768 个判断字段，全部完成、无跳过。一题中所有候选工具的 yes/no 判断均正确才得分。原版保留完整 LM head 和因果注意力，dev-1.0 使用决策头与内部提示；原版是官方后训练模型，dev-1.0 则从 Base 训练。这是接口级实测对照，不是严格控制变量的微调消融，也不是完整 BFCL 榜单复现。

提升主要来自多候选子集：`live_multiple` 准确率从 0.96% 升至 65.28%。`simple` 从 100.00% 降至 99.75%，`live_simple` 从 100.00% 降至 91.47%；必要工具召回也略有下降，详见完整报告。

## 模型与数据许可

发布代码与模型遵守随仓库保留的 Apache 2.0 声明及上游归属信息。实际训练数据具有多来源混合条款，包含非商业研究及其他限制；模型或源码许可不扩张数据使用权，也不提供无条件商用授权。下载、重建和使用前应阅读 `dev-data-1.0` 的逐来源清单。
