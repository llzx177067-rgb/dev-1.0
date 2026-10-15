# BFCL 已发表预测的 CPU 复算

此目录提供两模型已冻结预测的审计材料与纯 Python 标准库计分脚本。运行下面的命令只会重新读取预测并复算分数，不会加载模型、调用 GPU 或重新推理：

```bash
python evaluation/score_bfcl.py --output outputs/bfcl-rescored.json
```

默认文件 `bfcl-predictions.jsonl.gz` 包含同一批 1,694 个请求、每模型 4,768 个二元字段。文件只保留 case/request ID、候选工具顺序、yes/no 选项顺序、期望标签、两个模型的完整答案与概率、原始 payload SHA256。它不含 conversation、state、question 或工具参数说明原文，是评估审计产物，不属于 `dev-data-1.0` 训练数据。

脚本检查文件 SHA256、唯一 case/request ID、两模型完整覆盖、候选及选项一致性、概率合法性及 argmax，然后复算整题准确率、字段准确率、TP/FP/FN/TN、召回率、精确率、误选择率、分 track 及逐题配对结果。整题正确要求该题全部候选字段正确。固定产物应复算出 Qwen3.5-0.8B 为 667/1,694，dev-1.0 为 1,311/1,694。

范围是 **Decision Index 0.2.1 的 BFCL 工具名选择投影**，不包括参数生成、调用顺序、执行验证或完整 BFCL 榜单。模型提示词、输出头、注意力以及训练起点不同，不能把差值视为只改变微调权重的严格消融。

原始 BFCL 数据来自 [Gorilla/BFCL](https://github.com/gorilla-llm/gorilla/tree/916260dfc116bf06793a1af79b4ec8195b0453b6/berkeley-function-call-leaderboard/data)，固定 commit 为 `916260dfc116bf06793a1af79b4ec8195b0453b6`，数据源为 BFCL_v3 的 simple、live_simple、live_multiple。保留其 [Apache-2.0 许可证](../LICENSES/BFCL-Apache-2.0.txt)。DI harness 固定 commit 为 `f8552a13b643d39f61185f5951bb5fed16bedf51`；准确源文件 SHA 及两模型原始结果 SHA 见 `bfcl-predictions-manifest.json`。

如需重新执行推理，应另外获取该版本的官方 DI harness 与固定 BFCL 数据，使用模型卡说明的各自推理接口。此发布目录没有打包完整官方 harness。
