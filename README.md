# 52Hz Models

52Hz 的模型适配、训练、转换和推理实验。

## 计划内容

- 本地模型和远程模型 Provider
- Prompt 与结构化输出模板
- 拼音、混输和噪声输入数据处理
- LoRA 或其他微调实验
- 模型量化和导出
- CPU、GPU 和端侧推理基准
- 模型能力和运行时元数据

模型权重不直接提交到 Git。仓库只保存配置、脚本、评测结果和可复现实验记录。

## 本地推理

本次 Qwen 2B / 27B 选型实验的命令、用例和评分器都在本仓库。在本仓库根目录运行 `make help` 查看入口。完整流程需要 Linux、GNU Make、Bash、Python 3.8+、Docker（已配置 NVIDIA 容器运行时）、`curl`、`jq`、`flock` 和 `sha256sum`。`prepare` 下载的是推理镜像和模型，不安装 Docker 或显卡驱动。

```sh
make prepare MODEL=2b
make serve MODEL=2b
make status                # 就绪后再运行下一条；未就绪可用 make logs 看加载进度
make eval
make stop
```

`prepare` 不启动容器，可在能联网时单独运行。它复用已有镜像，把所选 GGUF 下载到 `~/.cache/52hz/models`（可用 `MODEL_DIR` 覆盖）；2B 权重约 1.2 GiB，27B 约 15.4 GiB，镜像另计。未完成的下载保留为 `.part`，重跑时续传；下载完成并校验 SHA-256 后才更名。`serve` 会先执行 `prepare`，无需手动重复准备，再从本地文件后台启动模型。`stop` 保留镜像和权重。

切换 27B 时先停 2B，再运行 `make serve MODEL=27b`。`make eval` 评测当前服务，不会自动换模型。需要区分两组结果时，可分别指定 `RUNS_DIR=runs/2b`、`RUNS_DIR=runs/27b`。服务只监听本机端口；使用非默认端口时为 `serve`、`status`、`smoke`、`infer` 和 `eval` 指定相同的 `PORT`（默认 8089）。

## 选型回归用例

`make eval` 一次完成请求导出、逐条推理和评分。每次运行打印新的 `runs/<时间>-<随机后缀>/` 目录，不覆盖旧结果：

- `requests.jsonl`：发给推理程序的盲测输入，不含标准答案。
- `responses.jsonl`：带请求 ID 的模型响应。连接失败或推理中断时，只保留 `.partial` 文件，不生成评分；模型输出不是有效 JSON 时，该条记为错误并继续评测。
- `score.json`：通过数、失败 ID 和错误提交 ID。`make eval` 返回成功表示评测完整执行，不表示模型全部答对；连接、输入或报告写入出错仍会返回非零。CI 需要全部通过时用 `make eval-strict`，对已有答卷执行 `make score` 也保留严格行为。
- `failures.jsonl`：逐条列出不通过用例的请求、期望和实际回复。用 `head -1 <本次目录>/failures.jsonl | jq .` 查看第一处差异。

也可单独使用各阶段；已有完整答卷时，重新评分不需要模型服务：

```sh
make test
make requests > /tmp/52hz-requests.jsonl
make infer REQUESTS=/tmp/52hz-requests.jsonl RESPONSES=/tmp/52hz-responses.jsonl
make score RESPONSES=/tmp/52hz-responses.jsonl
```

`make test` 不下载资源，不运行真实模型。`make smoke` 可向正在运行的模型发一条测试请求，使用 `KEYS=...` 和 `CONTEXT=...` 改变输入。批量评分要求每条请求对应一条相同 `id` 的 JSONL 响应，例如：

```jsonl
{"id":"001","action":"commit","candidates":["明天开会","明天开户"]}
{"id":"003","action":"ask","focus":"kaihu","options":["开会","开户"]}
```

`preview` 和 `commit` 检查动作和首选全文；`ask` 检查询问的是哪段原始按键，以及局部选项是否一致，不要求选项顺序一致。报告单列错误提交的 ID。

[13 条手写用例](tests/fixtures/ime_cases.jsonl)覆盖同键不同语境、个人记忆、混输、`More` 和逐处消歧。另有一条日常句，按固定种子生成干净全拼和五种变体：漏键、重复、同排邻键、相邻字符交换、相邻音节交换。这 19 条均为原创合成样例，用于比较行为，不代表日常输入的成功率。

公开的 [sentences-ctx-v1](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/eval/sentences-ctx-v1) 可作为重排对照，其固定候选池只包含正确答案已入池的用例。本仓库不分发该语料。
