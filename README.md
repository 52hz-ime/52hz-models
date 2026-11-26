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

本次 Qwen 2B / 27B 选型实验的命令、用例和评分器都在本仓库。在本仓库根目录运行 `make help` 查看入口。完整流程需要 Linux、GNU Make、Bash、Python 3.8+、[uv](https://docs.astral.sh/uv/)、Docker（已配置 NVIDIA 容器运行时）、`curl`、`jq`、`flock` 和 `sha256sum`。唯一的 Python 依赖是 `pypinyin`（用来把回复反译成拼音做输出校验），由 `uv.lock` 锁定；所有 Python 命令通过 `uv run python` 运行，第一次会自动创建 `.venv`。`prepare` 下载的是推理镜像和模型，不安装 Docker 或显卡驱动。

```sh
make prepare MODEL=2b
make serve MODEL=2b
make status                # 就绪后再运行下一条；未就绪可用 make logs 看加载进度
make eval
make stop
```

`WORKERS`（默认 4）是同时在途的请求数：`serve` 按它开服务端槽位，每个槽位 4096 上下文；`infer`、`eval`、`benchmark` 按它开线程。两边要一致。`WORKERS=1` 是逐条串行的参照配置；批处理会让 temperature 0 的结果和串行有微小差异。27B 开 4 个槽位在 24 GB 显卡上占约 17.8 GB。

`prepare` 不启动容器，可在能联网时单独运行。它复用已有镜像，把所选 GGUF 下载到 `~/.cache/52hz/models`（可用 `MODEL_DIR` 覆盖）；2B 权重约 1.2 GiB，27B 约 15.4 GiB，镜像另计。未完成的下载保留为 `.part`，重跑时续传；下载完成并校验 SHA-256 后才更名。`serve` 会先执行 `prepare`，无需手动重复准备，再从本地文件后台启动模型。`stop` 保留镜像和权重。

切换 27B 时先停 2B，再运行 `make serve MODEL=27b`。`make eval` 评测当前服务，不会自动换模型。需要区分两组结果时，可分别指定 `RUNS_DIR=runs/2b`、`RUNS_DIR=runs/27b`。服务只监听本机端口；使用非默认端口时为 `serve`、`status`、`smoke`、`infer` 和 `eval` 指定相同的 `PORT`（默认 8089）。

## Web 输入实验台

先启动本地模型（若已启动无需重启），另一个终端运行 `make web`，打开 **http://127.0.0.1:8090**。首次构建需要 Node.js 22.12+ 和 npm；不需要 Vercel 账号或 API key。`WEB_PORT` 设置页面端口，`PORT` 指向模型服务。Ctrl+C 只关闭网页服务，不停止模型。

在「原始按键」输入拼音，或用场景旁的载入按钮填入示例（切换或重新载入场景会清空输入区）。默认停止输入 400 ms 后异步预览，空格可结束输入；关闭实时预览可以对比只在结束时调用的延迟。用鼠标或键盘选中一段原始按键，按 F2 或点击 More 展开局部选项，没选中片段时使用整段。每次选择后把已选译文带入整句重评。由 More 发起的选择只回到编辑；由空格发起的选择在歧义消除时提交。组合态为空时可撤销最近一次上屏。原始按键内需要字面空格时使用 Shift+Space。

**首尾标点按原样保留，不做全半角转换。** 单独输入 `……` 或句首输入 `「` 直接上屏，不调用模型。输入 `nihao！` 时，`！` 立即可见，只把 `nihao` 送去解码；若需选词，选完正文再一起上屏，期间可继续输入下一段。`。！？`、`!?`、逗号和右括号等可以结束本段；可能出现在正文内部的 `.`、`-`、`'` 等不抢先断句，留到空格确认，因此 `3.14`、`PR-239` 和 `xi'an` 仍可连续输入。

页面显示当前模型、可编辑的上下文和逐行记忆；已提交文本也会加入后续请求的上下文。调用记录可展开查看请求、响应和端到端耗时（包括等待前一请求的时间，不含输入防抖），保留最近 50 次已发出的调用。页面采用 Vercel AI SDK UI 的 `useObject` 对完整 JSON 做结构校验；本地 Python 服务复用批量评测的 `infer.predict`，不按 token 提前上屏。同一页面只允许一个模型调用在途，后续只保留最新待发请求；旧结果标记为过期。输入和日志只保存在当前页面内存中，不写入 `runs/`。

浏览器回归：先 `make web-build`，首次测试时在 `web` 目录运行 `npx playwright install chromium`，再从仓库根目录运行 `make web-test`。用例会启动临时 Web 端口、模拟模型响应，不需要 GPU。

## 选型回归用例

默认解码器 `DECODER=ranked` 分两步，不强制切分拼音。首先使用本机 `llama.cpp` 的 `/apply-template` 和 `/completion` 从原始按键生成一句完整译文，读取 token 备选概率，再从有竞争的分叉点分别续写，最多保留五个不同整句；对可能的重字优先尝试另一条路径。这是有预算的分支搜索，不是穷尽所有解释或经过校准的置信度。主动 `More` 在候选不足时还会围绕指定片段提出替代整句。

裁决器读完整候选、原始按键、上下文、记忆和已选译文，返回候选编号而非另写文本。候选池默认不弹给用户；`type` 只能预览，空格后有实质歧义才询问，`More` 必须查看指定片段。无合法候选或缺少有效局部选项时保留输入，不强行提交。`responses.jsonl` 和 Web 调用记录的 `search` 字段可检查生成顺序、最终选择和两个阶段的毫秒耗时。排在首选后的候选仅为备选，不代表概率排序。

`DECODER=hint` 是第三种解码器：一次普通对话调用，带示例和一行机械切分的音节提示（`切分：jin tian wan shang chi fan?`，`?` 标出可能按错的位置），直接把回复作为唯一候选。它按顺序做三件事：

1. 回复里必须保留 `selected` 中用户选定的文字，否则退回 JSON 契约。
2. 把回复反译成拼音和按键比较，相差超过 5 个字符或按键长度的 30% 就不上屏，返回 error 保留原始输入。试跑里正确的句子从未超过 2，读得通但选词不同的近错都在四分之一以内，只有和按键无关的输出才超过三分之一，所以这道校验只拦离谱结果。
3. `type` 返回 preview，`space` 返回 commit。

它不追问，也不搜索候选，速度快一个量级，用来测“一次成功率”。追问是单独的课题，目前只有 `ranked` 有追问路径。音节切分只依赖内置的 424 个音节和音节频率表，`keys` 仍原样发送。

同一台模型服务上执行 `make eval DECODER=direct` 可复测原先的一次生成；`make eval DECODER=ranked` 使用新算法。Web 也使用同一配置，切换时只需重启 `make web DECODER=...`，不需要重新下载模型。`make smoke` 仍是单请求连接检查。

`make eval` 一次完成请求导出、逐条推理和评分。每次运行打印新的 `runs/<时间>-<随机后缀>/` 目录，不覆盖旧结果：

- `requests.jsonl`：发给推理程序的盲测输入，不含标准答案。
- `responses.jsonl`：带请求 ID 的模型响应。连接失败或推理中断时，只保留 `.partial` 文件，不生成评分；模型输出不是有效 JSON 时，该条记为错误并继续评测。
- `score.json`：通过数、失败 ID 和错误提交 ID。两阶段结果另有 `sentence_search`，只统计有标准全文的用例：`in_pool / target_sentences` 是正确句进池率，`winner_correct / in_pool` 是进池后的选中率；这两项不代替交互验收。`make eval` 返回成功表示评测完整执行，不表示模型全部答对；连接、输入或报告写入出错仍会返回非零。CI 需要全部通过时用 `make eval-strict`，对已有答卷执行 `make score` 也保留严格行为。
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

[14 条手写用例](tests/fixtures/ime_cases.jsonl)覆盖同键不同语境、个人记忆、混输、`More` 和逐处消歧。另有一条日常句，按固定种子生成干净全拼和五种变体：漏键、重复、同排邻键、相邻字符交换、相邻音节交换。这 20 条均为合成样例，用于比较行为，不代表日常输入的成功率。

推理适配器按原样发送 `keys`，不补齐错漏的字母；`hint` 解码器额外附带的切分只是提示，不改动 `keys`。型号、大小写和用户输入的空格均保留。拼音是意图的线索，不要求候选逐音严格匹配；候选还应结合上下文，读起来像用户真正想说的整句话。

公开的 [sentences-ctx-v1](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/eval/sentences-ctx-v1) 可作为重排对照，其固定候选池只包含正确答案已入池的用例。本仓库不分发该语料。

## 生成 benchmark

`make benchmark` 对 `BENCHMARK_CASES`（默认 52hz-eval 生成的 `benchmark/iteration-003/cases.jsonl`）导出盲测请求、逐条推理并评分，产物在 `runs/benchmark-<模型>-<时间>-<随机后缀>/`。试跑可以用分层小集：

```sh
make benchmark MODEL=27b DECODER=hint \
  BENCHMARK_CASES=/data/52hz-eval/build/benchmark/iteration-003/trial-100.jsonl
```

`score.json` 按产品定义分开统计，分母是全部用例：

- `passed`：一次成功率，只有 `commit` 且首选全文正确才计入。
- `wrong_commits`、`asks`、`errors`：错误上屏、主动询问、没有合法输出（`hint` 下包括因读音相差太远被拦下、没有上屏的）。
- `in_pool`：正确全文是否出现在 `search.pool` 里（只有 `ranked` 有候选池），是裁判选择的上限。
- `char_error_rate`：上屏结果相对标准答案的字错率。
- `by_operation`、`by_source`、`by_length`：按噪声操作、来源和长度分组，每组都带上面各项。

`failures.jsonl` 每行是一条失败用例的 `expected` 和 `actual`。在 27B、4 并发下，`ranked` 每条约 2 秒，`hint` 每条约 0.3 秒。
