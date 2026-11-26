.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

MODEL ?= 2b
PORT ?= 8089
WEB_PORT ?= 8090
# 同时在途的请求数：serve 按它开服务端槽位（每槽位 4096 上下文），infer 按它开线程；WORKERS=1 是逐条串行的参照配置
WORKERS ?= 4
IMAGE ?= ghcr.io/ggml-org/llama.cpp:server-cuda
# Python 依赖（pypinyin）由 uv 管理；首次运行会按 uv.lock 创建 .venv
PYTHON ?= uv run python
MODEL_DIR ?= $(HOME)/.cache/52hz/models
RUNS_DIR ?= runs
DECODER ?= ranked
export IME_DECODER := $(DECODER)
REQUESTS ?= /tmp/52hz-requests.jsonl
RESPONSES ?= /tmp/52hz-responses.jsonl
CONTAINER := 52hz-qwen
ENDPOINT := http://127.0.0.1:$(PORT)
export KEYS ?= mingtiankaihu
export CONTEXT ?= 银行那边预约好了，资料也带齐了。

ifeq ($(MODEL),2b)
  HF_REPO := unsloth/Qwen3.5-2B-GGUF
  HF_FILE := Qwen3.5-2B-Q4_K_M.gguf
  HF_REV := f6d5376be1edb4d416d56da11e5397a961aca8ae
  HF_SHA256 := aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223
else ifeq ($(MODEL),27b)
  HF_REPO := unsloth/Qwen3.8-27B-GGUF
  HF_FILE := Qwen3.8-27B-UD-Q4_K_M.gguf
  HF_REV := 4af51b3542f775b0a8e4b310bfb3f17e01c306c9
  HF_SHA256 := 1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529
endif

.PHONY: help prepare serve status smoke logs stop test requests infer score eval eval-strict benchmark web web-build web-test

help:
	@printf '%s\n' 'make prepare [MODEL=2b|27b] 下载镜像及所选权重，不启动服务（默认 2b）' \
	  'make serve [MODEL=2b|27b]   准备缺少的资源，再启动本地 Qwen' \
	  'make ... WORKERS=4          同时在途的请求数（serve 和 infer 要一致；默认 4，1 为串行）' \
	  'make status / make logs     检查是否就绪 / 跟随服务日志' \
	  'make web                    启动浏览器实验台（WEB_PORT=8090）' \
	  'make smoke [CONTEXT=...]    用 KEYS 和 CONTEXT 发一条输入法请求' \
	  'make stop                   停止服务；保留模型缓存' \
	  'make test                   离线测试用例和评分器（不调用模型）' \
	  'make eval                   请求 → 当前模型推理 → 评分，产物保存在 runs/ 下' \
	  'make eval-strict            同样生成报告，有任何错题则返回非零（CI）' \
	  'make requests > FILE        导出不含答案的 JSONL 请求' \
	  'make infer                  从 REQUESTS 读取，向 RESPONSES 写入模型响应' \
	  'make score RESPONSES=FILE   为已有的模型响应 JSONL 评分' \
	  'make benchmark MODEL=27b    运行生成的 benchmark（BENCHMARK_CASES=... 指定用例，需先 make serve）' \
	  'make ... DECODER=hint       带音节提示的单次直出解码器，只上屏不追问（默认 ranked；另有 direct）'

prepare:
	@test -n "$(HF_FILE)" || { echo 'MODEL must be 2b or 27b' >&2; exit 2; }
	@docker image inspect "$(IMAGE)" >/dev/null 2>&1 || docker pull "$(IMAGE)"
	@mkdir -p "$(MODEL_DIR)"
	@cd "$(MODEL_DIR)"; file='$(HF_FILE)'; exec 9>"$$file.lock"; flock 9; \
	  if test -f "$$file"; then echo "使用已缓存权重：$$PWD/$$file"; exit 0; fi; \
	  curl -fL --retry 3 -C - -o "$$file.part" 'https://huggingface.co/$(HF_REPO)/resolve/$(HF_REV)/$(HF_FILE)'; \
	  if ! printf '%s  %s\n' '$(HF_SHA256)' "$$file.part" | sha256sum -c -; then \
	    mv -f "$$file.part" "$$file.invalid"; exit 1; fi; \
	  mv "$$file.part" "$$file"

serve: prepare
	docker run -d --rm --pull=never --gpus all --name $(CONTAINER) -p "127.0.0.1:$(PORT):8080" \
	  -v "$(abspath $(MODEL_DIR)):/models:ro" \
	  "$(IMAGE)" -m "/models/$(HF_FILE)" --alias 52hz-qwen \
	  --host 0.0.0.0 --port 8080 -ngl 99 -c $$((4096 * $(WORKERS))) --parallel $(WORKERS) --no-mmproj --reasoning off
	@echo '服务启动中。用 make logs 查看加载进度，make status 确认就绪。'

status:
	@curl -fsS --max-time 5 '$(ENDPOINT)/health' | jq -e .

smoke:
	@jq -n --arg system '你是中文拼音输入法。根据上文和原始按键，只输出最可能的整句，不解释。' \
	  '{model:"52hz-qwen",temperature:0,max_tokens:32,messages:[{role:"system",content:$$system},{role:"user",content:("上文："+env.CONTEXT+"\n原始按键："+env.KEYS)}]}' | \
	  curl -fsS --max-time 120 -H 'Content-Type: application/json' --data-binary @- '$(ENDPOINT)/v1/chat/completions' | \
	  jq -er '.choices[0].message.content | select(type == "string" and length > 0)'

logs:
	docker logs -f $(CONTAINER)

stop:
	docker stop $(CONTAINER)

test:
	$(PYTHON) -B -m unittest discover -s tests -v

web/node_modules/.deps: web/package.json web/package-lock.json
	cd web && npm ci --no-audit --no-fund
	@touch $@

web/dist/index.html: web/node_modules/.deps web/index.html web/src.jsx web/style.css web/public/whale.svg
	cd web && npm run build

web-build: web/dist/index.html

web-test: web-build
	cd web && npm test

web: web-build
	$(PYTHON) -B web_lab.py --port "$(WEB_PORT)" --model-port "$(PORT)"

requests:
	@$(PYTHON) -B tests/ime_cases.py requests

infer:
	@$(PYTHON) -B infer.py "$(ENDPOINT)" "$(REQUESTS)" "$(RESPONSES)" --workers $(WORKERS)

score:
	@test -f "$(RESPONSES)" || { echo '缺少响应文件。运行 make eval，或先运行 make infer。' >&2; exit 2; }
	@$(PYTHON) -B tests/ime_cases.py score "$(RESPONSES)"

BENCHMARK_CASES ?= /data/52hz-eval/build/benchmark/iteration-003/cases.jsonl

benchmark:
	@mkdir -p "$(RUNS_DIR)"; run=$$(mktemp -d "$(RUNS_DIR)/benchmark-$(MODEL)-$$(date +%Y%m%d-%H%M%S)-XXXXXX"); \
	  echo "Benchmark 目录：$$run"; \
	  $(PYTHON) -B tests/benchmark_cases.py requests "$(BENCHMARK_CASES)" > "$$run/requests.jsonl"; \
	  $(PYTHON) -B infer.py "$(ENDPOINT)" "$$run/requests.jsonl" "$$run/responses.jsonl" --workers $(WORKERS); \
	  $(PYTHON) -B tests/benchmark_cases.py score "$(BENCHMARK_CASES)" "$$run/responses.jsonl" --failures "$$run/failures.jsonl" | tee "$$run/score.json"

eval eval-strict:
	@mkdir -p "$(RUNS_DIR)"; run=$$(mktemp -d "$(RUNS_DIR)/$$(date +%Y%m%d-%H%M%S)-XXXXXX"); \
	  echo "评测目录：$$run"; \
	  $(PYTHON) -B tests/ime_cases.py requests > "$$run/requests.jsonl"; \
	  $(PYTHON) -B infer.py "$(ENDPOINT)" "$$run/requests.jsonl" "$$run/responses.jsonl" --workers $(WORKERS); \
	  $(PYTHON) -B tests/ime_cases.py score "$$run/responses.jsonl" --failures "$$run/failures.jsonl" $(if $(filter eval,$@),--report-only) | tee "$$run/score.json"
