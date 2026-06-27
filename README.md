# operator-common-iterator

> 昇腾 CANN 算子约束提取提示词的**迭代优化闭环**：从算子文档出发，自动
> 提取结构化约束 → 生成 ATK 测试用例 → 远程执行 → 分析失败根因 →
> 自动优化提示词 → 进入下一轮，直到全部用例通过或达到上限。

## ✨ 特性

- **端到端闭环**：约束提取 → 用例生成 → 用例执行 → 失败分析 → 提示词优化，无需人工介入。
- **双 Session 隔离**：Session A（提取/生成/执行）与 Session B（分析/优化）使用独立
  LLM 会话，互不污染上下文，避免"记忆溢出"导致分析偏差。
- **多 LLM 后端**：开箱支持 `api`（DeepSeek / Z.AI / OpenAI 兼容协议）和
  `agent`（`claude` / `opencode` 等 CLI 子进程）两种调用方式。
- **真实 / Mock 双执行模式**：可对接昇腾 ATK 开发机做真实 SSH 远端执行，
  也支持 Mock 执行快速验证迭代逻辑。
- **自动重试 & 校验**：约束 JSON 经 Pydantic 严格校验，失败时把错误反馈
  给 LLM 重新提取（最多 3 次）。
- **根因分类**：将失败归因为 `constraint_extraction` / `generator_bug` /
  `executor_bug` 三类，前者触发提示词进化，后两者直接终止并报告。

## 📁 目录结构

```
operator-common-iterator/
├── orchestrator.py              # 主入口：迭代编排器（推荐从这里开始）
├── constraint_extractor.py      # Session A：约束提取（独立 LLM 调用）
├── result_analyzer.py           # Session B：根因分析 & 提示词优化
├── backends.py                  # LLM 后端抽象（APIBackend / CliAgentBackend）
├── config.py                    # pydantic-settings 配置（读取 .env）
├── servers.json                 # ATK 远程执行服务器配置（明文密码，仅限内网）
├── servers.json.example         # 服务器配置示例
├── .env                         # LLM API Key 等敏感配置（请勿提交）
│
├── executer/                    # 远端执行子图（复用 operator-agent）
│   ├── run_atk.py               # SSH 上传 + atk 命令 + 结果下载/解析
│   ├── generate_atk.py          # 用 cases + 签名表生成 ATK executor 脚本
│   ├── cpu_derivation.py        # 给 executor 补充 CPU golden reference
│   ├── ssh_executor.py          # asyncssh 封装（connect/upload/run）
│   ├── report_parser.py         # 解析远端 report/*.xlsx
│   ├── execution_result.py      # 执行结果数据结构
│   └── resources/
│       ├── generator.py         # ATK executor 脚本生成器（C++ 签名解析）
│       └── aclnn_extracted.txt  # aclnn 算子签名表
│
├── generators/                  # 用例生成器（来自 operator-agent，约束求解）
│   ├── facade.py
│   ├── operator_handle_main.py
│   └── …
│
├── prompts/
│   └── operator_constraints_extract_v1.md  # 初始约束提取提示词
│
├── docs/                        # 算子说明文档（输入）
│   ├── aclnnAlltoAllMatmul.md
│   └── aclnnNpuFormatCast.md
│
├── iterator_output/             # 每次运行的产物（按算子+时间戳组织）
├── logs/                        # 运行日志
└── workflow_iterate.js          # Workflow 多 Agent 编排脚本（可选）
```

## 🚀 快速开始

### 1. 准备环境

```bash
# Python ≥ 3.10
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt # 主要依赖：asyncssh, openpyxl, pydantic,
                                #          pydantic-settings, langchain-openai
```

> 项目依赖同目录的 `operator-project/operator-agent` 包（`packages/agent/src`
> 和 `packages/shared/src` 自动加入 `sys.path`）。若没有这个包，
> 用例生成会回退到 Mock。

### 2. 配置 `.env`

复制或编辑项目根目录下的 `.env`：

```ini
# LLM provider: "deepseek" or "zai"
LLM_PROVIDER=deepseek

# DeepSeek
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

# Z.AI（备选）
ZAI_API_KEY=
ZAI_BASE_URL=https://api.z.ai/api/paas/v4/
ZAI_MODEL=glm-5.1

# CLI Agent 模式（仅当 --backend agent 时生效）
CLI_AGENT_BIN=claude
CLI_AGENT_ARGS=-p {prompt} --print --output-format text

# LLM 参数
LLM_TEMPERATURE=0.1
```

> ⚠️ `.env` 已被 `.gitignore` 忽略，但生产环境请使用更安全的密钥管理。

### 3. 配置远端服务器（真实执行模式）

`servers.json` 是真实执行时使用的服务器清单，按平台名匹配：

```json
{
  "servers": [
    {
      "name": "Atlas A3 开发机",
      "ip": "192.168.1.100",
      "port": 22,
      "username": "operator_atk",
      "password": "your_password_here",
      "platforms": ["Atlas A3 训练系列产品/Atlas A3 推理系列产品"],
      "env_init_script": "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    }
  ]
}
```

字段含义见 `servers.json.example`。密码仅限内网环境，请妥善保管。

### 4. 运行

#### 主入口：迭代流水线（推荐）

```bash
# Mock 模式 — 无需服务器、不消耗 GPU，快速验证迭代逻辑
python orchestrator.py \
    --prompt prompts/operator_constraints_extract_v1.md \
    --doc docs/aclnnAlltoAllMatmul.md \
    --max-iterations 5 \
    --case-count 10 \
    --mock-exec

# 真实执行模式 — 走 SSH + ATK 远端执行完整闭环
python orchestrator.py \
    --prompt prompts/operator_constraints_extract_v1.md \
    --doc docs/aclnnAlltoAllMatmul.md \
    --max-iterations 5 \
    --case-count 10 \
    --server-config servers.json \
    --platform "Atlas A3 训练系列产品/Atlas A3 推理系列产品" \
    --log-level INFO

# 使用 CLI Agent 后端（claude / opencode 等）
python orchestrator.py \
    --prompt prompts/operator_constraints_extract_v1.md \
    --doc docs/aclnnAlltoAllMatmul.md \
    --backend agent \
    --mock-exec
```

**所有参数：**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--prompt` *(必填)* | 初始提示词 Markdown 路径 | — |
| `--doc` *(必填)* | 算子文档 Markdown 路径 | — |
| `--output-root` | 输出根目录 | `iterator_output` |
| `--max-iterations` | 最大迭代轮数 | `5` |
| `--case-count` | 每平台生成用例数 | `10` |
| `--mock-exec` / `--real-exec` | Mock 或真实执行 | `--real-exec` |
| `--platform` | 目标平台名（覆盖算子约束中的 product_support） | 文档自动推断 |
| `--server-config` | 服务器配置文件路径 | `servers.json` |
| `--backend` | LLM 后端：`api` / `agent` | `api` |
| `--log-level` | 日志级别 | `INFO` |

退出码：`0` = 全部用例通过；`1` = 未通过/达到上限/被判定为生成器/执行器 bug。

#### 子模块单独运行

```bash
# 1) 仅做约束提取（独立 Session A）
python constraint_extractor.py \
    --prompt prompts/operator_constraints_extract_v1.md \
    --doc docs/aclnnAlltoAllMatmul.md \
    --output iterator_output/iter_001/constraints.json \
    --operator-name aclnnAlltoAllMatmul

# 2) 仅做结果分析与提示词优化（独立 Session B）
python result_analyzer.py \
    --prompt prompts/operator_constraints_extract_v1.md \
    --doc docs/aclnnAlltoAllMatmul.md \
    --constraints iterator_output/iter_001/constraints.json \
    --cases iterator_output/iter_001/cases.json \
    --exec-result iterator_output/iter_001/execution_result.json \
    --output iterator_output/iter_001/analysis.json

# 3) 仅生成 ATK executor 脚本
python executer/resources/generator.py \
    iterator_output/iter_001/cases.json \
    -o iterator_output/iter_001/aclnnAlltoAllMatmul_atk_executor.py \
    --signatures executer/resources/aclnn_extracted.txt
```

## 🧠 工作原理

### 流水线

```
                ┌──────────────────────────────────────┐
                │            Session A (独立 LLM)       │
                │  1. 加载提示词 vN + 算子文档          │
                │  2. 提取约束 JSON（Pydantic 校验）     │
                │  3a. 生成用例（generators + 约束求解） │
                │  3b. 远端执行：SSH → atk → 解析报告    │
                └────────────────┬─────────────────────┘
                                 │ 文件
                                 ▼
                ┌──────────────────────────────────────┐
                │            Session B (独立 LLM)       │
                │  4. 分析执行结果 / 日志                │
                │  5. 判定根因                          │
                │     ├─ constraint_extraction → 优化提示词
                │     ├─ generator_bug       → 终止报告
                │     └─ executor_bug        → 终止报告
                └────────────────┬─────────────────────┘
                                 │ 改进的提示词 vN+1
                                 ▼
                          下一轮迭代 (≤ max_iterations)
```

### 终止条件

- ✅ **全部用例通过** → 退出码 0
- 🛑 **根因 = `generator_bug` / `executor_bug`** → 终止并报告（不可通过优化提示词解决）
- ⏳ **达到 `--max-iterations` 上限**仍未全过 → 终止并报告
- 💥 **SSH / SFTP / LLM 等引擎级错误** → 写入 `state["error"]`，回退 Mock 或终止

### 输出目录布局

```
iterator_output/
└── aclnnAlltoAllMatmul_20260627_120000/      # 一次完整运行
    ├── pipeline_summary.json                 # 顶层摘要
    ├── iter_001/
    │   ├── constraints.json                  # Session A 提取结果
    │   ├── cases.json                        # 生成的 ATK 用例
    │   ├── execution_result.json             # 远端执行结果
    │   ├── analysis.json                     # Session B 分析结果
    │   ├── extraction_raw_output.txt         # 提取失败时的原始 LLM 输出
    │   ├── prompt_v2.md                      # Session B 优化后的提示词
    │   ├── execution_results/                # 镜像的 atk.log + report/*.xlsx
    │   └── aclnnAlltoAllMatmul_atk_executor.py
    ├── iter_002/
    │   └── …
    └── operator_constraints_extract_v2.md    # 最新提示词副本
```

## 🧪 诊断工具（可选）

仓库根目录下还有几个独立诊断脚本，便于排障：

```bash
# 直接验证 DeepSeek 流式响应
python diag_deepseek.py

# 复现 LLM 流空闲超时场景
python diag_idle_timeout.py

# 验证 langchain 集成是否正常
python diag_langchain.py
```

## 🔧 常见问题

**Q: 跑真实执行时报 `SSH 连接失败`？**
A: 检查 `servers.json` 中的 IP/端口/账号密码，确认目标机器
`/home/operator_atk/` 目录存在且当前用户可写。

**Q: `atk` 命令找不到？**
A: 在 `servers.json` 中给该服务器设置 `env_init_script`（默认
`/usr/local/Ascend/ascend-toolkit/set_env.sh`），orchestrator 会在
SSH 执行 `atk` 之前自动 `source` 该脚本。

**Q: 约束提取一直重试失败？**
A: 打开 `iter_NNN/extraction_raw_output.txt` 看 LLM 原始返回，再用
`python constraint_extractor.py ...` 单跑一次确认是否是 Pydantic schema 缺失字段。

**Q: 怎么强制走 Mock 执行？**
A: 加 `--mock-exec`；若 `servers.json` 不存在或不可达，编排器也会自动回退 Mock。

**Q: 想要跑更多轮？**
A: 调大 `--max-iterations`（默认 5）。注意 Session B 优化提示词后
进入下一轮，迭代间提示词独立，不会污染。

## 📜 许可证

内部项目，请遵循所在组织的代码规范与许可证要求。