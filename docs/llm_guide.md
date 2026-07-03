# Running LLM agents · LLM 智能体使用指南

The LLM baseline asks any OpenAI-compatible chat model to pick a legal
action index each turn. **No API key is stored in this repository**.
In the Web GUI, choose **大模型 LLM** and fill the model/Base URL/API Key
fields when creating the room. In scripts or CLI examples, provide them
through environment variables.

Only enter an API key on an HTTPS GUI origin. Public HTTP pages are blocked
from creating LLM rooms because the key would travel in clear text.

## 1. Where to put your API key · 在哪里配置 API key

Set three environment variables before running:

| variable | meaning | example |
|---|---|---|
| `GUANDAN_LLM_API_KEY` | your API key (**required**) | `sk-...` |
| `GUANDAN_LLM_BASE_URL` | endpoint of an OpenAI-compatible service (optional in scripts; required by the Web GUI) | `https://api.deepseek.com/` |
| `GUANDAN_LLM_MODEL` | model name (**required**) | `deepseek-chat` |

Linux / macOS:

```bash
export GUANDAN_LLM_API_KEY="sk-your-key"
export GUANDAN_LLM_BASE_URL="https://api.deepseek.com/"
export GUANDAN_LLM_MODEL="deepseek-chat"
```

Windows PowerShell:

```powershell
$env:GUANDAN_LLM_API_KEY = "sk-your-key"
$env:GUANDAN_LLM_BASE_URL = "https://api.deepseek.com/"
$env:GUANDAN_LLM_MODEL = "deepseek-chat"
```

Tested provider endpoints (any OpenAI-compatible service works):

| provider | base_url |
|---|---|
| OpenAI | *(leave `GUANDAN_LLM_BASE_URL` unset)* |
| DeepSeek | `https://api.deepseek.com/` |
| Volcengine Ark (火山方舟) | `https://ark.cn-beijing.volces.com/api/v3` |
| Alibaba DashScope (百炼) | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| SiliconFlow (硅基流动) | `https://api.siliconflow.cn/v1` |
| Tencent LKEAP | `https://api.lkeap.cloud.tencent.com/v1` |
| local vLLM / Ollama | `http://localhost:8000/v1` etc. |

> Never commit keys. If a key ever lands in a commit, revoke it on the
> provider console immediately - removing the file is not enough.

## 2. Run a full match · 跑一场完整对局

```bash
pip install -e ".[llm]"        # installs the openai package

# LLM (seats 1/3) vs rule baseline base5 (seats 0/2), one full episode:
python examples/run_llm_match.py --opponent base5 --episodes 1

# options:
#   --opponent base1..base8|random|danzero|perfectdan
#   --prompt   compact|guided      (guided restates the rules in the prompt)
#   --model    overrides GUANDAN_LLM_MODEL
#   --record-dialogue              (keep prompt/reply pairs on the agents)
```

## 3. Use the agent in your own script · 在自己的代码中使用

```python
import numpy as np
import guandan_rlcard
from guandan_rlcard.baselines import get_agent_class
from guandan_rlcard.baselines.llm.llm_agent import LLMAgent

env = guandan_rlcard.make({'seed': 42})
Base5 = get_agent_class('base5')
env.set_agents([
    Base5(0, np.random.RandomState(0)),
    LLMAgent(1, np.random.RandomState(1)),     # reads the env vars
    Base5(2, np.random.RandomState(2)),
    LLMAgent(3, np.random.RandomState(3)),
])
trajectories, payoffs = env.run()
```

Behaviour details:

* The agent answers with an action index; unparseable replies fall back
  to a random legal action (`agent.fallback_count` tracks how often).
* `<think>...</think>` blocks from reasoning models are stripped.
* Prompt templates live in `guandan_rlcard/baselines/llm/prompts.py`
  (`COMPACT_PROMPT` is the original research prompt, `GUIDED_PROMPT`
  adds a rule summary); pass your own via `LLMAgent(template=...)`.

## 4. Build fine-tuning data · 生成微调数据

```bash
python examples/generate_dataset.py --episodes 10 --output dataset/base7.jsonl
```

This logs one record per decision (prompt, chosen action, legal list)
from the Base7 teacher policy - the same pipeline used to fine-tune
card-playing LLMs in the original research.
