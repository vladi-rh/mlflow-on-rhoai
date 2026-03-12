# Agent Tracing with MLflow

This folder contains a **LangChain/LangGraph agent** with automatic **MLflow tracing** and **MCP (Model Context Protocol)** tool integration.

## Overview

The agent:
- Uses **LangGraph ReAct Agent** architecture
- Connects to a **MaaS (Model as a Service)** OpenAI-compatible endpoint
- Supports **tool use** with real API integrations:
  - **Calculator** - Math operations (add, subtract, multiply, divide, sqrt, power)
  - **Weather** - Real-time weather via [Open-Meteo API](https://open-meteo.com/) (free, no API key)
  - **Search** - Web search via [DuckDuckGo](https://duckduckgo.com/) (free, no API key)
  - **Travel** - Flight search via [Kiwi MCP](https://mcp.kiwi.com) (free SaaS MCP server)
- **Automatically traces execution** using **MLflow autolog**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph ReAct Agent                       │
│               (MLflow autolog traces automatically)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌──────────────────────┐      ┌──────────────────────┐
│    ChatOpenAI        │      │   Tool Execution     │
│  (MaaS endpoint)     │      │ (Local + MCP tools)  │
└──────────┬───────────┘      └──────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                    MLflow Tracking Server                     │
│              Automatic tracing via mlflow.langchain.autolog   │
└──────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `traced_agent.py` | LangChain/LangGraph agent implementation |
| `run_tracing_demo.py` | Runner script with manual MLflow logging |
| `run_tracing_demo_autolog.py` | Simplified runner with autolog only |
| `run_tracing_demo_autolog_prompt.py` | Runner with MLflow Prompt Registry |
| `register_prompt.py` | Register prompts in MLflow Prompt Registry |
| `evaluate_agent.py` | Evaluation script with custom scorers |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment template (committed) |
| `.env` | Your secrets (NOT committed) |

## Quick Start

### 1. Install Dependencies

> **Note:** This demo uses the standard MLflow package (not the RHOAI SDK) for full LangChain autolog tracing support.

```bash
python -m venv mlflow-venv
source mlflow-venv/bin/activate
cd agent-tracing
pip install -r requirements.txt
```

### 2. Configure Environment

First, get your OpenShift token for RHOAI authentication:

```bash
export MLFLOW_TRACKING_TOKEN=$(oc whoami --show-token)
```

Then export your environment variables:

```bash
# MaaS (Model as a Service) Configuration
export MAAS_API_KEY=your-api-key-here
export MAAS_BASE_URL=https://litellm-litemaas.apps.xxx/v1
export MAAS_MODEL=qwen3-14b

# MLflow Configuration (RHOAI)
DS_GW=$(oc get route data-science-gateway -n openshift-ingress -o template --template='{{.spec.host}}')
export MLFLOW_TRACKING_URI="https://$DS_GW/mlflow"
export MLFLOW_EXPERIMENT_NAME=mlflow-agent-tracing
export MLFLOW_WORKSPACE=ai-bu-shared
export MLFLOW_TRACKING_INSECURE_TLS=true

# MCP Server (optional)
export MCP_SERVER_URL=https://mcp.kiwi.com
export MCP_SERVER_ENABLED=true
```

Then generate your `.env` file using `envsubst`:

```bash
envsubst < .env.example > .env
```

Alternatively, you can manually copy and edit the `.env` file:

```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Run the Agent

```bash
# Batch mode with example queries
python run_tracing_demo_autolog.py
```

### 4. View Traces

Get the MLflow UI URL from the OpenShift route:

```bash
DS_GW=$(oc get route data-science-gateway -n openshift-ingress -o template --template='{{.spec.host}}')
echo "MLflow UI: https://$DS_GW/mlflow"
```

Open the URL and navigate to the **Traces** tab.

The MLflow UI is also accessible from the OpenShift console Applications menu or the RHOAI dashboard.

## Tool Integrations

### Calculator
Built-in tool supporting: `add`, `subtract`, `multiply`, `divide`, `sqrt`, `power`

### Weather (Open-Meteo API)
Real-time weather data from [Open-Meteo](https://open-meteo.com/) - **free, no API key required**.

Supported cities: New York, London, Tokyo, Sydney, San Francisco, Paris, Berlin, Madrid, Barcelona, Los Angeles

Example: "What's the weather in Tokyo?"

### Search (DuckDuckGo)
Web search via [DuckDuckGo](https://duckduckgo.com/) using `langchain-community` - **free, no API key required**.

Example: "Search for information about machine learning"

### Travel (Kiwi MCP)
Flight and travel search via [Kiwi MCP server](https://mcp.kiwi.com) - **free SaaS MCP server**.

Example: "Find me a direct flight from San Francisco to Tokyo on March 31st"

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAAS_API_KEY` | API key for MaaS endpoint | (required) |
| `MAAS_BASE_URL` | MaaS endpoint URL | `https://litellm-litemaas.apps.prod.rhoai.rh-aiservices-bu.com/v1` |
| `MAAS_MODEL` | Model name | `Llama-4-Scout-17B-16E-W4A16` |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://127.0.0.1:5000` |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name | `langchain-agent-tracing` |
| `MCP_SERVER_ENABLED` | Enable MCP tools | `true` |
| `MCP_SERVER_URL` | MCP server URL | `https://mcp.kiwi.com` |

## MCP Integration

The agent connects to the **Kiwi MCP server** for travel/flight searches:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "travel": {
        "url": "https://mcp.kiwi.com",
        "transport": "streamable_http",
    }
})
tools = await client.get_tools()
```

Example queries with MCP:
- "Find me a direct flight from San Francisco to Tokyo on March 31st"
- "Search for flights from London to Paris next week"
- "What are the cheapest flights to Barcelona?"

## Adding Custom Tools

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description for the LLM."""
    return f"Result: {param}"

# Add to LOCAL_TOOLS in traced_agent.py
LOCAL_TOOLS = [calculator, get_weather, search, my_tool]
```

## MLflow Prompt Registry

Register and manage versioned prompts centrally in MLflow:

```bash
# Register all prompt variants (English, Spanish, Bilingual)
python register_prompt.py

# List registered prompts
python register_prompt.py --list

# Load a specific prompt version
python register_prompt.py --load agent-system-prompt-es --version 1
```

Run the agent with different languages:

```bash
# English (default)
python run_tracing_demo_autolog_prompt.py

# Spanish - responds in Spanish
python run_tracing_demo_autolog_prompt.py --lang es

# Bilingual - responds in user's language
python run_tracing_demo_autolog_prompt.py --lang bilingual

# Use specific prompt version
python run_tracing_demo_autolog_prompt.py --lang es --prompt-version 2
```

### Prompt Registry UI

View registered prompts with version history and tags:

![Prompt Registry - Version List](../docs/prompt1.png)

View traces linked to specific prompt versions:

![Prompt Registry - Linked Traces](../docs/prompt2.png)

## Running Evaluation

```bash
python evaluate_agent.py
```

Results appear in the MLflow UI under the **Evaluation** tab.

## API Credits

- Weather data: [Open-Meteo](https://open-meteo.com/) - Free weather API
- Web search: [DuckDuckGo](https://duckduckgo.com/) - Privacy-focused search
- Travel search: [Kiwi MCP](https://mcp.kiwi.com) - Free MCP server
