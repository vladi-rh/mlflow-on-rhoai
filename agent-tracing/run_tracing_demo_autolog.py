#!/usr/bin/env python3
"""
Runner script for LangChain/LangGraph agent with MLflow AUTOLOG tracing.

This is a SIMPLIFIED version that relies purely on mlflow.langchain.autolog()
to capture all traces automatically. Compare with run_tracing_demo.py to see
the difference between manual logging and autolog.

AUTOLOG automatically captures:
- LLM calls (input/output, tokens, latency)
- Tool invocations with arguments and results
- Agent execution flow with nested spans
- Chain/graph execution timing

NO MANUAL LOGGING:
- No mlflow.log_param() calls
- No mlflow.set_tag() calls
- No mlflow.log_metric() calls
- No mlflow.start_run() context managers

Usage:
    python run_tracing_demo_autolog.py
"""

import os
import sys
import asyncio
import logging
import warnings

# Suppress ALL warnings before any other imports
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Disable MLflow logging BEFORE importing mlflow
logging.getLogger("mlflow").disabled = True
logging.getLogger("mlflow.utils.autologging_utils").disabled = True

from dotenv import load_dotenv

# Load .env BEFORE importing mlflow so env vars are available
# Use override=True to ensure .env values take precedence over shell exports
load_dotenv(override=True)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Check for RHOAI authentication
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
if not os.environ.get("MLFLOW_TRACKING_TOKEN"):
    print("ERROR: MLFLOW_TRACKING_TOKEN not set.")
    print("Run: export MLFLOW_TRACKING_TOKEN=$(oc whoami --show-token)")
    sys.exit(1)

import mlflow

# Re-disable after import (mlflow may reset loggers)
logging.getLogger("mlflow").disabled = True
logging.getLogger("mlflow.utils.autologging_utils").disabled = True

from traced_agent import (
    create_agent_graph,
    create_agent_with_mcp,
    get_config_from_env,
    get_mcp_config_from_env,
    LOCAL_TOOLS,
)


def setup_mlflow():
    """Configure MLflow for autolog tracing."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "agent-tracing-autolog")

    mlflow.set_tracking_uri(mlflow_uri)

    try:
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else "new"
    except Exception as e:
        print(f"Warning: Could not set experiment '{experiment_name}': {e}")
        experiment_id = "default"

    # Enable autolog for LangChain
    try:
        mlflow.langchain.autolog()
        print("✓ MLflow LangChain autolog enabled")
    except Exception as e:
        print(f"✗ LangChain autolog error: {e}")

    print(f"MLflow tracking URI: {mlflow_uri}")
    print(f"Experiment: {experiment_name}")

    return mlflow_uri


async def run_queries(agent, mcp_client=None):
    """Run example queries - autolog captures everything automatically."""

    queries = [
        "What is 25 * 17 + 89?",
        "What's the weather in Tokyo and New York?",
        "Search for information about MLflow tracing.",
    ]

    if mcp_client:
        queries.append("Find me a flight from San Francisco to Tokyo on April 15th 2026")

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("=" * 60)

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            response = result["messages"][-1].content
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"\nError: {e}")

    print(f"\n{'='*60}")
    print("Done! Check MLflow UI for traces.")
    print("=" * 60)


async def main():
    """Main entry point."""
    print("=" * 60)
    print("LangGraph Agent with MLflow AUTOLOG")
    print("=" * 60 + "\n")

    mlflow_uri = setup_mlflow()

    config = get_config_from_env()
    mcp_config = get_mcp_config_from_env()

    print(f"\nModel: {config.model}")

    # Create agent
    mcp_client = None
    if mcp_config:
        print(f"MCP enabled: {list(mcp_config.keys())}")
        try:
            agent, mcp_client = await create_agent_with_mcp(config, mcp_config)
            # Get all tools from the agent to verify MCP tools loaded
            mcp_tools = await mcp_client.get_tools()
            print(f"MCP tools loaded: {[t.name for t in mcp_tools]}")
        except Exception as e:
            print(f"MCP failed: {e}, using local tools")
            agent = create_agent_graph(config)
    else:
        agent = create_agent_graph(config)

    print(f"Local tools: {[t.name for t in LOCAL_TOOLS]}")

    await run_queries(agent, mcp_client)


if __name__ == "__main__":
    asyncio.run(main())
