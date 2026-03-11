#!/usr/bin/env python3
"""
Runner script for LangChain/LangGraph agent with MLflow tracing.

This script:
1. Sets up MLflow tracing (manual spans for RHOAI compatibility)
2. Creates a LangGraph agent with tools (local + optional MCP)
3. Runs the agent with sample queries or in interactive mode
4. All traces appear in MLflow's Traces tab

Usage:
    python run_tracing_demo.py           # Batch mode with example queries
    python run_tracing_demo.py -i        # Interactive mode

Environment variables (see .env.example):
    MAAS_API_KEY: API key for MaaS endpoint
    MAAS_BASE_URL: MaaS endpoint URL
    MAAS_MODEL: Model name (default: Llama-4-Scout-17B-16E-W4A16)
    MLFLOW_TRACKING_URI: MLflow server URI
    MLFLOW_EXPERIMENT_NAME: Experiment name (default: langchain-agent-tracing)
    MCP_SERVER_ENABLED: Set to 'true' to enable MCP tools
    MCP_SERVER_URL: MCP server URL (default: https://mcp.kiwi.com)
"""

import os
import sys
import asyncio
import json
from dotenv import load_dotenv

# Load .env BEFORE importing mlflow so env vars are available
load_dotenv()

# Disable ALL warnings first
import warnings
warnings.filterwarnings("ignore")

import urllib3
import logging
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Suppress MLflow autolog context warnings (known issue with async LangGraph)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.CRITICAL)
logging.getLogger("mlflow").setLevel(logging.ERROR)

# Check for RHOAI authentication (optional for ODH servers)
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
if "rhoai" in mlflow_uri.lower() or "redhat" in mlflow_uri.lower():
    if not os.environ.get("MLFLOW_TRACKING_TOKEN"):
        print("ERROR: MLFLOW_TRACKING_TOKEN not set (required for RHOAI).")
        print("Run: export MLFLOW_TRACKING_TOKEN=$(oc whoami --show-token)")
        sys.exit(1)

import mlflow

from traced_agent import (
    AgentConfig,
    create_agent_graph,
    create_agent_with_mcp,
    get_config_from_env,
    get_mcp_config_from_env,
    LOCAL_TOOLS,
)


def setup_mlflow():
    """
    Configure MLflow for tracing.

    Returns the tracking URI for display purposes.
    """
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "langchain-agent-tracing")

    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_uri)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    # Try to enable autolog for LangChain (may not work with all MLflow versions)
    try:
        mlflow.langchain.autolog()
        print("MLflow LangChain autolog enabled")
    except Exception as e:
        print(f"Note: LangChain autolog not available: {e}")

    print(f"MLflow tracking URI: {mlflow_uri}")
    print(f"Experiment: {experiment_name} (ID: {experiment_id})")

    return mlflow_uri, experiment_id


def extract_trace_info(result):
    """Extract trace information from agent result for logging."""
    trace_info = {
        "num_messages": len(result.get("messages", [])),
        "message_types": [],
    }

    for msg in result.get("messages", []):
        msg_type = type(msg).__name__
        trace_info["message_types"].append(msg_type)

        # Extract tool calls if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            trace_info["tool_calls"] = [
                {"name": tc.get("name", "unknown"), "id": tc.get("id", "")}
                for tc in msg.tool_calls
            ]

    return trace_info


async def run_example_queries_async(agent, mlflow_uri, mcp_client=None):
    """Run example queries asynchronously with manual MLflow tracing."""

    example_queries = [
        "What is 25 * 17 + 89?",
        "What's the weather in Tokyo and New York?",
        "Search for information about MLflow tracing.",
    ]

    # Add travel query if MCP is enabled
    if mcp_client:
        example_queries.append(
            "Find me a direct flight from San Francisco to Tokyo on April 15th 2026"
        )

    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("=" * 60)

        try:
            # Start an MLflow run for each query
            with mlflow.start_run(run_name=f"agent-query-{i}"):
                # Log input
                mlflow.log_param("query", query[:250])
                mlflow.set_tag("query_index", str(i))

                # Invoke the agent
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": query}]}
                )

                # Extract and log trace info
                trace_info = extract_trace_info(result)
                mlflow.log_param("num_messages", trace_info["num_messages"])

                if "tool_calls" in trace_info:
                    mlflow.set_tag("tools_used", json.dumps([tc["name"] for tc in trace_info["tool_calls"]]))

                # Extract response
                response = result["messages"][-1].content
                print(f"\nResponse:\n{response}")

                # Log response info
                mlflow.set_tag("response_length", str(len(response) if response else 0))
                mlflow.set_tag("response_preview", (response[:500] if response else "")[:500])

                # Log metrics
                mlflow.log_metric("response_length", len(response) if response else 0)
                mlflow.log_metric("message_count", trace_info["num_messages"])

        except Exception as e:
            print(f"\nError: {e}")
            # Still try to log the error
            try:
                with mlflow.start_run(run_name=f"agent-query-{i}-error"):
                    mlflow.log_param("query", query[:250])
                    mlflow.set_tag("error", str(e)[:500])
                    mlflow.log_metric("success", 0)
            except Exception:
                pass

    print(f"\n{'='*60}")
    print("All queries completed! Check MLflow UI for traces.")
    print(f"Open {mlflow_uri} in your browser.")
    print("Navigate to the 'Traces' tab to see the execution traces.")
    print("=" * 60)


def run_example_queries_sync(agent, mlflow_uri):
    """Run example queries synchronously with manual MLflow tracing."""

    example_queries = [
        "What is 25 * 17 + 89?",
        "What's the weather in Tokyo and New York?",
        "Search for information about MLflow tracing.",
    ]

    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("=" * 60)

        try:
            # Start an MLflow run for each query
            with mlflow.start_run(run_name=f"agent-query-{i}"):
                # Log input
                mlflow.log_param("query", query[:250])
                mlflow.set_tag("query_index", str(i))

                # Invoke the agent
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )

                # Extract and log trace info
                trace_info = extract_trace_info(result)
                mlflow.log_param("num_messages", trace_info["num_messages"])

                if "tool_calls" in trace_info:
                    mlflow.set_tag("tools_used", json.dumps([tc["name"] for tc in trace_info["tool_calls"]]))

                # Extract response
                response = result["messages"][-1].content
                print(f"\nResponse:\n{response}")

                # Log response info
                mlflow.set_tag("response_length", str(len(response) if response else 0))
                mlflow.set_tag("response_preview", (response[:500] if response else "")[:500])

                # Log metrics
                mlflow.log_metric("response_length", len(response) if response else 0)
                mlflow.log_metric("message_count", trace_info["num_messages"])

        except Exception as e:
            print(f"\nError: {e}")

    print(f"\n{'='*60}")
    print("All queries completed! Check MLflow UI for traces.")
    print(f"Open {mlflow_uri} in your browser.")
    print("Navigate to the 'Traces' tab to see the execution traces.")
    print("=" * 60)


async def run_interactive_async(agent, mlflow_uri, mcp_client=None):
    """Run interactive mode with async agent."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60 + "\n")

    query_num = 0
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            query_num += 1
            with mlflow.start_run(run_name=f"interactive-{query_num}"):
                mlflow.log_param("query", user_input[:250])

                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]}
                )

                response = result["messages"][-1].content
                print(f"\nAssistant: {response}\n")

                mlflow.log_metric("response_length", len(response) if response else 0)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def run_interactive_sync(agent, mlflow_uri):
    """Run interactive mode synchronously."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60 + "\n")

    query_num = 0
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            query_num += 1
            with mlflow.start_run(run_name=f"interactive-{query_num}"):
                mlflow.log_param("query", user_input[:250])

                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]}
                )

                response = result["messages"][-1].content
                print(f"\nAssistant: {response}\n")

                mlflow.log_metric("response_length", len(response) if response else 0)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


async def main_async():
    """Main async entry point."""
    # Note: load_dotenv() already called at module import

    print("=" * 60)
    print("LangChain/LangGraph Agent with MLflow Tracing")
    print("=" * 60 + "\n")

    # Setup MLflow tracing
    mlflow_uri, experiment_id = setup_mlflow()

    # Get configurations
    config = get_config_from_env()
    mcp_config = get_mcp_config_from_env()

    print(f"\nModel: {config.model}")
    print(f"Base URL: {config.base_url}")

    # Validate API key
    if not config.api_key:
        print("\nWarning: MAAS_API_KEY not set. API calls may fail.")

    # Create agent (with or without MCP)
    mcp_client = None
    if mcp_config:
        print(f"MCP enabled: {list(mcp_config.keys())}")
        try:
            agent, mcp_client = await create_agent_with_mcp(config, mcp_config)
            print("MCP tools loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load MCP tools: {e}")
            print("Falling back to local tools only")
            agent = create_agent_graph(config)
    else:
        print("MCP disabled, using local tools only")
        agent = create_agent_graph(config)

    print(f"Local tools: {[t.name for t in LOCAL_TOOLS]}")

    # Run based on mode
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    try:
        if interactive:
            await run_interactive_async(agent, mlflow_uri, mcp_client)
        else:
            await run_example_queries_async(agent, mlflow_uri, mcp_client)
    finally:
        pass


def main_sync():
    """Synchronous entry point (no MCP)."""
    # Note: load_dotenv() already called at module import

    print("=" * 60)
    print("LangChain/LangGraph Agent with MLflow Tracing")
    print("=" * 60 + "\n")

    # Setup MLflow tracing
    mlflow_uri, experiment_id = setup_mlflow()

    # Get configuration
    config = get_config_from_env()

    print(f"\nModel: {config.model}")
    print(f"Base URL: {config.base_url}")

    # Validate API key
    if not config.api_key:
        print("\nWarning: MAAS_API_KEY not set. API calls may fail.")

    # Create agent with local tools only
    print("Using local tools only (sync mode)")
    agent = create_agent_graph(config)
    print(f"Local tools: {[t.name for t in LOCAL_TOOLS]}")

    # Run based on mode
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    if interactive:
        run_interactive_sync(agent, mlflow_uri)
    else:
        run_example_queries_sync(agent, mlflow_uri)


def main():
    """Main entry point - chooses async or sync based on MCP config."""
    # Note: load_dotenv() already called at module import

    # Check if MCP is enabled to decide async vs sync
    mcp_enabled = os.environ.get("MCP_SERVER_ENABLED", "false").lower() == "true"

    if mcp_enabled:
        # Use async for MCP support
        asyncio.run(main_async())
    else:
        # Use sync for simpler execution without MCP
        main_sync()


if __name__ == "__main__":
    main()
