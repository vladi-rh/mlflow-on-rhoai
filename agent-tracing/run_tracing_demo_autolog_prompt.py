#!/usr/bin/env python3
"""
Runner script with MLflow Prompt Registry integration.

This version loads the system prompt from MLflow Prompt Registry,
enabling versioned, centralized prompt management.

Features:
- Loads prompts from MLflow Prompt Registry
- Supports multiple languages (en, es, bilingual)
- All tracing via mlflow.langchain.autolog()

Usage:
    python run_tracing_demo_autolog_prompt.py                    # English (default)
    python run_tracing_demo_autolog_prompt.py --lang es          # Spanish
    python run_tracing_demo_autolog_prompt.py --lang bilingual   # Bilingual
    python run_tracing_demo_autolog_prompt.py --prompt-version 2 # Specific version
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

# Load .env BEFORE importing mlflow
load_dotenv(override=True)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Check for RHOAI authentication
if not os.environ.get("MLFLOW_TRACKING_TOKEN"):
    print("ERROR: MLFLOW_TRACKING_TOKEN not set.")
    print("Run: export MLFLOW_TRACKING_TOKEN=$(oc whoami --show-token)")
    sys.exit(1)

import mlflow

# Re-disable after import
logging.getLogger("mlflow").disabled = True
logging.getLogger("mlflow.utils.autologging_utils").disabled = True

from traced_agent import (
    create_agent_graph,
    create_agent_with_mcp,
    get_config_from_env,
    get_mcp_config_from_env,
    LOCAL_TOOLS,
    DEFAULT_SYSTEM_PROMPT,
)


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

def load_prompt_from_registry(language: str = "en", version: int = None) -> str:
    """
    Load system prompt from MLflow Prompt Registry.

    Args:
        language: "en", "es", or "bilingual"
        version: Specific version number (None = latest)
                 When version is specified, loads from unified "agent-system-prompt"
                 (v1=English, v2=Bilingual)

    Returns:
        Prompt template string
    """
    # If version is specified, use the unified versioned prompt
    if version:
        prompt_name = "agent-system-prompt"
        uri = f"prompts:/{prompt_name}/{version}"
        version_desc = f"v{version} ({'English' if version == 1 else 'Bilingual' if version == 2 else 'custom'})"
    else:
        # Use language-specific prompts
        prompt_names = {
            "en": "agent-system-prompt-en",
            "es": "agent-system-prompt-es",
            "bilingual": "agent-system-prompt-bilingual",
        }
        prompt_name = prompt_names.get(language, "agent-system-prompt-en")
        uri = f"prompts:/{prompt_name}/latest"
        version_desc = None

    try:
        prompt = mlflow.genai.load_prompt(uri)
        if version_desc:
            print(f"✓ Loaded prompt: {prompt_name} {version_desc}")
        else:
            print(f"✓ Loaded prompt: {prompt_name} (version {prompt.version})")
        return prompt.template
    except Exception as e:
        print(f"⚠ Could not load prompt from registry: {e}")
        print("  Falling back to default prompt")
        return DEFAULT_SYSTEM_PROMPT


# =============================================================================
# MLFLOW SETUP
# =============================================================================

def setup_mlflow():
    """Configure MLflow for autolog tracing."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "agent-tracing-prompt-registry")

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


# =============================================================================
# QUERY EXECUTION
# =============================================================================

def get_queries(language: str, mcp_enabled: bool, queries_lang: str = None) -> list:
    """
    Get example queries based on language.

    Args:
        language: Prompt language (en, es, bilingual)
        mcp_enabled: Whether MCP tools are available
        queries_lang: Override language for queries (to test bilingual prompt)
    """
    # Use queries_lang if specified, otherwise match prompt language
    lang = queries_lang or language

    if lang == "es":
        queries = [
            "¿Cuánto es 25 * 17 + 89?",
            "¿Cómo está el clima en Madrid y Barcelona?",
            "Busca información sobre MLflow tracing.",
        ]
        if mcp_enabled:
            queries.append("Busca vuelos de Madrid a Nueva York para el 15 de abril 2026")
    elif lang == "mixed":
        # Mixed queries to test bilingual prompt
        queries = [
            "What is 25 * 17 + 89?",
            "¿Cómo está el clima en Tokyo?",
            "Search for information about MLflow tracing.",
            "¿Cuánto es la raíz cuadrada de 144?",
        ]
        if mcp_enabled:
            queries.append("Busca vuelos de San Francisco a Tokyo para el 15 de abril 2026")
    else:  # English
        queries = [
            "What is 25 * 17 + 89?",
            "What's the weather in Tokyo and New York?",
            "Search for information about MLflow tracing.",
        ]
        if mcp_enabled:
            queries.append("Find me a flight from San Francisco to Tokyo on April 15th 2026")

    return queries


async def run_queries(agent, queries: list):
    """Run queries - autolog captures everything automatically."""
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


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent with MLflow Prompt Registry")
    parser.add_argument(
        "--lang", "-l",
        choices=["en", "es", "bilingual"],
        default="en",
        help="Prompt language (default: en)"
    )
    parser.add_argument(
        "--prompt-version", "-v",
        type=int,
        default=None,
        help="Specific prompt version (default: latest)"
    )
    parser.add_argument(
        "--queries-lang", "-q",
        choices=["en", "es", "mixed"],
        default=None,
        help="Override query language (useful for testing bilingual prompt with Spanish queries)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LangGraph Agent with MLflow Prompt Registry")
    print("=" * 60 + "\n")

    mlflow_uri = setup_mlflow()

    # Load prompt from registry
    print(f"\nLanguage: {args.lang}")
    system_prompt = load_prompt_from_registry(args.lang, args.prompt_version)

    config = get_config_from_env()
    mcp_config = get_mcp_config_from_env()

    print(f"Model: {config.model}")

    # Create agent with the loaded prompt
    mcp_client = None
    if mcp_config:
        print(f"MCP enabled: {list(mcp_config.keys())}")
        try:
            agent, mcp_client = await create_agent_with_mcp(
                config, mcp_config, system_prompt=system_prompt
            )
            mcp_tools = await mcp_client.get_tools()
            print(f"MCP tools loaded: {[t.name for t in mcp_tools]}")
        except Exception as e:
            print(f"MCP failed: {e}, using local tools")
            agent = create_agent_graph(config, system_prompt=system_prompt)
    else:
        agent = create_agent_graph(config, system_prompt=system_prompt)

    print(f"Local tools: {[t.name for t in LOCAL_TOOLS]}")

    # Get language-appropriate queries
    if args.queries_lang:
        print(f"Query language: {args.queries_lang}")
    queries = get_queries(args.lang, mcp_client is not None, args.queries_lang)

    await run_queries(agent, queries)


if __name__ == "__main__":
    asyncio.run(main())
