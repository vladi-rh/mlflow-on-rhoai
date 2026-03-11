#!/usr/bin/env python3
"""
MLflow GenAI Evaluation for LangChain/LangGraph agent.

This script shows how to:
1. Create an evaluation dataset
2. Run the agent on evaluation inputs using mlflow.genai.evaluate()
3. Apply built-in and custom scorers
4. View and analyze results in MLflow's Evaluation UI

Usage:
    python evaluate_agent.py

Environment variables (see .env.example):
    MAAS_API_KEY: API key for MaaS endpoint
    MAAS_BASE_URL: MaaS endpoint URL
    MAAS_MODEL: Model name
    MLFLOW_TRACKING_URI: MLflow server URI
    MLFLOW_TRACKING_TOKEN: OpenShift token for RHOAI auth
"""

import os
import sys
import warnings
import logging

# Suppress warnings before other imports
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# Load .env with override to ensure env values take precedence
load_dotenv(override=True)

# Suppress MLflow autolog context warnings
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.CRITICAL)

import mlflow
from mlflow.genai.scorers import scorer

from traced_agent import (
    AgentConfig,
    create_agent_graph,
    get_config_from_env,
    LOCAL_TOOLS,
)


# =============================================================================
# Evaluation Dataset
# =============================================================================

# Define evaluation examples using the mlflow.genai.evaluate() schema:
# - inputs: dict of input arguments to the predict function
# - expectations: ground truth values for scoring
EVAL_DATASET = [
    {
        "inputs": {"user_message": "What is 25 * 17 + 89?"},
        "expectations": {"expected_answer": "514"},
    },
    {
        "inputs": {"user_message": "What is the square root of 144?"},
        "expectations": {"expected_answer": "12"},
    },
    {
        "inputs": {"user_message": "Calculate 256 divided by 16"},
        "expectations": {"expected_answer": "16"},
    },
    {
        "inputs": {"user_message": "What's the weather in Tokyo?"},
        "expectations": {"expected_answer": "Temperature"},
    },
    {
        "inputs": {"user_message": "What's the weather in New York?"},
        "expectations": {"expected_answer": "Temperature"},
    },
    {
        "inputs": {"user_message": "Search for information about MLflow tracing"},
        "expectations": {"expected_answer": "MLflow"},
    },
    {
        "inputs": {"user_message": "Multiply 33 by 3 and then add 1"},
        "expectations": {"expected_answer": "100"},
    },
    {
        "inputs": {"user_message": "What is 1000 minus 273?"},
        "expectations": {"expected_answer": "727"},
    },
]


# =============================================================================
# Custom Scorers using @scorer decorator
# =============================================================================


@scorer
def contains_expected(
    inputs: dict,
    outputs: str,
    expectations: dict,
) -> bool:
    """
    Custom scorer: Check if the output contains the expected answer.

    Returns True if the expected answer appears in the output.
    """
    if outputs is None or expectations is None:
        return False

    expected = expectations.get("expected_answer", "")
    return str(expected).lower() in str(outputs).lower()


@scorer
def response_length(
    outputs: str,
) -> float:
    """
    Custom scorer: Score based on response length.

    Returns a score between 0 and 1.
    Optimal length is between 50-500 characters.
    """
    if outputs is None:
        return 0.0

    length = len(str(outputs))
    if length < 20:
        return 0.2  # Too short
    elif length < 50:
        return 0.6  # A bit short
    elif length <= 500:
        return 1.0  # Good length
    elif length <= 1000:
        return 0.8  # A bit long
    else:
        return 0.5  # Too verbose


@scorer
def has_numeric_result(
    outputs: str,
) -> bool:
    """
    Custom scorer: Check if the output contains numeric results.

    Useful for math questions.
    """
    if outputs is None:
        return False
    return any(c.isdigit() for c in str(outputs))


# =============================================================================
# Agent Setup
# =============================================================================

# Global agent instance (created once, reused for all evaluations)
_agent = None


def get_agent():
    """Get or create the LangGraph agent instance."""
    global _agent
    if _agent is None:
        config = get_config_from_env()

        if not config.api_key:
            print("Warning: MAAS_API_KEY not set. API calls may fail.")

        _agent = create_agent_graph(config)
        print(f"Agent created with model: {config.model}")
        print(f"Tools: {[t.name for t in LOCAL_TOOLS]}")

    return _agent


def predict_fn(user_message: str) -> str:
    """
    Predict function for mlflow.genai.evaluate().

    This function takes the input and returns the agent's response.
    The function signature must match the keys in the 'inputs' dict.
    """
    agent = get_agent()
    try:
        # Invoke the LangGraph agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        # Extract response from messages
        response = result["messages"][-1].content
        return response
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# Main Evaluation Logic
# =============================================================================


def run_evaluation():
    """
    Run the evaluation using mlflow.genai.evaluate().

    This uses MLflow's GenAI evaluation API which provides:
    - Built-in scorers for common metrics
    - Custom scorer support
    - Results in the Evaluation UI
    """

    # Check for required token
    if not os.environ.get("MLFLOW_TRACKING_TOKEN"):
        print("Warning: MLFLOW_TRACKING_TOKEN not set.")
        print("Run: export MLFLOW_TRACKING_TOKEN=$(oc whoami --show-token)")

    # Setup MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    # Use env experiment name or default to evaluation-specific name
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "agent-evaluation"
    )
    # Add -eval suffix if not already present
    if not experiment_name.endswith("-eval") and not experiment_name.endswith("-evaluation"):
        experiment_name = f"{experiment_name}-eval"

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Warning: Could not set experiment '{experiment_name}': {e}")
        experiment_name = "agent-evaluation"
        mlflow.set_experiment(experiment_name)

    # Enable autolog for tracing during evaluation
    try:
        mlflow.langchain.autolog()
    except Exception as e:
        print(f"Note: autolog setup: {e}")

    print("=" * 60)
    print("LangChain Agent Evaluation with mlflow.genai.evaluate()")
    print("=" * 60)
    print(f"\nMLflow tracking URI: {mlflow_uri}")
    print(f"Experiment: {experiment_name}")

    # Initialize the agent
    agent = get_agent()

    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(EVAL_DATASET)} examples...")
    print("=" * 60)

    # Define scorers to use
    scorers = [
        contains_expected,
        response_length,
        has_numeric_result,
    ]

    # Run the evaluation
    result = mlflow.genai.evaluate(
        data=EVAL_DATASET,
        predict_fn=predict_fn,
        scorers=scorers,
    )

    # Print results summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nAggregated Metrics:")
    for metric_name, metric_value in result.metrics.items():
        if isinstance(metric_value, float):
            print(f"  - {metric_name}: {metric_value:.2%}")
        else:
            print(f"  - {metric_name}: {metric_value}")

    # Print per-example results if available
    if result.tables:
        table_name = list(result.tables.keys())[0] if result.tables else None
        if table_name:
            print(f"\nPer-Example Results (from {table_name}):")
            print("-" * 60)
            for i, row in enumerate(result.tables[table_name].itertuples(), 1):
                inputs_str = getattr(row, 'inputs', str(row)[:50])
                outputs_str = getattr(row, 'outputs', '')
                output_display = (
                    str(outputs_str)[:100] + "..."
                    if len(str(outputs_str)) > 100
                    else outputs_str
                )
                print(f"\n[{i}] Input: {inputs_str}")
                print(f"    Output: {output_display}")

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"\nView results in MLflow UI:")
    print(f"  {mlflow_uri}/#/experiments (look for '{experiment_name}')")
    print(f"\nThe evaluation results will appear in the 'Evaluation' tab.")
    print("=" * 60)

    return result


def main():
    """Main entry point."""
    load_dotenv()
    run_evaluation()


if __name__ == "__main__":
    main()
