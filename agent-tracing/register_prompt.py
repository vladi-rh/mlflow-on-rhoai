#!/usr/bin/env python3
"""
Register agent system prompts in MLflow Prompt Registry.

This script registers versioned prompts that can be loaded by agents at runtime.
Prompts are stored in MLflow and can be versioned, tagged, and retrieved by name.

Usage:
    python register_prompt.py                    # Register default prompts
    python register_prompt.py --list             # List registered prompts
    python register_prompt.py --version 2        # Load specific version

Environment:
    MLFLOW_TRACKING_URI: MLflow server URI
    MLFLOW_TRACKING_TOKEN: Auth token for RHOAI
"""

import os
import sys
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv(override=True)

# Check auth before importing mlflow
if not os.environ.get("MLFLOW_TRACKING_TOKEN"):
    print("ERROR: MLFLOW_TRACKING_TOKEN not set.")
    print("Run: export MLFLOW_TRACKING_TOKEN=$(oc whoami --show-token)")
    sys.exit(1)

import mlflow

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

AGENT_SYSTEM_PROMPT_EN = """You are a helpful AI assistant with access to tools.

When answering questions:
1. Think through what tools might help answer the question
2. Use tools when they would provide useful information
3. Synthesize the results into a clear, helpful response
4. Be concise but thorough in your explanations

You have access to tools for:
- Calculations (add, subtract, multiply, divide, sqrt, power)
- Real-time weather lookups via Open-Meteo API
- Web search via DuckDuckGo
- Travel/flight information via MCP (if enabled)"""

AGENT_SYSTEM_PROMPT_ES = """Eres un asistente de IA útil con acceso a herramientas.

Al responder preguntas:
1. Piensa qué herramientas podrían ayudar a responder la pregunta
2. Usa las herramientas cuando proporcionen información útil
3. Sintetiza los resultados en una respuesta clara y útil
4. Sé conciso pero completo en tus explicaciones

Tienes acceso a herramientas para:
- Cálculos (suma, resta, multiplicación, división, raíz cuadrada, potencia)
- Consultas del clima en tiempo real via Open-Meteo API
- Búsqueda web via DuckDuckGo
- Información de viajes/vuelos via MCP (si está habilitado)

IMPORTANTE: Responde siempre en español."""

AGENT_SYSTEM_PROMPT_BILINGUAL = """You are a helpful AI assistant with access to tools.
Eres un asistente de IA útil con acceso a herramientas.

When answering questions / Al responder preguntas:
1. Think through what tools might help / Piensa qué herramientas pueden ayudar
2. Use tools when useful / Usa herramientas cuando sean útiles
3. Synthesize into a clear response / Sintetiza en una respuesta clara
4. Be concise but thorough / Sé conciso pero completo

Available tools / Herramientas disponibles:
- Calculator / Calculadora (add, subtract, multiply, divide, sqrt, power)
- Weather / Clima (Open-Meteo API)
- Web search / Búsqueda web (DuckDuckGo)
- Travel / Viajes (MCP flights - if enabled)

Respond in the language the user uses. If the user writes in Spanish, respond in Spanish.
Responde en el idioma que use el usuario. Si el usuario escribe en español, responde en español."""


# =============================================================================
# PROMPT REGISTRY FUNCTIONS
# =============================================================================

def setup_mlflow():
    """Configure MLflow connection."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"MLflow URI: {mlflow_uri}")


def register_prompt(name: str, template: str, language: str, commit_message: str):
    """Register a prompt in MLflow Prompt Registry."""
    try:
        prompt = mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
            tags={
                "language": language,
                "type": "system-prompt",
                "agent": "langgraph-react",
                "tools": "calculator,weather,search,mcp",
            },
        )
        print(f"✓ Registered prompt: {name} (version {prompt.version})")
        return prompt
    except Exception as e:
        # Check if prompt already exists, then update
        if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"  Prompt '{name}' exists, creating new version...")
            prompt = mlflow.genai.register_prompt(
                name=name,
                template=template,
                commit_message=f"Update: {commit_message}",
                tags={
                    "language": language,
                    "type": "system-prompt",
                    "agent": "langgraph-react",
                    "tools": "calculator,weather,search,mcp",
                },
            )
            print(f"✓ Updated prompt: {name} (version {prompt.version})")
            return prompt
        else:
            print(f"✗ Failed to register {name}: {e}")
            return None


def load_prompt(name: str, version: int = None):
    """Load a prompt from MLflow Prompt Registry."""
    try:
        if version:
            prompt = mlflow.genai.load_prompt(f"prompts:/{name}/{version}")
        else:
            prompt = mlflow.genai.load_prompt(f"prompts:/{name}/latest")
        print(f"✓ Loaded prompt: {name} (version {prompt.version})")
        return prompt
    except Exception as e:
        print(f"✗ Failed to load {name}: {e}")
        return None


def list_prompts():
    """List all registered prompts."""
    try:
        from mlflow import MlflowClient
        client = MlflowClient()

        # Search for registered models that are prompts
        # Note: MLflow stores prompts as a type of registered model
        print("\nRegistered Prompts:")
        print("-" * 60)

        # Try to load known prompt names
        prompt_names = [
            "agent-system-prompt-en",
            "agent-system-prompt-es",
            "agent-system-prompt-bilingual",
        ]

        for name in prompt_names:
            try:
                prompt = mlflow.genai.load_prompt(f"prompts:/{name}/latest")
                print(f"  {name}: version {prompt.version}")
            except Exception:
                pass

    except Exception as e:
        print(f"Error listing prompts: {e}")


def register_all_prompts():
    """Register all prompt variants."""
    print("\n" + "=" * 60)
    print("Registering Agent System Prompts")
    print("=" * 60 + "\n")

    prompts = [
        ("agent-system-prompt-en", AGENT_SYSTEM_PROMPT_EN, "en", "English system prompt for ReAct agent"),
        ("agent-system-prompt-es", AGENT_SYSTEM_PROMPT_ES, "es", "Spanish system prompt for ReAct agent"),
        ("agent-system-prompt-bilingual", AGENT_SYSTEM_PROMPT_BILINGUAL, "bilingual", "Bilingual system prompt"),
    ]

    registered = []
    for name, template, lang, message in prompts:
        prompt = register_prompt(name, template, lang, message)
        if prompt:
            registered.append(prompt)

    print(f"\n✓ Registered {len(registered)}/{len(prompts)} prompts")
    return registered


def register_versioned_prompts():
    """
    Register multiple versions of the same prompt for comparison.

    This creates:
    - agent-system-prompt v1: English only
    - agent-system-prompt v2: Bilingual (responds in user's language)
    """
    print("\n" + "=" * 60)
    print("Registering Versioned Prompts for Comparison")
    print("=" * 60 + "\n")

    # Version 1: English only
    print("Registering Version 1 (English)...")
    try:
        v1 = mlflow.genai.register_prompt(
            name="agent-system-prompt",
            template=AGENT_SYSTEM_PROMPT_EN,
            commit_message="v1: English-only system prompt",
            tags={
                "language": "en",
                "version_type": "english-only",
                "type": "system-prompt",
                "agent": "langgraph-react",
            },
        )
        print(f"✓ Registered: agent-system-prompt v{v1.version} (English)")
    except Exception as e:
        if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
            print("  Prompt already exists, skipping v1...")
            v1 = None
        else:
            print(f"✗ Error: {e}")
            v1 = None

    # Version 2: Bilingual
    print("\nRegistering Version 2 (Bilingual)...")
    try:
        v2 = mlflow.genai.register_prompt(
            name="agent-system-prompt",
            template=AGENT_SYSTEM_PROMPT_BILINGUAL,
            commit_message="v2: Bilingual - responds in user's language",
            tags={
                "language": "bilingual",
                "version_type": "bilingual",
                "type": "system-prompt",
                "agent": "langgraph-react",
            },
        )
        print(f"✓ Registered: agent-system-prompt v{v2.version} (Bilingual)")
    except Exception as e:
        print(f"✗ Error registering v2: {e}")
        v2 = None

    print("\n" + "-" * 60)
    print("Comparison:")
    print("  - Version 1: English only - always responds in English")
    print("  - Version 2: Bilingual - responds in the user's language")
    print("-" * 60)
    print("\nUsage:")
    print("  python run_tracing_demo_autolog_prompt.py --prompt-version 1  # English")
    print("  python run_tracing_demo_autolog_prompt.py --prompt-version 2  # Bilingual")

    return [v1, v2]


def main():
    parser = argparse.ArgumentParser(description="MLflow Prompt Registry Manager")
    parser.add_argument("--list", action="store_true", help="List registered prompts")
    parser.add_argument("--load", type=str, help="Load a specific prompt by name")
    parser.add_argument("--version", type=int, help="Prompt version to load")
    parser.add_argument("--register", action="store_true", help="Register all prompts (default)")
    parser.add_argument("--versioned", action="store_true",
                        help="Register versioned prompts (v1=English, v2=Bilingual) for comparison")

    args = parser.parse_args()

    setup_mlflow()

    if args.list:
        list_prompts()
    elif args.load:
        prompt = load_prompt(args.load, args.version)
        if prompt:
            print(f"\nTemplate:\n{'-'*40}\n{prompt.template}\n{'-'*40}")
    elif args.versioned:
        # Register versioned prompts for comparison
        register_versioned_prompts()
    else:
        # Default: register all prompts
        register_all_prompts()


if __name__ == "__main__":
    main()
