"""
Phase 0 smoke test.

PURPOSE: Prove that:
  1. Your virtual environment is set up correctly.
  2. The Google GenAI SDK is installed and importable.
  3. Your API key is loaded from .env.
  4. You can successfully reach the Gemini API and get a response.

"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to sys.path so `from src.utils.llm import ...` works
# when running this file directly with `python tests/phase0_smoke_test.py`.
# In a real package install (pip install -e .) we wouldn't need this hack,
# but for a learning project it keeps things simple.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.llm import MODEL_FLASH, call_llm


def main() -> None:
    print("Phase 0 smoke test: calling Gemini 2.5 Flash...\n")

    response = call_llm(
        model=MODEL_FLASH,
        system="Answer in one short sentence.",
        user="Tell me something interesting about the universe.",
        # max_tokens=100,
    )

    print("--- Response ---")
    print(response.text)
    print("\n--- Usage ---")
    print(f"Model:         {response.model}")
    print(f"Input tokens:  {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print("Cost:          $0.00 (free tier)")

    print("\n[OK] Environment is working !")


if __name__ == "__main__":
    main()
