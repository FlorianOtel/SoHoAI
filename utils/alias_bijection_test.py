#!/usr/bin/env python3
"""Bijection and alias resolution test for SoHoAI claude-code-* aliases.

Runs offline with no network dependency. Test sets dummy env vars before
importing main to avoid missing-credential errors during app construction.

Run: python utils/alias_bijection_test.py
     (from worktree root, with venv activated)
"""

import sys
import os

# Set dummy credentials before importing main (which constructs the FastAPI app)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-bijection-test")
os.environ.setdefault("OLLAMA_API_KEY", "test-key-for-bijection-test")

# Ensure we can import from the worktree
sys.path.insert(0, ".")

try:
    from main import (
        _PROXY_EXPOSED_MODELS,
        _claude_code_alias_for,
        _claude_code_alias_to_public,
        _resolve_proxy_model,
    )
except Exception as e:
    print(f"FAIL: Could not import from main.py: {e}")
    sys.exit(1)


def test_bijection():
    """Assert _claude_code_alias_to_public(_claude_code_alias_for(k)) == k for all k."""
    failures = []
    for public_id in _PROXY_EXPOSED_MODELS:
        alias = _claude_code_alias_for(public_id)
        resolved = _claude_code_alias_to_public(alias)
        if resolved != public_id:
            failures.append(
                f"Bijection broken for {public_id}: "
                f"alias={alias}, resolved={resolved}"
            )
    return failures


def test_non_anthropic_alias_prefix():
    """Assert that non-Anthropic aliases start with claude-code-."""
    failures = []
    for public_id in _PROXY_EXPOSED_MODELS:
        if not public_id.startswith("anthropic/") and not public_id.startswith("claude-"):
            alias = _claude_code_alias_for(public_id)
            if not alias.startswith("claude-code-"):
                failures.append(
                    f"Non-Anthropic model {public_id} has alias {alias} "
                    f"(expected to start with claude-code-)"
                )
    return failures


def test_anthropic_native_aliases():
    """Assert that anthropic/claude-* keys produce bare model IDs (no double-prefix)."""
    failures = []
    for public_id in _PROXY_EXPOSED_MODELS:
        if public_id.startswith("anthropic/"):
            alias = _claude_code_alias_for(public_id)
            expected = public_id.split("/", 1)[1]
            if alias != expected:
                failures.append(
                    f"Anthropic native {public_id} produced alias {alias}, "
                    f"expected {expected}"
                )
    return failures


def test_resolution():
    """Assert that _resolve_proxy_model(_claude_code_alias_for(k)) == _PROXY_EXPOSED_MODELS[k]."""
    failures = []
    for public_id in _PROXY_EXPOSED_MODELS:
        alias = _claude_code_alias_for(public_id)
        resolved_alias = _resolve_proxy_model(alias)
        expected_alias = _PROXY_EXPOSED_MODELS[public_id]
        if resolved_alias != expected_alias:
            failures.append(
                f"Resolution failed for {public_id}: "
                f"alias={alias}, resolved_to={resolved_alias}, expected={expected_alias}"
            )
    return failures


def main():
    """Run all tests and report results."""
    all_failures = []

    print("Running bijection tests...")
    all_failures.extend(test_bijection())

    print("Running non-Anthropic alias prefix test...")
    all_failures.extend(test_non_anthropic_alias_prefix())

    print("Running Anthropic native alias test...")
    all_failures.extend(test_anthropic_native_aliases())

    print("Running resolution test...")
    all_failures.extend(test_resolution())

    if all_failures:
        print("\nFAIL: The following assertions failed:")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1

    # Success
    model_count = len(_PROXY_EXPOSED_MODELS)
    anthropic_count = sum(1 for k in _PROXY_EXPOSED_MODELS if k.startswith("anthropic/") or k.startswith("claude-"))
    non_anthropic_count = model_count - anthropic_count
    print(
        f"\nOK: bijection verified for {model_count} models "
        f"({anthropic_count} Anthropic, {non_anthropic_count} non-Anthropic), "
        f"all aliases resolve correctly"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
