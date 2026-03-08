"""End-to-end generation parity harness with cached target-model baselines."""

import argparse
from typing import Any

import pytest
from SPEC.SPEC import engines
from spec_test_utils import (
    add_common_cli_args,
    assert_text_case_matches,
    common_cli_config_from_args,
    common_cli_config_from_pytest,
    ensure_text_generation_cache,
)


def _default_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "tiny_completion",
            "prompt": "The sky is",
            "max_tokens": 8,
        },
        {
            "name": "tiny_code",
            "prompt": "def add(",
            "max_tokens": 10,
        },
        {
            "name": "tiny_list",
            "prompt": "Benefits:",
            "max_tokens": 8,
        },
    ]


def _baseline_config(
    cases: list[dict[str, Any]],
    cli: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_model_id": cli["target_model_id"],
        "revision": cli["revision"],
        "seed": cli["seed"],
        "temperature": cli["temperature"],
        "top_p": None,
        "top_k": None,
        "repeat_penalty": cli["repeat_penalty"],
        "repeat_last_n": cli["repeat_last_n"],
        "cases": cases,
    }


def _cli_config_from_pytest(pytestconfig) -> dict[str, Any]:
    cli = common_cli_config_from_pytest(pytestconfig)
    cli["enabled"] = pytestconfig.getoption("--spec-e2e")
    cli["refresh_baseline"] = pytestconfig.getoption("--spec-e2e-refresh-baseline")
    return cli


def _cli_config_from_args(argv: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-e2e", action="store_true")
    parser.add_argument("--spec-e2e-refresh-baseline", action="store_true")
    add_common_cli_args(parser)
    args = parser.parse_args(argv)
    cli = common_cli_config_from_args(args)
    cli["enabled"] = args.spec_e2e
    cli["refresh_baseline"] = args.spec_e2e_refresh_baseline
    return cli


def _run_e2e(cli: dict[str, Any]) -> None:
    cases = _default_cases()
    baseline_config = _baseline_config(cases, cli)
    baseline_payload = ensure_text_generation_cache(
        "spec_e2e",
        "target-model baseline",
        baseline_config,
        cli["refresh_baseline"],
        lambda: engines.TargetModelEngine(
            model_id=baseline_config["target_model_id"],
            temperature=baseline_config["temperature"],
            top_p=baseline_config["top_p"],
            top_k=baseline_config["top_k"],
            seed=baseline_config["seed"],
            repeat_penalty=baseline_config["repeat_penalty"],
            repeat_last_n=baseline_config["repeat_last_n"],
            revision=baseline_config["revision"],
        ),
    )

    common_kwargs = {
        "draft_model_id": cli["draft_model_id"],
        "target_model_id": baseline_config["target_model_id"],
        "gamma": cli["gamma"],
        "seed": baseline_config["seed"],
        "revision": baseline_config["revision"],
    }

    sync_engine = engines.SpecDecodingEngine(
        temperature=baseline_config["temperature"],
        top_p=baseline_config["top_p"],
        top_k=baseline_config["top_k"],
        repeat_penalty=baseline_config["repeat_penalty"],
        repeat_last_n=baseline_config["repeat_last_n"],
        **common_kwargs,
    )
    async_engine = engines.AsyncSpecDecodingEngine(**common_kwargs)

    for case in baseline_payload["results"]:
        assert_text_case_matches("SpecDecodingEngine", sync_engine, case)
        assert_text_case_matches("AsyncSpecDecodingEngine", async_engine, case)


def test_engine_outputs_match_cached_target_baseline(pytestconfig):
    cli = _cli_config_from_pytest(pytestconfig)
    if not cli["enabled"]:
        pytest.skip("pass --spec-e2e to run cached end-to-end generation parity tests")
    _run_e2e(cli)


if __name__ == "__main__":
    import sys

    cli = _cli_config_from_args(sys.argv[1:])
    _run_e2e(cli)
    print("✓ test_engine_outputs_match_cached_target_baseline")
