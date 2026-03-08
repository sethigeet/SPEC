"""Parity tests between Candle's bare Llama model and SPEC's paged-KV model."""

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
            "name": "tiny_phrase",
            "prompt": "Hello",
            "max_tokens": 6,
        },
        {
            "name": "tiny_symbols",
            "prompt": "1 +",
            "max_tokens": 6,
        },
        {
            "name": "tiny_heading",
            "prompt": "Rust:",
            "max_tokens": 6,
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
    cli["enabled"] = pytestconfig.getoption("--spec-model-parity")
    cli["refresh_baseline"] = pytestconfig.getoption("--spec-model-parity-refresh-baseline")
    return cli


def _cli_config_from_args(argv: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-model-parity", action="store_true")
    parser.add_argument("--spec-model-parity-refresh-baseline", action="store_true")
    add_common_cli_args(parser)
    args = parser.parse_args(argv)
    cli = common_cli_config_from_args(args)
    cli["enabled"] = args.spec_model_parity
    cli["refresh_baseline"] = args.spec_model_parity_refresh_baseline
    return cli


def _run_model_parity(cli: dict[str, Any]) -> None:
    cases = _default_cases()
    baseline_config = _baseline_config(cases, cli)
    baseline_payload = ensure_text_generation_cache(
        "spec_model_parity",
        "bare-model baseline",
        baseline_config,
        cli["refresh_baseline"],
        lambda: engines.BareModelEngine(
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

    custom_engine = engines.TargetModelEngine(
        model_id=baseline_config["target_model_id"],
        temperature=baseline_config["temperature"],
        top_p=baseline_config["top_p"],
        top_k=baseline_config["top_k"],
        seed=baseline_config["seed"],
        repeat_penalty=baseline_config["repeat_penalty"],
        repeat_last_n=baseline_config["repeat_last_n"],
        revision=baseline_config["revision"],
    )

    for case in baseline_payload["results"]:
        assert_text_case_matches("TargetModelEngine", custom_engine, case)


def test_custom_model_matches_bare_llama(pytestconfig):
    cli = _cli_config_from_pytest(pytestconfig)
    if not cli["enabled"]:
        pytest.skip("pass --spec-model-parity to run bare-model parity tests")
    _run_model_parity(cli)


if __name__ == "__main__":
    import sys

    cli = _cli_config_from_args(sys.argv[1:])
    _run_model_parity(cli)
    print("✓ test_custom_model_matches_bare_llama")
