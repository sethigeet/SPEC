import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def cache_path(namespace: str, config: dict[str, Any]) -> Path:
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Path(".cache") / namespace / f"{digest}.json"


def read_cache(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_text_generation_cache(
    namespace: str,
    label: str,
    baseline_config: dict[str, Any],
    refresh: bool,
    engine_factory: Callable[[], Any],
) -> dict[str, Any]:
    path = cache_path(namespace, baseline_config)
    if path.exists() and not refresh:
        return read_cache(path)

    action = "Refreshing" if refresh and path.exists() else "Generating"
    print(f"{action} cached {label} at {path}. This can take a while on CPU.")

    engine = engine_factory()
    payload = {
        "baseline_config": baseline_config,
        "results": [],
    }
    total_cases = len(baseline_config["cases"])
    for index, case in enumerate(baseline_config["cases"], start=1):
        print(
            f"[{label} {index}/{total_cases}] case={case['name']} max_tokens={case['max_tokens']}"
        )
        text = engine.generate(case["prompt"], max_tokens=case["max_tokens"])
        payload["results"].append(
            {
                "name": case["name"],
                "prompt": case["prompt"],
                "max_tokens": case["max_tokens"],
                "text": text,
            }
        )

    write_cache(path, payload)
    print(f"{label} cache written to {path}")
    return payload


def assert_text_case_matches(
    engine_name: str,
    engine: Any,
    case: dict[str, Any],
) -> None:
    actual = engine.generate(case["prompt"], max_tokens=case["max_tokens"])
    expected = case["text"]
    assert actual == expected, (
        f"{engine_name} diverged for case '{case['name']}'. "
        f"expected text length {len(expected)}, got {len(actual)}."
    )


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec-target-model", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--spec-draft-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--spec-revision", default="main")
    parser.add_argument("--spec-seed", type=int, default=42)
    parser.add_argument("--spec-temperature", type=float, default=0.0)
    parser.add_argument("--spec-repeat-penalty", type=float, default=1.0)
    parser.add_argument("--spec-repeat-last-n", type=int, default=64)
    parser.add_argument("--spec-gamma", type=int, default=5)


def common_cli_config_from_pytest(pytestconfig) -> dict[str, Any]:
    return {
        "target_model_id": pytestconfig.getoption("--spec-target-model"),
        "draft_model_id": pytestconfig.getoption("--spec-draft-model"),
        "revision": pytestconfig.getoption("--spec-revision"),
        "seed": pytestconfig.getoption("--spec-seed"),
        "temperature": pytestconfig.getoption("--spec-temperature"),
        "repeat_penalty": pytestconfig.getoption("--spec-repeat-penalty"),
        "repeat_last_n": pytestconfig.getoption("--spec-repeat-last-n"),
        "gamma": pytestconfig.getoption("--spec-gamma"),
    }


def common_cli_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_model_id": args.spec_target_model,
        "draft_model_id": args.spec_draft_model,
        "revision": args.spec_revision,
        "seed": args.spec_seed,
        "temperature": args.spec_temperature,
        "repeat_penalty": args.spec_repeat_penalty,
        "repeat_last_n": args.spec_repeat_last_n,
        "gamma": args.spec_gamma,
    }
