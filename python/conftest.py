def pytest_addoption(parser):
    group = parser.getgroup("spec-e2e")
    group.addoption(
        "--spec-e2e",
        action="store_true",
        default=False,
        help="run cached end-to-end generation parity tests",
    )
    group.addoption(
        "--spec-e2e-refresh-baseline",
        action="store_true",
        default=False,
        help="regenerate the cached target-model baseline",
    )
    group.addoption(
        "--spec-model-parity",
        action="store_true",
        default=False,
        help="run bare-model versus custom-kv-cache parity tests",
    )
    group.addoption(
        "--spec-model-parity-refresh-baseline",
        action="store_true",
        default=False,
        help="regenerate the cached bare-model baseline",
    )
    group.addoption(
        "--spec-target-model",
        action="store",
        default="HuggingFaceTB/SmolLM2-360M",
        help="target/full model id for the cached baselines",
    )
    group.addoption(
        "--spec-draft-model",
        action="store",
        default="HuggingFaceTB/SmolLM2-135M",
        help="draft model id for speculative engines",
    )
    group.addoption(
        "--spec-revision",
        action="store",
        default="main",
        help="model revision to use for all engines",
    )
    group.addoption(
        "--spec-seed",
        action="store",
        type=int,
        default=42,
        help="seed used by baseline and speculative engines",
    )
    group.addoption(
        "--spec-temperature",
        action="store",
        type=float,
        default=0.0,
        help="sampling temperature for the baseline and sync engine",
    )
    group.addoption(
        "--spec-repeat-penalty",
        action="store",
        type=float,
        default=1.0,
        help="repeat penalty for the baseline and sync engine",
    )
    group.addoption(
        "--spec-repeat-last-n",
        action="store",
        type=int,
        default=64,
        help="repeat-penalty window for the baseline and sync engine",
    )
    group.addoption(
        "--spec-gamma",
        action="store",
        type=int,
        default=5,
        help="gamma value for speculative engines",
    )
