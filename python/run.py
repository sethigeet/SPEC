from SPEC.SPEC.engines import SpecDecodingEngine

engine = SpecDecodingEngine(
    draft_model_id="HuggingFaceTB/SmolLM2-135M",
    target_model_id="HuggingFaceTB/SmolLM2-360M",
    gamma=5,  # draft tokens per verification batch
    seed=42,
)

output = engine.generate("The meaning of life is", max_tokens=100)
print(output)
