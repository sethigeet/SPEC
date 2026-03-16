use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Llama,
    Qwen3Dense,
}

pub struct HubModelFiles {
    pub raw_config: Vec<u8>,
    pub filenames: Vec<PathBuf>,
}

pub fn load_model_files(model_id: &str, revision: &str) -> Result<HubModelFiles> {
    let api = Api::new().context("failed to create HF Hub API")?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let config_path = repo.get("config.json").context("config.json not found")?;
    let raw_config = std::fs::read(&config_path)?;
    let filenames = {
        let single = repo.get("model.safetensors");
        match single {
            Ok(path) => vec![path],
            Err(_) => {
                let index_path = repo
                    .get("model.safetensors.index.json")
                    .context("neither model.safetensors nor index found")?;
                let index_raw = std::fs::read(&index_path)?;
                let index: Value = serde_json::from_slice(&index_raw)?;
                let weight_map = index["weight_map"]
                    .as_object()
                    .context("invalid index format: no weight_map")?;
                let mut files: Vec<String> = weight_map
                    .values()
                    .filter_map(|value| value.as_str().map(ToOwned::to_owned))
                    .collect();
                files.sort();
                files.dedup();
                files
                    .into_iter()
                    .map(|file| {
                        repo.get(&file)
                            .with_context(|| format!("failed to get {file}"))
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        }
    };

    Ok(HubModelFiles {
        raw_config,
        filenames,
    })
}

pub fn detect_model_family(raw_config: &[u8]) -> Result<ModelFamily> {
    let value: Value = serde_json::from_slice(raw_config).context("invalid model config json")?;
    let model_type = value
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let architectures: Vec<&str> = value
        .get("architectures")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();

    let is_qwen3 = model_type == "qwen3" || architectures.iter().any(|arch| arch.starts_with("Qwen3"));
    let is_moe = model_type.contains("moe")
        || architectures
            .iter()
            .any(|arch| arch.contains("Moe") || arch.contains("MoE"));

    if is_qwen3 {
        if is_moe {
            bail!("Qwen3 MoE models are not supported yet; only dense Qwen 3.5 models are supported");
        }
        return Ok(ModelFamily::Qwen3Dense);
    }

    Ok(ModelFamily::Llama)
}

pub fn parse_eos_token_ids(raw_config: &[u8]) -> Result<Vec<u32>> {
    let value: Value = serde_json::from_slice(raw_config).context("invalid model config json")?;
    let eos = match value.get("eos_token_id") {
        Some(Value::Number(id)) => vec![id
            .as_u64()
            .context("eos_token_id must be a non-negative integer")? as u32],
        Some(Value::Array(ids)) => ids
            .iter()
            .map(|id| {
                id.as_u64()
                    .context("eos_token_id array must contain non-negative integers")
                    .map(|id| id as u32)
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => bail!("unsupported eos_token_id format"),
        None => Vec::new(),
    };
    Ok(eos)
}

pub fn load_tokenizer(model_id: &str, revision: &str) -> Result<Tokenizer> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("tokenizer.json not found")?;
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
}
