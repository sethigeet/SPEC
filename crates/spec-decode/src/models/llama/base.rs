use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as llama_model;
use hf_hub::{api::sync::Api, Repo, RepoType};

pub struct BaseLlama {
    model: llama_model::Llama,
    cache: llama_model::Cache,
    cfg: BaseLlamaConfig,
    pos: usize,
}

#[derive(Clone)]
pub struct BaseLlamaConfig {
    config: llama_model::Config,
    eos_token_id: Option<llama_model::LlamaEosToks>,
    device: Device,
    dtype: DType,
}

impl BaseLlama {
    pub fn from_hub(model_id: &str, revision: &str, device: &Device, dtype: DType) -> Result<Self> {
        let api = Api::new().context("failed to create HF Hub API")?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let config_path = repo.get("config.json").context("config.json not found")?;
        let raw = std::fs::read(&config_path)?;
        let llama_config: llama_model::LlamaConfig = serde_json::from_slice(&raw)?;
        let eos_token_id = llama_config.eos_token_id.clone();
        let config = llama_config.into_config(cfg!(feature = "flash-attn"));

        let filenames = {
            let single = repo.get("model.safetensors");
            match single {
                Ok(path) => vec![path],
                Err(_) => {
                    let index_path = repo
                        .get("model.safetensors.index.json")
                        .context("neither model.safetensors nor index found")?;
                    let index_raw = std::fs::read(&index_path)?;
                    let index: serde_json::Value = serde_json::from_slice(&index_raw)?;
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

        let cache = llama_model::Cache::new(true, dtype, &config, device)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
        let model = llama_model::Llama::load(vb, &config)?;

        Ok(Self {
            model,
            cache,
            cfg: BaseLlamaConfig {
                config,
                eos_token_id,
                device: device.clone(),
                dtype,
            },
            pos: 0,
        })
    }

    pub fn forward(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        let input = Tensor::new(token_ids, &self.cfg.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, self.pos, &mut self.cache)?;
        self.pos += token_ids.len();
        Ok(logits.squeeze(0)?)
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        self.cache =
            llama_model::Cache::new(true, self.cfg.dtype, &self.cfg.config, &self.cfg.device)?;
        self.pos = 0;
        Ok(())
    }

    pub fn is_eos(&self, token: u32) -> bool {
        match &self.cfg.eos_token_id {
            Some(llama_model::LlamaEosToks::Single(id)) => token == *id,
            Some(llama_model::LlamaEosToks::Multiple(ids)) => ids.contains(&token),
            None => false,
        }
    }
}
