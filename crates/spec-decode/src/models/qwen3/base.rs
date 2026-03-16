use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3 as qwen3_model;

use crate::models::utils::{load_model_files, parse_eos_token_ids};

pub struct BaseQwen3 {
    model: qwen3_model::ModelForCausalLM,
    cfg: BaseQwen3Config,
    pos: usize,
}

#[derive(Clone)]
pub struct BaseQwen3Config {
    config: qwen3_model::Config,
    eos_token_ids: Vec<u32>,
    device: Device,
    dtype: DType,
}

impl BaseQwen3 {
    pub fn from_hub(model_id: &str, revision: &str, device: &Device, dtype: DType) -> Result<Self> {
        let files = load_model_files(model_id, revision)?;
        let config: qwen3_model::Config = serde_json::from_slice(&files.raw_config)?;
        let eos_token_ids = parse_eos_token_ids(&files.raw_config)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.filenames, dtype, device)? };
        let model = qwen3_model::ModelForCausalLM::new(&config, vb)?;

        Ok(Self {
            model,
            cfg: BaseQwen3Config {
                config,
                eos_token_ids,
                device: device.clone(),
                dtype,
            },
            pos: 0,
        })
    }

    pub fn forward(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        let input = Tensor::new(token_ids, &self.cfg.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, self.pos)?;
        self.pos += token_ids.len();
        Ok(logits.squeeze(0)?)
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        self.pos = 0;
        let _ = (&self.cfg.config, self.cfg.dtype);
        Ok(())
    }

    pub fn is_eos(&self, token: u32) -> bool {
        self.cfg.eos_token_ids.contains(&token)
    }
}
