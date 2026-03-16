use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::models::utils::{detect_model_family, load_model_files, ModelFamily};
use crate::models::{
    BaseLlama, BaseQwen3, PagedLlama, PagedQwen3,
};

pub enum BaseModel {
    Llama(BaseLlama),
    Qwen3(BaseQwen3),
}

impl BaseModel {
    pub fn from_hub(model_id: &str, revision: &str, device: &Device, dtype: DType) -> Result<Self> {
        let files = load_model_files(model_id, revision)?;
        match detect_model_family(&files.raw_config)? {
            ModelFamily::Llama => BaseLlama::from_hub(model_id, revision, device, dtype).map(Self::Llama),
            ModelFamily::Qwen3Dense => {
                BaseQwen3::from_hub(model_id, revision, device, dtype).map(Self::Qwen3)
            }
        }
    }

    pub fn forward(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        match self {
            Self::Llama(model) => model.forward(token_ids),
            Self::Qwen3(model) => model.forward(token_ids),
        }
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        match self {
            Self::Llama(model) => model.reset_cache(),
            Self::Qwen3(model) => model.reset_cache(),
        }
    }

    pub fn is_eos(&self, token: u32) -> bool {
        match self {
            Self::Llama(model) => model.is_eos(token),
            Self::Qwen3(model) => model.is_eos(token),
        }
    }
}

pub enum PagedModel {
    Llama(PagedLlama),
    Qwen3(PagedQwen3),
}

impl PagedModel {
    pub fn from_hub(model_id: &str, revision: &str, device: &Device, dtype: DType) -> Result<Self> {
        let files = load_model_files(model_id, revision)?;
        match detect_model_family(&files.raw_config)? {
            ModelFamily::Llama => PagedLlama::from_hub(model_id, revision, device, dtype).map(Self::Llama),
            ModelFamily::Qwen3Dense => {
                PagedQwen3::from_hub(model_id, revision, device, dtype).map(Self::Qwen3)
            }
        }
    }

    pub fn forward(&mut self, token_ids: &[u32], epoch: usize) -> Result<Tensor> {
        match self {
            Self::Llama(model) => model.forward(token_ids, epoch),
            Self::Qwen3(model) => model.forward(token_ids, epoch),
        }
    }

    pub fn reset_cache(&mut self) {
        match self {
            Self::Llama(model) => model.reset_cache(),
            Self::Qwen3(model) => model.reset_cache(),
        }
    }

    pub fn truncate_cache_to(&mut self, new_len: usize) {
        match self {
            Self::Llama(model) => model.truncate_cache_to(new_len),
            Self::Qwen3(model) => model.truncate_cache_to(new_len),
        }
    }

    pub fn rollback_cache(&mut self, dead_epoch: usize) {
        match self {
            Self::Llama(model) => model.rollback_cache(dead_epoch),
            Self::Qwen3(model) => model.rollback_cache(dead_epoch),
        }
    }

    pub fn is_eos(&self, token: u32) -> bool {
        match self {
            Self::Llama(model) => model.is_eos(token),
            Self::Qwen3(model) => model.is_eos(token),
        }
    }

    pub fn device(&self) -> &Device {
        match self {
            Self::Llama(model) => &model.cfg.device,
            Self::Qwen3(model) => &model.cfg.device,
        }
    }
}
