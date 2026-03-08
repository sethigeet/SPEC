use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

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
