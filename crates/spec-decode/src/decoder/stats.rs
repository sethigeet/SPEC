/// Statistics from a speculative decoding run.
#[derive(Debug, Clone)]
pub struct Stats {
    /// Total number of draft tokens proposed across all steps.
    pub draft_proposed: usize,
    /// Number of draft tokens accepted by the target model.
    pub draft_accepted: usize,
    /// Total number of speculative decoding steps.
    pub num_steps: usize,
    /// Total tokens generated (including prompt).
    pub total_tokens: usize,
}

impl Stats {
    /// Create empty stats.
    pub fn new() -> Self {
        Self {
            draft_proposed: 0,
            draft_accepted: 0,
            num_steps: 0,
            total_tokens: 0,
        }
    }

    /// Draft token acceptance rate as a fraction in [0.0, 1.0].
    ///
    /// Returns 0.0 if no tokens were proposed.
    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_proposed == 0 {
            0.0
        } else {
            self.draft_accepted as f64 / self.draft_proposed as f64
        }
    }

    /// Average number of draft tokens accepted per step.
    pub fn avg_accepted_per_step(&self) -> f64 {
        if self.num_steps == 0 {
            0.0
        } else {
            self.draft_accepted as f64 / self.num_steps as f64
        }
    }
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}
