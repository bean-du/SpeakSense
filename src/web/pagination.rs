use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Pagination {
    pub index: u64,
    pub size: u64,
}

impl Default for Pagination {
    fn default() -> Self {
        Self { index: 1, size: 10 }
    }
}

impl Pagination {
    pub fn offset(&self) -> u64 {
        (self.index - 1) * self.size
    }

    pub fn limit(&self) -> u64 {
        self.size
    }

    pub fn check(&self) -> Self {
        if self.index < 1 || self.size < 1 {
            return Self::default();
        }
        self.clone()
    }
}
