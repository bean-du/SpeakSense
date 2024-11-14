use std::fmt;

#[derive(Debug)]
pub enum AuthError {
    MissingApiKey,
    InvalidApiKey,
    KeyExpired,
    KeySuspended,
    InsufficientPermissions,
    RateLimitExceeded,
    StorageError(String),
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthError::MissingApiKey => write!(f, "Missing API key"),
            AuthError::InvalidApiKey => write!(f, "Invalid API key"),
            AuthError::KeyExpired => write!(f, "API key has expired"),
            AuthError::KeySuspended => write!(f, "API key is suspended"),
            AuthError::InsufficientPermissions => write!(f, "Insufficient permissions"),
            AuthError::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            AuthError::StorageError(e) => write!(f, "Storage error: {}", e),
        }
    }
}

impl std::error::Error for AuthError {}

impl From<anyhow::Error> for AuthError {
    fn from(err: anyhow::Error) -> Self {
        AuthError::StorageError(err.to_string())
    }
} 