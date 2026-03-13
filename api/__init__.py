"""
RAG-Advanced API Module.

FastAPI REST API for strategy execution, evaluation, and benchmarking.

Exports:
    Authentication:
        - ApiKeyAuth: API key authentication handler
        - ApiKeyInfo: Information about an API key
        - VerificationResult: Result of key verification
        - generate_api_key: Generate new API key
        - hash_api_key: Hash an API key
        - verify_key_format: Validate key format
"""

__version__ = "0.1.0"

from api.auth import (
    ApiKeyAuth,
    ApiKeyInfo,
    VerificationResult,
    generate_api_key,
    hash_api_key,
    verify_key_format,
)

from api.rate_limiter import (
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    check_rate_limit,
)

__all__ = [
    "__version__",
    # Authentication
    "ApiKeyAuth",
    "ApiKeyInfo",
    "VerificationResult",
    "generate_api_key",
    "hash_api_key",
    "verify_key_format",
    # Rate Limiting
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimiter",
    "check_rate_limit",
]
