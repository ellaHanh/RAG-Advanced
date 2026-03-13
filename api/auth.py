"""
RAG-Advanced API Authentication.

API key verification using SHA256 hash against database.
Supports key validation, expiration checking, and rate limit tracking.

Usage:
    from api.auth import ApiKeyAuth, generate_api_key

    # Generate a new API key
    key, key_hash = generate_api_key()

    # Verify API key
    auth = ApiKeyAuth(db_pool)
    result = await auth.verify_key(api_key)
    if result.is_valid:
        print(f"Authenticated: {result.key_id}")
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestration.errors import AuthenticationError


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

API_KEY_PREFIX = "rag_"
API_KEY_LENGTH = 32  # bytes, will be hex encoded to 64 chars


# =============================================================================
# Models
# =============================================================================


class ApiKeyInfo(BaseModel):
    """
    Information about an API key.

    Attributes:
        key_id: Unique identifier for the key.
        name: Human-readable name for the key.
        is_active: Whether the key is currently active.
        created_at: When the key was created.
        expires_at: When the key expires (None = never).
        last_used_at: When the key was last used.
        rate_limit: Requests per minute limit.
        scopes: List of permitted scopes.
    """

    model_config = ConfigDict(frozen=True)

    key_id: str = Field(..., description="Key identifier")
    name: str = Field(default="", description="Key name")
    is_active: bool = Field(default=True, description="Is key active")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    expires_at: datetime | None = Field(default=None, description="Expiration time")
    last_used_at: datetime | None = Field(default=None, description="Last used time")
    rate_limit: int = Field(default=60, ge=0, description="Rate limit (req/min)")
    scopes: list[str] = Field(default_factory=list, description="Permitted scopes")


@dataclass
class VerificationResult:
    """
    Result of API key verification.

    Attributes:
        is_valid: Whether the key is valid.
        key_info: Information about the key (if valid).
        error_message: Error message (if invalid).
        error_code: Error code for programmatic handling.
    """

    is_valid: bool
    key_info: ApiKeyInfo | None = None
    error_message: str | None = None
    error_code: str | None = None

    @property
    def key_id(self) -> str | None:
        """Get key ID if valid."""
        return self.key_info.key_id if self.key_info else None


# =============================================================================
# Key Generation
# =============================================================================


def generate_api_key(prefix: str = API_KEY_PREFIX) -> tuple[str, str]:
    """
    Generate a new API key and its hash.

    Args:
        prefix: Prefix for the API key.

    Returns:
        Tuple of (api_key, key_hash).

    Example:
        >>> key, key_hash = generate_api_key()
        >>> print(f"Key: {key}")  # rag_abc123...
        >>> # Store key_hash in database, give key to user
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(API_KEY_LENGTH)
    key_body = random_bytes.hex()

    # Create full key with prefix
    api_key = f"{prefix}{key_body}"

    # Hash the key
    key_hash = hash_api_key(api_key)

    return api_key, key_hash


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using SHA256.

    Args:
        api_key: The raw API key.

    Returns:
        SHA256 hash of the key.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_key_format(api_key: str, prefix: str = API_KEY_PREFIX) -> bool:
    """
    Verify API key format is valid.

    Args:
        api_key: The API key to verify.
        prefix: Expected prefix.

    Returns:
        True if format is valid.
    """
    if not api_key or not isinstance(api_key, str):
        return False

    if not api_key.startswith(prefix):
        return False

    # Check length (prefix + 64 hex chars)
    expected_length = len(prefix) + (API_KEY_LENGTH * 2)
    if len(api_key) != expected_length:
        return False

    # Check if body is valid hex
    key_body = api_key[len(prefix):]
    try:
        int(key_body, 16)
        return True
    except ValueError:
        return False


# =============================================================================
# API Key Authentication
# =============================================================================


class ApiKeyAuth:
    """
    API key authentication handler.

    Verifies API keys against hashed values stored in database.
    Handles expiration, rate limiting, and scope validation.

    Example:
        >>> auth = ApiKeyAuth(db_pool)
        >>> result = await auth.verify_key("rag_abc123...")
        >>> if result.is_valid:
        ...     print(f"Welcome {result.key_info.name}")
    """

    def __init__(
        self,
        db_pool: Any | None = None,
        cache_ttl_seconds: int = 60,
    ) -> None:
        """
        Initialize the authentication handler.

        Args:
            db_pool: Database connection pool (asyncpg).
            cache_ttl_seconds: How long to cache key info.
        """
        self._db_pool = db_pool
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[ApiKeyInfo, datetime]] = {}

    async def verify_key(
        self,
        api_key: str,
        required_scopes: list[str] | None = None,
    ) -> VerificationResult:
        """
        Verify an API key.

        Args:
            api_key: The API key to verify.
            required_scopes: Optional list of required scopes.

        Returns:
            VerificationResult with validation status.
        """
        # Validate format first
        if not verify_key_format(api_key):
            return VerificationResult(
                is_valid=False,
                error_message="Invalid API key format",
                error_code="INVALID_FORMAT",
            )

        # Hash the key
        key_hash = hash_api_key(api_key)

        # Check cache first
        cached = self._get_from_cache(key_hash)
        if cached:
            return self._validate_key_info(cached, required_scopes)

        # Look up in database
        key_info = await self._lookup_key(key_hash)
        if key_info is None:
            return VerificationResult(
                is_valid=False,
                error_message="Invalid API key",
                error_code="INVALID_KEY",
            )

        # Cache the result
        self._add_to_cache(key_hash, key_info)

        # Validate and return
        result = self._validate_key_info(key_info, required_scopes)

        # Update last used time if valid
        if result.is_valid and self._db_pool:
            await self._update_last_used(key_info.key_id)

        return result

    async def create_key(
        self,
        name: str,
        scopes: list[str] | None = None,
        rate_limit: int = 60,
        expires_in_days: int | None = None,
    ) -> tuple[str, ApiKeyInfo]:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key.
            scopes: List of permitted scopes.
            rate_limit: Requests per minute limit.
            expires_in_days: Days until expiration (None = never).

        Returns:
            Tuple of (raw_api_key, key_info).
        """
        api_key, key_hash = generate_api_key()

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        key_info = ApiKeyInfo(
            key_id=key_hash[:16],  # Use first 16 chars of hash as ID
            name=name,
            is_active=True,
            expires_at=expires_at,
            rate_limit=rate_limit,
            scopes=scopes or [],
        )

        if self._db_pool:
            await self._store_key(key_hash, key_info)

        return api_key, key_info

    async def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: The key ID to revoke.

        Returns:
            True if key was revoked.
        """
        if not self._db_pool:
            return False

        try:
            async with self._db_pool.acquire() as conn:
                result = await conn.execute(
                    "UPDATE api_keys SET is_active = false WHERE key_id = $1",
                    key_id,
                )
                # Clear cache
                self._clear_cache_for_key_id(key_id)
                return result == "UPDATE 1"
        except Exception as e:
            logger.exception(f"Failed to revoke key: {e}")
            return False

    def verify_key_sync(
        self,
        api_key: str,
        key_store: dict[str, ApiKeyInfo] | None = None,
        required_scopes: list[str] | None = None,
    ) -> VerificationResult:
        """
        Synchronous key verification (for testing).

        Args:
            api_key: The API key to verify.
            key_store: Dictionary of key_hash -> ApiKeyInfo.
            required_scopes: Optional required scopes.

        Returns:
            VerificationResult.
        """
        if not verify_key_format(api_key):
            return VerificationResult(
                is_valid=False,
                error_message="Invalid API key format",
                error_code="INVALID_FORMAT",
            )

        key_hash = hash_api_key(api_key)

        if not key_store:
            return VerificationResult(
                is_valid=False,
                error_message="No key store provided",
                error_code="NO_STORE",
            )

        key_info = key_store.get(key_hash)
        if key_info is None:
            return VerificationResult(
                is_valid=False,
                error_message="Invalid API key",
                error_code="INVALID_KEY",
            )

        return self._validate_key_info(key_info, required_scopes)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _validate_key_info(
        self,
        key_info: ApiKeyInfo,
        required_scopes: list[str] | None,
    ) -> VerificationResult:
        """Validate key info against requirements."""
        # Check if active
        if not key_info.is_active:
            return VerificationResult(
                is_valid=False,
                key_info=key_info,
                error_message="API key has been revoked",
                error_code="KEY_REVOKED",
            )

        # Check expiration
        if key_info.expires_at and key_info.expires_at < datetime.now(UTC):
            return VerificationResult(
                is_valid=False,
                key_info=key_info,
                error_message="API key has expired",
                error_code="KEY_EXPIRED",
            )

        # Check scopes
        if required_scopes:
            missing_scopes = set(required_scopes) - set(key_info.scopes)
            if missing_scopes:
                return VerificationResult(
                    is_valid=False,
                    key_info=key_info,
                    error_message=f"Missing required scopes: {missing_scopes}",
                    error_code="INSUFFICIENT_SCOPES",
                )

        return VerificationResult(
            is_valid=True,
            key_info=key_info,
        )

    def _get_from_cache(self, key_hash: str) -> ApiKeyInfo | None:
        """Get key info from cache if not expired."""
        if key_hash not in self._cache:
            return None

        key_info, cached_at = self._cache[key_hash]
        if datetime.now(UTC) - cached_at > timedelta(seconds=self._cache_ttl):
            del self._cache[key_hash]
            return None

        return key_info

    def _add_to_cache(self, key_hash: str, key_info: ApiKeyInfo) -> None:
        """Add key info to cache."""
        self._cache[key_hash] = (key_info, datetime.now(UTC))

    def _clear_cache_for_key_id(self, key_id: str) -> None:
        """Clear cache entries for a key ID."""
        to_remove = [
            kh for kh, (info, _) in self._cache.items()
            if info.key_id == key_id
        ]
        for kh in to_remove:
            del self._cache[kh]

    async def _lookup_key(self, key_hash: str) -> ApiKeyInfo | None:
        """Look up key in database."""
        if not self._db_pool:
            return None

        try:
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT key_id, name, is_active, created_at, expires_at,
                           last_used_at, rate_limit, scopes
                    FROM api_keys
                    WHERE key_hash = $1
                    """,
                    key_hash,
                )

                if row is None:
                    return None

                return ApiKeyInfo(
                    key_id=row["key_id"],
                    name=row["name"],
                    is_active=row["is_active"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    last_used_at=row["last_used_at"],
                    rate_limit=row["rate_limit"],
                    scopes=row["scopes"] or [],
                )

        except Exception as e:
            logger.exception(f"Database lookup failed: {e}")
            return None

    async def _store_key(self, key_hash: str, key_info: ApiKeyInfo) -> None:
        """Store key in database."""
        if not self._db_pool:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO api_keys (key_hash, key_id, name, is_active,
                                         created_at, expires_at, rate_limit, scopes)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    key_hash,
                    key_info.key_id,
                    key_info.name,
                    key_info.is_active,
                    key_info.created_at,
                    key_info.expires_at,
                    key_info.rate_limit,
                    key_info.scopes,
                )
        except Exception as e:
            logger.exception(f"Failed to store key: {e}")
            raise

    async def _update_last_used(self, key_id: str) -> None:
        """Update last used timestamp."""
        if not self._db_pool:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE api_keys SET last_used_at = $1 WHERE key_id = $2",
                    datetime.now(UTC),
                    key_id,
                )
        except Exception as e:
            logger.warning(f"Failed to update last_used_at: {e}")

    async def verify_and_update_atomic(
        self,
        api_key: str,
        required_scopes: list[str] | None = None,
    ) -> VerificationResult:
        """
        Verify API key and update usage atomically.

        Uses a single UPDATE...RETURNING query to atomically verify and update
        the key, preventing race conditions in concurrent access.

        Args:
            api_key: The API key to verify.
            required_scopes: Optional list of required scopes.

        Returns:
            VerificationResult with validation status.
        """
        # Validate format first
        if not verify_key_format(api_key):
            return VerificationResult(
                is_valid=False,
                error_message="Invalid API key format",
                error_code="INVALID_FORMAT",
            )

        key_hash = hash_api_key(api_key)

        if not self._db_pool:
            return VerificationResult(
                is_valid=False,
                error_message="Database not configured",
                error_code="NO_DATABASE",
            )

        try:
            async with self._db_pool.acquire() as conn:
                # Atomic update and return in single query
                row = await conn.fetchrow(
                    """
                    UPDATE api_keys
                    SET last_used_at = $1,
                        usage_count = COALESCE(usage_count, 0) + 1
                    WHERE key_hash = $2
                    RETURNING key_id, name, is_active, created_at, expires_at,
                              last_used_at, rate_limit, scopes, usage_count
                    """,
                    datetime.now(UTC),
                    key_hash,
                )

                if row is None:
                    return VerificationResult(
                        is_valid=False,
                        error_message="Invalid API key",
                        error_code="INVALID_KEY",
                    )

                key_info = ApiKeyInfo(
                    key_id=row["key_id"],
                    name=row["name"],
                    is_active=row["is_active"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    last_used_at=row["last_used_at"],
                    rate_limit=row["rate_limit"],
                    scopes=row["scopes"] or [],
                )

                # Validate and return
                return self._validate_key_info(key_info, required_scopes)

        except Exception as e:
            logger.exception(f"Atomic verification failed: {e}")
            return VerificationResult(
                is_valid=False,
                error_message="Verification error",
                error_code="VERIFICATION_ERROR",
            )
