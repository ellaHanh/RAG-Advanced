"""
Unit tests for API Key Authentication.

Tests cover:
- Key generation
- Key hashing
- Format validation
- Key verification (valid, invalid, expired, revoked)
- Scope validation
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from api.auth import (
    API_KEY_LENGTH,
    API_KEY_PREFIX,
    ApiKeyAuth,
    ApiKeyInfo,
    VerificationResult,
    generate_api_key,
    hash_api_key,
    verify_key_format,
)


# =============================================================================
# Test: Key Generation
# =============================================================================


class TestKeyGeneration:
    """Tests for API key generation."""

    def test_generate_api_key_format(self):
        """Test generated key has correct format."""
        key, key_hash = generate_api_key()

        assert key.startswith(API_KEY_PREFIX)
        assert len(key) == len(API_KEY_PREFIX) + (API_KEY_LENGTH * 2)

    def test_generate_api_key_unique(self):
        """Test generated keys are unique."""
        keys = [generate_api_key()[0] for _ in range(100)]
        assert len(set(keys)) == 100

    def test_generate_api_key_hash_is_different(self):
        """Test hash is different from key."""
        key, key_hash = generate_api_key()

        assert key != key_hash
        assert len(key_hash) == 64  # SHA256 hex digest

    def test_generate_api_key_custom_prefix(self):
        """Test custom prefix."""
        key, _ = generate_api_key(prefix="test_")

        assert key.startswith("test_")

    def test_hash_api_key_deterministic(self):
        """Test hashing is deterministic."""
        key, _ = generate_api_key()

        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)

        assert hash1 == hash2

    def test_hash_api_key_different_for_different_keys(self):
        """Test different keys have different hashes."""
        key1, _ = generate_api_key()
        key2, _ = generate_api_key()

        hash1 = hash_api_key(key1)
        hash2 = hash_api_key(key2)

        assert hash1 != hash2


# =============================================================================
# Test: Format Validation
# =============================================================================


class TestFormatValidation:
    """Tests for API key format validation."""

    def test_valid_format(self):
        """Test valid key format passes."""
        key, _ = generate_api_key()
        assert verify_key_format(key) is True

    def test_invalid_prefix(self):
        """Test invalid prefix fails."""
        assert verify_key_format("wrong_" + "a" * 64) is False

    def test_too_short(self):
        """Test too short key fails."""
        assert verify_key_format("rag_" + "a" * 10) is False

    def test_too_long(self):
        """Test too long key fails."""
        assert verify_key_format("rag_" + "a" * 100) is False

    def test_invalid_hex(self):
        """Test non-hex characters fail."""
        # 'z' is not valid hex
        assert verify_key_format("rag_" + "z" * 64) is False

    def test_empty_key(self):
        """Test empty key fails."""
        assert verify_key_format("") is False

    def test_none_key(self):
        """Test None fails."""
        assert verify_key_format(None) is False  # type: ignore

    def test_custom_prefix_validation(self):
        """Test validation with custom prefix."""
        key, _ = generate_api_key(prefix="custom_")
        assert verify_key_format(key, prefix="custom_") is True
        assert verify_key_format(key, prefix="rag_") is False


# =============================================================================
# Test: ApiKeyInfo Model
# =============================================================================


class TestApiKeyInfo:
    """Tests for ApiKeyInfo model."""

    def test_basic_creation(self):
        """Test basic info creation."""
        info = ApiKeyInfo(key_id="test123")

        assert info.key_id == "test123"
        assert info.is_active is True
        assert info.rate_limit == 60

    def test_with_all_fields(self):
        """Test info with all fields."""
        expires = datetime.now(UTC) + timedelta(days=30)
        info = ApiKeyInfo(
            key_id="test123",
            name="Test Key",
            is_active=True,
            expires_at=expires,
            rate_limit=100,
            scopes=["read", "write"],
        )

        assert info.name == "Test Key"
        assert info.expires_at == expires
        assert info.scopes == ["read", "write"]


# =============================================================================
# Test: VerificationResult
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_valid_result(self):
        """Test valid verification result."""
        info = ApiKeyInfo(key_id="test123")
        result = VerificationResult(is_valid=True, key_info=info)

        assert result.is_valid is True
        assert result.key_id == "test123"

    def test_invalid_result(self):
        """Test invalid verification result."""
        result = VerificationResult(
            is_valid=False,
            error_message="Invalid key",
            error_code="INVALID_KEY",
        )

        assert result.is_valid is False
        assert result.error_code == "INVALID_KEY"
        assert result.key_id is None


# =============================================================================
# Test: ApiKeyAuth - Synchronous Verification
# =============================================================================


class TestApiKeyAuthSync:
    """Tests for synchronous key verification."""

    @pytest.fixture
    def auth(self) -> ApiKeyAuth:
        """Create auth instance."""
        return ApiKeyAuth()

    @pytest.fixture
    def key_store(self) -> tuple[str, dict[str, ApiKeyInfo]]:
        """Create a key and key store."""
        key, key_hash = generate_api_key()
        info = ApiKeyInfo(
            key_id="test123",
            name="Test Key",
            scopes=["read", "write"],
        )
        return key, {key_hash: info}

    def test_verify_valid_key(self, auth: ApiKeyAuth, key_store: tuple[str, dict]):
        """Test verification of valid key."""
        key, store = key_store
        result = auth.verify_key_sync(key, store)

        assert result.is_valid is True
        assert result.key_info.key_id == "test123"

    def test_verify_invalid_format(self, auth: ApiKeyAuth, key_store: tuple[str, dict]):
        """Test verification of invalid format."""
        _, store = key_store
        result = auth.verify_key_sync("invalid_key", store)

        assert result.is_valid is False
        assert result.error_code == "INVALID_FORMAT"

    def test_verify_unknown_key(self, auth: ApiKeyAuth, key_store: tuple[str, dict]):
        """Test verification of unknown key."""
        _, store = key_store
        other_key, _ = generate_api_key()  # Different key
        result = auth.verify_key_sync(other_key, store)

        assert result.is_valid is False
        assert result.error_code == "INVALID_KEY"

    def test_verify_expired_key(self, auth: ApiKeyAuth):
        """Test verification of expired key."""
        key, key_hash = generate_api_key()
        info = ApiKeyInfo(
            key_id="expired",
            expires_at=datetime.now(UTC) - timedelta(days=1),  # Expired yesterday
        )
        store = {key_hash: info}

        result = auth.verify_key_sync(key, store)

        assert result.is_valid is False
        assert result.error_code == "KEY_EXPIRED"

    def test_verify_revoked_key(self, auth: ApiKeyAuth):
        """Test verification of revoked key."""
        key, key_hash = generate_api_key()
        info = ApiKeyInfo(
            key_id="revoked",
            is_active=False,
        )
        store = {key_hash: info}

        result = auth.verify_key_sync(key, store)

        assert result.is_valid is False
        assert result.error_code == "KEY_REVOKED"

    def test_verify_with_required_scopes(self, auth: ApiKeyAuth, key_store: tuple[str, dict]):
        """Test verification with required scopes."""
        key, store = key_store
        result = auth.verify_key_sync(key, store, required_scopes=["read"])

        assert result.is_valid is True

    def test_verify_missing_scopes(self, auth: ApiKeyAuth, key_store: tuple[str, dict]):
        """Test verification with missing scopes."""
        key, store = key_store
        result = auth.verify_key_sync(key, store, required_scopes=["admin"])

        assert result.is_valid is False
        assert result.error_code == "INSUFFICIENT_SCOPES"


# =============================================================================
# Test: ApiKeyAuth - Async Verification
# =============================================================================


class TestApiKeyAuthAsync:
    """Tests for async key verification."""

    @pytest.mark.asyncio
    async def test_verify_invalid_format(self):
        """Test async verification of invalid format."""
        auth = ApiKeyAuth()
        result = await auth.verify_key("invalid_key")

        assert result.is_valid is False
        assert result.error_code == "INVALID_FORMAT"

    @pytest.mark.asyncio
    async def test_verify_no_db_pool(self):
        """Test async verification without database."""
        auth = ApiKeyAuth()
        key, _ = generate_api_key()
        result = await auth.verify_key(key)

        # Without DB pool, key lookup returns None
        assert result.is_valid is False
        assert result.error_code == "INVALID_KEY"


# =============================================================================
# Test: Caching
# =============================================================================


class TestCaching:
    """Tests for key info caching."""

    def test_cache_hit(self):
        """Test cache is used for repeated lookups."""
        auth = ApiKeyAuth(cache_ttl_seconds=60)
        key, key_hash = generate_api_key()
        info = ApiKeyInfo(key_id="cached")

        # Manually add to cache
        auth._cache[key_hash] = (info, datetime.now(UTC))

        # Should get from cache
        cached = auth._get_from_cache(key_hash)
        assert cached is not None
        assert cached.key_id == "cached"

    def test_cache_expiry(self):
        """Test expired cache entries are not returned."""
        auth = ApiKeyAuth(cache_ttl_seconds=1)
        key, key_hash = generate_api_key()
        info = ApiKeyInfo(key_id="expired_cache")

        # Add to cache with old timestamp
        auth._cache[key_hash] = (info, datetime.now(UTC) - timedelta(seconds=10))

        # Should not get from expired cache
        cached = auth._get_from_cache(key_hash)
        assert cached is None


# =============================================================================
# Test: Atomic Verification
# =============================================================================


class TestAtomicVerification:
    """Tests for atomic key verification and update."""

    @pytest.mark.asyncio
    async def test_atomic_verify_invalid_format(self):
        """Test atomic verification rejects invalid format."""
        auth = ApiKeyAuth()
        result = await auth.verify_and_update_atomic("invalid_key")

        assert result.is_valid is False
        assert result.error_code == "INVALID_FORMAT"

    @pytest.mark.asyncio
    async def test_atomic_verify_no_database(self):
        """Test atomic verification requires database."""
        auth = ApiKeyAuth()
        key, _ = generate_api_key()
        result = await auth.verify_and_update_atomic(key)

        assert result.is_valid is False
        assert result.error_code == "NO_DATABASE"

    @pytest.mark.asyncio
    async def test_atomic_verify_validates_scopes(self):
        """Test atomic verification validates scopes (format check only)."""
        auth = ApiKeyAuth()
        # Invalid format will be caught before scope check
        result = await auth.verify_and_update_atomic(
            "bad_key",
            required_scopes=["admin"],
        )

        assert result.is_valid is False
        assert result.error_code == "INVALID_FORMAT"
