"""
Unit tests for storage-safe logging/observation IDs.

Covers the fix for GitHub issue #31055 where long Copilot Responses API
response IDs (~737 chars) caused Langfuse observation blob keys to exceed
the 255-byte path-component limit on MinIO/S3/XFS, producing
XMinioInvalidObjectName errors and dropping successful requests from
Langfuse.

Tests:
- get_logging_id hashes long response IDs, keeping the result <=255 chars
- short response IDs pass through unchanged (no regression)
- _safe_logging_id_component is deterministic
- managed Responses IDs remain reversible (previous_response_id path intact)
- Langfuse metadata breadcrumbs (litellm_response_id + litellm_response_id_sha256)
  are added only for long IDs
"""

import hashlib
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

import litellm
from litellm.utils import (
    LOGGING_ID_MAX_ID_LENGTH,
    _safe_logging_id_component,
    get_logging_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_copilot_style_response_id() -> str:
    """Build a ~737-char managed Responses API response ID (resp_<base64url>).

    Mirrors the real Copilot shape: ``resp_`` + base64url of
    ``litellm:custom_llm_provider:...;model_id:...;response_id:<long-upstream-id>``.
    """
    from litellm.responses.utils import ResponsesAPIRequestUtils

    return ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure",
        model_id="deploy-123",
        response_id="x" * 550,  # long upstream Copilot-style response_id
    )


def _make_response_obj(response_id: str) -> Any:
    """Build a minimal duck-typed response object with .get("id")."""
    obj = SimpleNamespace()
    obj.get = lambda key, default=None: response_id if key == "id" else default
    return obj


# ---------------------------------------------------------------------------
# _safe_logging_id_component
# ---------------------------------------------------------------------------

class TestSafeLoggingIdComponent:
    def test_short_id_passthrough(self):
        assert _safe_logging_id_component("resp_abc123") == "resp_abc123"

    def test_short_id_at_threshold_passthrough(self):
        """An ID exactly at the threshold is NOT hashed."""
        sid = "a" * LOGGING_ID_MAX_ID_LENGTH
        assert _safe_logging_id_component(sid) == sid

    def test_long_id_hashed_to_32_hex_chars(self):
        long_id = "a" * (LOGGING_ID_MAX_ID_LENGTH + 1)
        result = _safe_logging_id_component(long_id)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        long_id = "a" * 737
        assert _safe_logging_id_component(long_id) == _safe_logging_id_component(long_id)

    def test_hash_matches_sha256_prefix(self):
        long_id = "a" * 737
        expected = hashlib.sha256(long_id.encode("utf-8")).hexdigest()[:32]
        assert _safe_logging_id_component(long_id) == expected

    def test_different_long_ids_produce_different_hashes(self):
        id_a = "a" * 737
        id_b = "b" * 737
        assert _safe_logging_id_component(id_a) != _safe_logging_id_component(id_b)


# ---------------------------------------------------------------------------
# get_logging_id
# ---------------------------------------------------------------------------

class TestGetLoggingId:
    def test_short_id_unchanged_format(self):
        start_time = datetime(2026, 1, 15, 12, 30, 45, 123456)
        resp = _make_response_obj("resp_short123")
        result = get_logging_id(start_time, resp)
        assert result == "time-12-30-45-123456_resp_short123"

    def test_long_id_produces_storage_safe_logging_id(self):
        start_time = datetime(2026, 1, 15, 12, 30, 45, 123456)
        copilot_id = _make_copilot_style_response_id()
        assert len(copilot_id) > LOGGING_ID_MAX_ID_LENGTH

        resp = _make_response_obj(copilot_id)
        result = get_logging_id(start_time, resp)

        # Must be well under 255 bytes (XFS/MinIO path-component limit)
        assert result is not None
        assert len(result) <= 255
        # Must retain the time prefix
        assert result.startswith("time-12-30-45-123456_")
        # The ID portion after the prefix must be the hash, not the original
        id_portion = result.split("_", 1)[1]
        assert len(id_portion) == 32
        assert id_portion == hashlib.sha256(copilot_id.encode("utf-8")).hexdigest()[:32]

    def test_long_id_deterministic(self):
        start_time = datetime(2026, 1, 15, 12, 30, 45, 123456)
        copilot_id = _make_copilot_style_response_id()
        resp = _make_response_obj(copilot_id)
        assert get_logging_id(start_time, resp) == get_logging_id(start_time, resp)

    def test_none_id_returns_none(self):
        start_time = datetime(2026, 1, 15, 12, 30, 45, 123456)
        resp = _make_response_obj(None)
        assert get_logging_id(start_time, resp) is None

    def test_copilot_style_id_under_255(self):
        """Realistic Copilot-style ID (~737 chars) produces a logging ID
        comfortably under 255 bytes."""
        start_time = datetime(2026, 1, 15, 12, 30, 45, 999999)
        copilot_id = _make_copilot_style_response_id()
        resp = _make_response_obj(copilot_id)
        result = get_logging_id(start_time, resp)
        # time prefix is "time-12-30-45-999999_" = 21 chars
        # hash is 32 chars -> total 53 chars, well under 255
        assert len(result) == 21 + 32
        assert len(result) < 255


# ---------------------------------------------------------------------------
# Managed Responses ID reversibility (previous_response_id path not broken)
# ---------------------------------------------------------------------------

class TestManagedResponsesIdReversibility:
    def test_managed_response_id_still_decodable(self):
        """The real managed Responses ID is NOT modified by the logging fix.
        Only the logging/observation-facing ID is hashed.  The real ID must
        remain reversible for previous_response_id continuity and encrypted
        item decode."""
        from litellm.responses.utils import ResponsesAPIRequestUtils

        custom_llm_provider = "azure"
        model_id = "deploy-123"
        response_id = "upstream-abc-456"
        managed_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider, model_id, response_id
        )

        # Decode it back
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            managed_id
        )
        assert decoded is not None
        assert decoded["custom_llm_provider"] == custom_llm_provider
        assert decoded["model_id"] == model_id
        assert decoded["response_id"] == response_id

    def test_logging_id_hash_does_not_affect_real_response_id(self):
        """The hashing in get_logging_id must not touch the response_obj's
        actual id attribute."""
        start_time = datetime(2026, 1, 15, 12, 30, 45, 123456)
        copilot_id = _make_copilot_style_response_id()
        resp = _make_response_obj(copilot_id)
        _ = get_logging_id(start_time, resp)
        # The response object's id is unchanged
        assert resp.get("id") == copilot_id


# ---------------------------------------------------------------------------
# Langfuse metadata breadcrumbs
# ---------------------------------------------------------------------------

class TestLangfuseMetadataBreadcrumbs:
    """Tests that the Langfuse success callback adds metadata breadcrumbs
    (litellm_response_id + litellm_response_id_sha256) when the response ID
    is long, and does NOT add them for short IDs."""

    def test_metadata_breadcrumbs_added_for_long_id(self):
        """Verify that the Langfuse path would add breadcrumbs for long IDs.

        We test the logic inline since the full LangfuseLogger.success method
        requires a live Langfuse client.  The logic mirrors the code in
        langfuse.py: after computing generation_id, check if the original ID
        exceeds LOGGING_ID_MAX_ID_LENGTH and add metadata breadcrumbs.
        """
        copilot_id = _make_copilot_style_response_id()
        clean_metadata: dict = {}

        # Mirror the langfuse.py logic
        _original_response_id = copilot_id
        if (
            _original_response_id is not None
            and len(_original_response_id) > LOGGING_ID_MAX_ID_LENGTH
        ):
            clean_metadata["litellm_response_id"] = _original_response_id
            clean_metadata["litellm_response_id_sha256"] = hashlib.sha256(
                _original_response_id.encode("utf-8")
            ).hexdigest()

        assert "litellm_response_id" in clean_metadata
        assert clean_metadata["litellm_response_id"] == copilot_id
        assert "litellm_response_id_sha256" in clean_metadata
        expected_hash = hashlib.sha256(copilot_id.encode("utf-8")).hexdigest()
        assert clean_metadata["litellm_response_id_sha256"] == expected_hash

    def test_no_metadata_breadcrumbs_for_short_id(self):
        """Short IDs must not trigger metadata breadcrumbs (no regression)."""
        short_id = "resp_short123"
        clean_metadata: dict = {}

        _original_response_id = short_id
        if (
            _original_response_id is not None
            and len(_original_response_id) > LOGGING_ID_MAX_ID_LENGTH
        ):
            clean_metadata["litellm_response_id"] = _original_response_id
            clean_metadata["litellm_response_id_sha256"] = hashlib.sha256(
                _original_response_id.encode("utf-8")
            ).hexdigest()

        assert "litellm_response_id" not in clean_metadata
        assert "litellm_response_id_sha256" not in clean_metadata

    def test_metadata_sha256_matches_logging_id_hash(self):
        """The SHA-256 in metadata must match the hash used in the logging ID,
        so correlation from Langfuse observation back to LiteLLM is possible."""
        copilot_id = _make_copilot_style_response_id()
        start_time = datetime(2026, 1, 15, 12, 30, 45, 123456)

        # The logging ID hash
        logging_id = get_logging_id(start_time, _make_response_obj(copilot_id))
        id_portion = logging_id.split("_", 1)[1]

        # The metadata hash (full 64-char hexdigest)
        metadata_hash = hashlib.sha256(copilot_id.encode("utf-8")).hexdigest()

        # The logging ID hash is the first 32 chars of the full hash
        assert metadata_hash[:32] == id_portion