"""Regression: shared-backend capability registration must not veto a family.

2026-10-12 production incident: fifteen alias deployments share one backend
key (``openai/gpt-6.1-sol``), which ``register_model`` resolves to the built-in
family entry. Registrations onto that shared key are last-write-wins, and one
alias carrying ``supports_reasoning: False`` overwrote the family entry's
``True``. Every request-path gate reading the family entry then rejected
``reasoning.effort`` with ``UnsupportedParamsError`` ("model cost map entry
lacks supports_reasoning") for a model that does support it — in the long-lived
server process only, since a fresh process sees the pristine map, which made
every fresh-process probe pass while the worker crash-looped.

The shared key already guards ``mode`` against exactly this downgrade
("prevents alias registration from downgrading the shared backend mode").
Capability flags now follow the same rule: a falsy ``supports_*`` flag never
travels the shared key in the negative direction. Positive flags still
propagate (custom backends rely on that), and each deployment's own ``model_id``
entry keeps its exact flags, so an operator's explicit negative remains visible
where deployment-scoped logic reads it.
"""

import litellm
from litellm import Router

_FAMILY = "gpt-6.1-sol"


def _plant_family_flag(monkeypatch, value: bool) -> None:
    """Make the premise deterministic regardless of map drift."""
    entry = dict(litellm.model_cost.get(_FAMILY, {}))
    entry["supports_reasoning"] = value
    monkeypatch.setitem(litellm.model_cost, _FAMILY, entry)


def _aliases(*supports_values: bool) -> Router:
    return Router(
        model_list=[
            {
                "model_name": f"alias-{index}",
                "litellm_params": {"model": f"openai/{_FAMILY}", "api_key": "test-key"},
                "model_info": {"supports_reasoning": value},
            }
            for index, value in enumerate(supports_values)
        ]
    )


def test_family_capability_survives_alias_with_negative_flag(monkeypatch):
    """The incident: one alias's False must not veto the built-in family entry."""
    _plant_family_flag(monkeypatch, True)
    _aliases(True, False, False)
    assert litellm.model_cost[_FAMILY]["supports_reasoning"] is True


def test_negative_alias_still_registers_unchanged_on_its_own_entry(monkeypatch):
    """The deployment's model_id entry keeps its exact flags (the escape hatch)."""
    _plant_family_flag(monkeypatch, True)
    router = _aliases(False)
    deployment_id = router.get_model_list()[0]["model_info"]["id"]
    entry = litellm.model_cost.get(deployment_id)
    assert entry is not None
    assert entry.get("supports_reasoning") is False


def test_positive_flags_still_propagate_for_custom_backends(monkeypatch):
    """A custom backend gains capability flags from its aliases.

    The ``openai/`` prefix is provider resolution only — the model name is
    unknown to the cost map, which is the point: an unpriced custom backend
    must still receive positive capability flags.
    """
    _plant_family_flag(monkeypatch, True)
    Router(
        model_list=[
            {
                "model_name": "custom-a",
                "litellm_params": {"model": "openai/my-proprietary-model", "api_key": "test-key"},
                "model_info": {"supports_reasoning": True},
            }
        ]
    )
    entry = litellm.model_cost.get("openai/my-proprietary-model")
    assert entry is not None
    assert entry.get("supports_reasoning") is True
