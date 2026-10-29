import json

import httpx
import pytest
from syrupy.assertion import SnapshotAssertion
from tests.helpers.llm import CFG, ScriptedLLM, ok, provider

from alpha_factory.agents.flow import call_with_repair
from alpha_factory.agents.llm_client import LLMError, OpenAICompatClient, to_strict_schema
from alpha_factory.contracts import GoalProposals, ProposalBatch, RawProposalBatch, Reflection


def client(responses, seen=None, cfg=CFG):  # type: ignore[no-untyped-def]
    sleeps: list[float] = []
    c = OpenAICompatClient(
        cfg,
        "https://llm.example/v1",
        "KEY",
        transport=provider(responses, seen),
        sleep=sleeps.append,
    )
    return c, sleeps


def walk(node):  # type: ignore[no-untyped-def]
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


@pytest.mark.parametrize("model", [ProposalBatch, Reflection, GoalProposals])
def test_strict_schemas_require_every_property_and_forbid_extras(model) -> None:  # type: ignore[no-untyped-def]
    schema = to_strict_schema(model)
    objects = [n for n in walk(schema) if n.get("type") == "object" and "properties" in n]
    assert objects
    for o in objects:
        assert o["additionalProperties"] is False
        assert set(o["required"]) == set(o["properties"])  # all required
    assert not any("default" in n for n in walk(schema))


def test_optional_fields_become_nullable_and_required_ones_do_not() -> None:
    props = to_strict_schema(Reflection)["$defs"]["BeliefUpdate"]["properties"]
    assert "null" in props["belief_id"]["anyOf"][-1]["type"] or props["belief_id"]["anyOf"][-1] == {
        "type": "null"
    }
    assert props["action"].get("type") != ["string", "null"]  # a required enum stays non-null
    fp = to_strict_schema(ProposalBatch)["$defs"]["FactorProposal"]["properties"]
    assert {"type": "null"} in fp["condition"]["anyOf"]


@pytest.mark.parametrize(
    "model", [ProposalBatch, Reflection, GoalProposals], ids=lambda m: m.__name__
)
def test_strict_schema_snapshot(model, snapshot: SnapshotAssertion) -> None:  # type: ignore[no-untyped-def]
    assert to_strict_schema(model) == snapshot


def test_a_request_carries_the_strict_schema_and_the_response_is_costed() -> None:
    seen: list[httpx.Request] = []
    c, _ = client([ok({"goals": []}, 2000, 1000)], seen)
    r = c.complete_json(role="proposer", system="sys", user="usr", output_model=GoalProposals)
    body = json.loads(seen[0].content)
    assert seen[0].headers["authorization"] == "Bearer KEY" and str(seen[0].url).endswith(
        "/v1/chat/completions"
    )
    assert body["model"] == "m-prop" and body["temperature"] == 0.8 and body["max_tokens"] == 6000
    assert (
        body["response_format"]["type"] == "json_schema"
        and body["response_format"]["json_schema"]["strict"] is True
    )
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    (call,) = r.calls
    assert call.ok and (call.tokens_in, call.tokens_out) == (2000, 1000)
    assert call.cost_usd == pytest.approx((2000 * 1.0 + 1000 * 2.0) / 1e6)
    assert len(call.prompt_hash) == 64 and r.text == '{"goals": []}'


def test_plain_json_mode_puts_the_schema_in_the_prompt() -> None:
    seen: list[httpx.Request] = []
    c, _ = client([ok("{}")], seen, cfg=CFG.model_copy(update={"supports_json_schema": False}))
    c.complete_json(role="manager", system="sys", user="usr", output_model=GoalProposals)
    body = json.loads(seen[0].content)
    assert (
        body["response_format"] == {"type": "json_object"}
        and "GoalProposals" in body["messages"][0]["content"]
    ) or "goals" in body["messages"][0]["content"]
    assert body["model"] == "m-man"


def test_429_and_5xx_are_retried_with_backoff_and_every_attempt_is_recorded() -> None:
    c, sleeps = client([httpx.Response(429), httpx.Response(503), ok({"a": 1})])
    r = c.complete_json(role="proposer", system="s", user="u", output_model=GoalProposals)
    assert [(x.ok, x.error) for x in r.calls] == [
        (False, "HTTP 429"),
        (False, "HTTP 503"),
        (True, None),
    ]
    assert len(sleeps) == 2 and sleeps[1] > 0
    assert r.calls[0].cost_usd == 0.0


def test_transport_errors_are_retried_but_a_bad_request_is_not() -> None:
    c, _ = client([httpx.ConnectError("down"), ok({"a": 1})])
    assert (
        len(c.complete_json(role="reflector", system="s", user="u", output_model=Reflection).calls)
        == 2
    )
    c2, _ = client([httpx.Response(400, json={"error": "bad schema"})])
    with pytest.raises(LLMError) as e:
        c2.complete_json(role="proposer", system="s", user="u", output_model=GoalProposals)
    assert len(e.value.calls) == 1 and not e.value.calls[0].ok


def test_giving_up_after_the_retries_reports_every_attempt() -> None:
    c, _ = client([httpx.Response(500)] * 4)
    with pytest.raises(LLMError, match="4 attempts") as e:
        c.complete_json(role="proposer", system="s", user="u", output_model=GoalProposals)
    assert len(e.value.calls) == 4


# --- the call flow -------------------------------------------------------------------------------------
GOOD = json.dumps({"rationale": "r", "proposals": []})


def test_a_valid_reply_needs_one_call() -> None:
    llm = ScriptedLLM(GOOD)
    res = call_with_repair(
        llm,
        role="proposer",
        system="s",
        user="u",
        output_model=ProposalBatch,
        parse=RawProposalBatch.model_validate_json,
    )  # type: ignore[arg-type]
    assert res.value is not None and len(res.calls) == 1 and res.error is None


def test_an_invalid_reply_gets_one_repair_call_that_includes_the_errors() -> None:
    llm = ScriptedLLM('{"rationale": 5}', GOOD)
    res = call_with_repair(
        llm,
        role="proposer",
        system="s",
        user="u",
        output_model=ProposalBatch,
        parse=RawProposalBatch.model_validate_json,
        contract_hash="c" * 64,
    )  # type: ignore[arg-type]
    assert res.value is not None and len(res.calls) == 2
    repair = llm.requests[1]
    assert (
        repair["role"] == "repair"
        and "failed validation" in repair["user"]
        and '{"rationale": 5}' in repair["user"]
    )
    assert all(c.contract_hash == "c" * 64 for c in res.calls)


def test_a_second_failure_gives_no_value() -> None:
    llm = ScriptedLLM("not json", "still not json")
    res = call_with_repair(
        llm,
        role="proposer",
        system="s",
        user="u",
        output_model=ProposalBatch,
        parse=RawProposalBatch.model_validate_json,
    )  # type: ignore[arg-type]
    assert (
        res.value is None and "invalid after repair" in (res.error or "") and len(llm.requests) == 2
    )


def test_a_provider_failure_keeps_its_call_records() -> None:
    res = call_with_repair(
        ScriptedLLM(),
        role="proposer",
        system="s",
        user="u",
        output_model=ProposalBatch,
        parse=RawProposalBatch.model_validate_json,
    )  # type: ignore[arg-type]
    assert res.value is None and len(res.calls) == 1 and not res.calls[0].ok
