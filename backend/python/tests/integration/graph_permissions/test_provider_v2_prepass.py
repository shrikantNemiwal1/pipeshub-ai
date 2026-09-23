"""The precomputed visibility sets must answer exactly as the per-hop rule does.

`visible_ids`/`below_seed_ids` are opt-in, so every other module in this suite
exercises only the per-hop path. Without this module "421 green" would say
nothing whatever about the substitution -- which is the whole point of it.

So each partition is queried twice, once each way, and the results compared.
The comparison is the test; the timings live outside the suite.
"""

import pytest

pytestmark = pytest.mark.integration

USER = "user-u"
ORG = "org-1"
GRANTEES = ["user-u", "group-g", "role-r", "team-t", "orgnode-1"]
GATED_APPS = [
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "ex-app",
    "swm-app", "pl-app", "gp-app", "flag-app", "kb-1", "deep-app",
]

_MODE = {"GROUP": "group", "APP_DIRECT": "app_direct"}
EMPTY = {
    "search_query": None, "node_types": None, "record_types": None,
    "indexing_status": None, "created_at": None, "updated_at": None,
    "size": None, "origins": None, "connector_ids": None,
    "record_group_ids": None, "only_containers": False,
}


async def _partition(provider, part, **extra):
    """One partition, exactly as kh_search._query drives it."""
    return await provider.get_knowledge_hub_children_v2(
        user_key=USER, org_id=ORG, parent_id=part["partitionId"], limit=200,
        grantee_ids=GRANTEES, gated_app_ids=GATED_APPS,
        sort_field="name", sort_dir="ASC", flatten=True,
        partition=_MODE.get(part["partitionKind"]),
        direction="next", include_ids=True, after=None, **EMPTY, **extra,
    )


def _rows(result):
    return [r["id"] for r in result["partitions"][0]["rows"]]


def _ids(result):
    out = result["partitions"][0].get("ids") or []
    return sorted(x["id"] if isinstance(x, dict) else x for x in out)


async def test_the_prepass_answers_as_the_per_hop_rule_does(
    loaded_graph, neo4j_provider
) -> None:
    """Every partition, both ways, same answer.

    A difference here means the pre-pass and the rule disagree about who may see
    what -- the failure that matters, in either direction: an id the rule
    returns and the pre-pass loses is a denial, one the pre-pass adds is a leak.
    """
    parts = await neo4j_provider.get_knowledge_hub_partitions_v2(
        gated_app_ids=GATED_APPS, org_id=ORG,
    )
    sets_by_app = {}
    for app_id in GATED_APPS:
        sets_by_app[app_id] = await neo4j_provider.get_knowledge_hub_visible_sets_v2(
            app_id=app_id, grantee_ids=GRANTEES, gated_app_ids=GATED_APPS,
        )

    compared, with_rows, diffs = 0, 0, []
    for part in parts:
        if part["partitionKind"] not in ("GROUP", "APP_DIRECT", "COLLECTION"):
            continue
        sets = sets_by_app.get(part["appId"])
        if sets is None:
            continue
        baseline = await _partition(neo4j_provider, part)
        candidate = await _partition(
            neo4j_provider, part,
            visible_ids=sets["visible_ids"],
            below_seed_ids=sets["below_seed_ids"],
        )
        compared += 1
        if _rows(baseline):
            with_rows += 1
        if _rows(baseline) != _rows(candidate) or _ids(baseline) != _ids(candidate):
            diffs.append({
                "partition": f"{part['partitionKind']}:{part['partitionId']}",
                "lost": sorted(set(_ids(baseline)) - set(_ids(candidate)))[:8],
                "gained": sorted(set(_ids(candidate)) - set(_ids(baseline)))[:8],
            })

    # A comparison over nothing passes trivially, and so does one where every
    # partition is empty: neither would exercise the substitution at all.
    assert compared >= 5, f"only {compared} partitions compared; test is vacuous"
    assert with_rows >= 1, "no partition returned any row; nothing was exercised"
    assert any(s["visible_ids"] for s in sets_by_app.values()), (
        "every visible set was empty, so membership was never the deciding term"
    )
    assert not diffs, f"pre-pass disagrees with the per-hop rule: {diffs}"


async def test_the_sets_are_built_from_the_shipped_rule(neo4j_provider) -> None:
    """Pin the anti-drift property, which behaviour cannot show.

    The pre-pass is only safe because it is generated from `_kh_v2_rule_cypher`.
    Hand-writing its Cypher would let the two forms diverge silently the next
    time the rule changes, and every test above would still pass on the day of
    the change.
    """
    seen = {}
    real = neo4j_provider._kh_v2_rule_cypher

    def spy(*args, **kwargs):
        text = real(*args, **kwargs)
        seen.setdefault(kwargs.get("allow_strict", "$allowStrict"), text)
        return text

    neo4j_provider._kh_v2_rule_cypher = spy
    try:
        await neo4j_provider.get_knowledge_hub_visible_sets_v2(
            app_id="pl-app", grantee_ids=GRANTEES, gated_app_ids=GATED_APPS,
        )
    finally:
        neo4j_provider._kh_v2_rule_cypher = real

    assert "$allowStrict" in seen, "the visible set must use the shipped rule"
    assert "false" in seen, "the seed set must use the shipped strict-refused rule"
