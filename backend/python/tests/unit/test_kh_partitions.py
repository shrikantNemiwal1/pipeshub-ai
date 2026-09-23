"""Partition descriptors follow §3.9, D7 and D45, whatever order the query returned."""

from app.utils.kh_partitions import APPS_PARTITION_ID, build_partitions


def kinds(partitions):
    return [(p["partitionKind"], p["partitionId"], p["appId"]) for p in partitions]


def test_no_apps_still_searches_the_apps_partition() -> None:
    assert kinds(build_partitions([])) == [("APPS", APPS_PARTITION_ID, None)]


def test_a_collection_is_one_partition_whatever_it_contains() -> None:
    """D7, and PG-03's partition-shape half: its folders are not groups, and its
    direct children do not add an App-direct partition.

    PG-03 also requires both items to carry the collection's role; that half is
    asserted where roles are read, not here.
    """
    rows = [{"appId": "kb-1", "isCollection": True, "groups": ["kf-1"], "hasDirect": True}]
    assert kinds(build_partitions(rows))[1:] == [("COLLECTION", "kb-1", "kb-1")]


def test_a_connector_app_gets_a_partition_per_top_level_group_and_one_app_direct() -> None:
    """PG-02's discovery half: one partition per top-level group, deduplicated.

    PG-02's other half -- that content inside a *nested* group stays in its
    top-level group's partition -- is a traversal property, asserted against the
    real engines in `test_provider_v2_partitions.py`, not decidable from these
    descriptors alone.
    """
    rows = [{"appId": "gp-app", "isCollection": False, "groups": ["rg-b", "rg-a", "rg-a"], "hasDirect": True}]
    assert kinds(build_partitions(rows))[1:] == [
        ("GROUP", "rg-a", "gp-app"), ("GROUP", "rg-b", "gp-app"), ("APP_DIRECT", "gp-app", "gp-app"),
    ]


def test_no_app_direct_partition_without_direct_children() -> None:
    rows = [{"appId": "pl-app", "isCollection": False, "groups": ["pl-rg1"], "hasDirect": False}]
    assert kinds(build_partitions(rows))[1:] == [("GROUP", "pl-rg1", "pl-app")]


def test_the_order_is_stable_whatever_order_the_query_returned() -> None:
    rows = [
        {"appId": "b-app", "isCollection": False, "groups": ["b1"], "hasDirect": False},
        {"appId": "a-app", "isCollection": True, "groups": [], "hasDirect": False},
    ]
    assert build_partitions(rows) == build_partitions(list(reversed(rows)))
    assert [p["partitionId"] for p in build_partitions(rows)] == [APPS_PARTITION_ID, "a-app", "b1"]
