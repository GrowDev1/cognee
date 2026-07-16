"""Regression: DLT orphan cleanup must purge the graph + vector stores, not just
the relational one.

`_delete_dlt_orphans` previously gated the graph/vector delete on
`has_data_related_nodes` (a relational-ledger check). On the default Ladybug
graph-provenance stack that check is False, so deleted DLT rows were removed from
the relational store but left behind in the graph and vector stores — stale data
stayed searchable. This ingests two rows through the real add+cognify pipeline
(local Ladybug + LanceDB + fastembed, mocked LLM), deletes one via a hard-delete
marker, drives orphan cleanup, and asserts the row is gone from all three stores.
"""

import hashlib
import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user

DATASET = "dlt_purge_ds"


@pytest_asyncio.fixture
async def clean_env(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")
    pytest.importorskip("fastembed")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("LLM_API_KEY", "sk-mocked")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "384")
    root = pathlib.Path(tmp_path)
    monkeypatch.setenv("DLT_DATA_DIR", str(root / "dlt"))

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()
    yield
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


def _mock_llm():
    """Patch the LLM's structured output with canned graph + summary."""
    from unittest.mock import patch

    from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

    async def _out(text_input, system_prompt, response_model, **kw):
        name = getattr(response_model, "__name__", "")
        if name == "KnowledgeGraph":
            h = hashlib.md5((text_input or "").encode()).hexdigest()[:8]
            return KnowledgeGraph(
                nodes=[
                    Node(id=f"c_{h}_a", name=f"c_{h}_a", type="Concept", description="x"),
                    Node(id=f"c_{h}_b", name=f"c_{h}_b", type="Concept", description="x"),
                ],
                edges=[
                    Edge(
                        source_node_id=f"c_{h}_a",
                        target_node_id=f"c_{h}_b",
                        relationship_name="rel",
                    )
                ],
            )
        if name == "SummarizedContent":
            return SummarizedContent(summary=(text_input or "")[:120], description="")
        return response_model()

    return patch(
        "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output",
        side_effect=_out,
    )


def _dlt_source(rows):
    import dlt

    @dlt.resource(
        name="widgets",
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def widgets():
        yield from rows

    return widgets


async def _store_counts():
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    nodes, _ = await (await get_graph_engine()).get_graph_data()
    ve = await get_vector_engine_async()
    try:
        vec = await (await ve.get_collection("DocumentChunk_text")).count_rows()
    except Exception:
        vec = 0
    return len(nodes), vec


async def _dlt_pks(user):
    ds = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DATASET]
        )
    )[0]
    rows = await get_dataset_data(ds.id)
    return sorted(
        d.external_metadata.get("primary_key_value")
        for d in rows
        if isinstance(d.external_metadata, dict) and d.external_metadata.get("source") == "dlt"
    )


@pytest.mark.asyncio
async def test_dlt_orphan_cleanup_purges_graph_and_vector(clean_env):
    from cognee.tasks.ingestion.resolve_dlt_sources import resolve_dlt_sources

    user = await get_default_user()
    kwargs = dict(primary_key="id", write_disposition="merge", max_rows_per_table=0)

    with _mock_llm():
        # Ingest two rows through the real add + cognify pipeline.
        await cognee.remember(
            _dlt_source(
                [
                    {
                        "id": "a",
                        "body": "alpha runbook restart the payments service",
                        "_deleted": False,
                    },
                    {
                        "id": "b",
                        "body": "beta onboarding request vpn access from it",
                        "_deleted": False,
                    },
                ]
            ),
            dataset_name=DATASET,
            **kwargs,
        )
        assert await _dlt_pks(user) == ["a", "b"]
        nodes_before, vec_before = await _store_counts()
        assert nodes_before > 0 and vec_before == 2  # graph populated, 2 chunks

        # Delete 'b' upstream: resolve_dlt_sources runs the merge (hard-deletes b
        # from the dlt destination) and returns the deferred orphan cleanup.
        _, orphan_cleanup = await resolve_dlt_sources(
            _dlt_source([{"id": "b", "_deleted": True}]), dataset_name=DATASET, user=user, **kwargs
        )
        assert orphan_cleanup is not None
        await orphan_cleanup()

    # 'b' must be gone from ALL three stores.
    assert await _dlt_pks(user) == ["a"]  # relational
    nodes_after, vec_after = await _store_counts()
    assert vec_after == 1, f"vector not purged: {vec_before} -> {vec_after} (expected 1)"
    assert nodes_after < nodes_before, f"graph not purged: {nodes_before} -> {nodes_after}"
