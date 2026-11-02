from langchain_core.documents import Document

from app.services.retrieval import reciprocal_rank_fusion


def _doc(chunk_id: str, text: str = "") -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


def test_rrf_favors_documents_ranked_highly_in_multiple_lists():
    dense = [_doc("a"), _doc("b"), _doc("c")]
    sparse = [_doc("b"), _doc("a"), _doc("d")]

    fused = reciprocal_rank_fusion([dense, sparse])
    fused_ids = [doc.metadata["chunk_id"] for doc in fused]

    # "a" and "b" are ranked near the top of both lists, so they should
    # outrank "c" and "d", each of which only appears in one list.
    assert set(fused_ids[:2]) == {"a", "b"}
    assert set(fused_ids[2:]) == {"c", "d"}


def test_rrf_dedupes_by_chunk_id():
    dense = [_doc("a"), _doc("b")]
    sparse = [_doc("a"), _doc("c")]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert len(fused) == 3
    assert len({doc.metadata["chunk_id"] for doc in fused}) == 3


def test_rrf_skips_documents_without_chunk_id():
    fused = reciprocal_rank_fusion([[Document(page_content="no id")]])
    assert fused == []
