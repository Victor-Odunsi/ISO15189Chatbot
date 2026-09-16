from app.services.ingestion import _stable_chunk_id


def test_stable_chunk_id_is_deterministic():
    id1 = _stable_chunk_id("doc.pdf", 0, "hello world")
    id2 = _stable_chunk_id("doc.pdf", 0, "hello world")
    assert id1 == id2


def test_stable_chunk_id_changes_with_content():
    id1 = _stable_chunk_id("doc.pdf", 0, "hello world")
    id2 = _stable_chunk_id("doc.pdf", 0, "goodbye world")
    assert id1 != id2


def test_stable_chunk_id_changes_with_source_or_index():
    base = _stable_chunk_id("doc.pdf", 0, "hello world")
    assert base != _stable_chunk_id("other.pdf", 0, "hello world")
    assert base != _stable_chunk_id("doc.pdf", 1, "hello world")
