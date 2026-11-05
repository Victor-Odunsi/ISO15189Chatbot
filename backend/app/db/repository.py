from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import Message, SessionSummary
from app.db.session import AsyncSessionLocal


async def get_chat_history(session_id: str) -> list[dict]:
    """The full transcript -- for the UI. The LLM gets services.memory.get_chat_context()."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message.role, Message.content)
            .where(Message.session_id == session_id)
            .order_by(Message.id)
        )
        return [{"role": role, "content": content} for role, content in result.all()]


async def get_unsummarized_messages(session_id: str, limit: int) -> tuple[SessionSummary | None, list[Message]]:
    """The session's running summary (if any) plus up to `limit` of the most
    recent messages it doesn't cover yet, oldest first. `limit` is a hard
    backstop: summarization normally keeps far fewer than this unsummarized,
    but if it keeps failing, the context must still stay bounded."""
    async with AsyncSessionLocal() as session:
        summary = await session.get(SessionSummary, session_id)
        rows = (
            await session.execute(
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.id > (summary.summarized_up_to_id if summary else 0),
                )
                .order_by(Message.id.desc())
                .limit(limit)
            )
        ).scalars().all()
    return summary, list(reversed(rows))


async def upsert_summary(session_id: str, summary: str, summarized_up_to_id: int) -> None:
    async with AsyncSessionLocal() as session:
        stmt = pg_insert(SessionSummary).values(
            session_id=session_id, summary=summary, summarized_up_to_id=summarized_up_to_id
        )
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[SessionSummary.session_id],
                set_={
                    "summary": stmt.excluded.summary,
                    "summarized_up_to_id": stmt.excluded.summarized_up_to_id,
                    "updated_at": func.now(),
                },
            )
        )
        await session.commit()


async def insert_message(session_id: str, role: str, content) -> None:
    if hasattr(content, "content"):
        content = content.content
    elif isinstance(content, dict):
        content = str(content.get("output"))
    elif not isinstance(content, str):
        content = str(content)

    async with AsyncSessionLocal() as session:
        session.add(Message(session_id=str(session_id), role=role, content=content))
        await session.commit()


async def delete_session(session_id: str) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(delete(Message).where(Message.session_id == session_id))
        await session.execute(delete(SessionSummary).where(SessionSummary.session_id == session_id))
        await session.commit()
