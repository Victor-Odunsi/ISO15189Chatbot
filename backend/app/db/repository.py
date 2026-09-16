from sqlalchemy import delete, select

from app.db.models import Message
from app.db.session import AsyncSessionLocal


async def get_chat_history(session_id: str) -> list[dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message.role, Message.content)
            .where(Message.session_id == session_id)
            .order_by(Message.id)
        )
        return [{"role": role, "content": content} for role, content in result.all()]


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
        await session.commit()
