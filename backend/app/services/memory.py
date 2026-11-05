"""
Bounded conversation memory: the LLM sees a running summary of older turns
plus only the most recent messages verbatim, instead of the whole transcript
(which grew the prompt -- and its latency and token cost -- every turn).

Request path:  get_chat_context()       -- 1 DB round trip, no LLM call
After reply:   maybe_update_summary()   -- folds old messages into the summary
                                           once enough have accumulated; runs
                                           after the user already has their answer
"""
import logging

from app.db.repository import get_unsummarized_messages, upsert_summary
from app.services.chain import summarize_messages

logger = logging.getLogger(__name__)

# Last 3 user/assistant turns stay verbatim -- enough for follow-ups like
# "how do I measure it for glucose" to resolve "it".
RECENT_MESSAGES = 6
# Summarize in batches rather than every turn: one summarization LLM call per
# 2 turns, while the verbatim tail stays between 6 and 9 messages.
SUMMARIZE_BATCH = 4
# Hard cap on unsummarized messages loaded, in case summarization keeps failing.
MAX_CONTEXT_MESSAGES = RECENT_MESSAGES + 2 * SUMMARIZE_BATCH


async def get_chat_context(session_id: str) -> list[dict]:
    summary, messages = await get_unsummarized_messages(session_id, MAX_CONTEXT_MESSAGES)
    context = [{"role": m.role, "content": m.content} for m in messages]
    if summary is not None:
        context.insert(0, {"role": "system", "content": f"Summary of the earlier conversation: {summary.summary}"})
    return context


async def maybe_update_summary(session_id: str) -> bool:
    """Returns True if the summary was updated. Never raises: a failed
    summarization only means the verbatim tail grows until the next attempt."""
    try:
        summary, messages = await get_unsummarized_messages(session_id, MAX_CONTEXT_MESSAGES)
        if len(messages) < RECENT_MESSAGES + SUMMARIZE_BATCH:
            return False
        to_fold = [{"role": m.role, "content": m.content} for m in messages[:-RECENT_MESSAGES]]
        new_summary = await summarize_messages(summary.summary if summary else None, to_fold)
        await upsert_summary(session_id, new_summary, messages[-RECENT_MESSAGES - 1].id)
        return True
    except Exception:
        logger.exception(f"Summary update failed for session {session_id}")
        return False
