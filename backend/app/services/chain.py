import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.models.schemas import QueryAnalysis
from app.services.llm import get_llm
from app.services.retrieval import hybrid_retrieve

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a chat history and the latest user question, do two things:\n"
        "1. Rewrite the question as a standalone question that can be understood "
        "without the chat history. If it already stands alone, leave it as is. "
        "For greetings or small talk, leave it as is.\n"
        "2. Classify the user's intent into exactly one of:\n"
        "- 'chitchat': ONLY greetings (hi, hello), thanks, farewells, or direct "
        "questions about the assistant itself (e.g. 'what can you do', 'who are "
        "you'). Never use this for any question seeking actual information, "
        "even if that information is unrelated to ISO 15189 -- 'what's the "
        "capital of France?' or 'what's the weather today?' are real questions "
        "and must be 'general', not 'chitchat'. Only use 'chitchat' when the "
        "user is not asking for any information at all.\n"
        "- 'general': any direct question or request for information or "
        "explanation, whether or not it relates to ISO 15189.\n"
        "- 'checklist': the user wants an audit/compliance checklist.\n"
        "- 'sop': the user wants a Standard Operating Procedure document."
    )),
    ('placeholder', '{chat_history}'),
    ("human", "{input}"),
])

CHITCHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly assistant for the ISO 15189:2022 laboratory
quality management standard. Respond briefly and naturally to greetings, thanks,
or questions about what you can do. If relevant, mention that you can answer
questions about the standard, generate compliance checklists, or draft SOPs --
all grounded in the retrieved text, with citations. Do not invent ISO 15189
content here; this is small talk, not a content question.
"""),
    ('placeholder', '{chat_history}'),
    ("human", "{standalone_question}"),
])

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ISO 15189 expert assistant with your own knowledge of
the standard, grounded in the retrieved excerpts below — ground every claim in
them, and never use outside knowledge. These excerpts are your own reference
material, not something the user supplied; never phrase a response as "the
text you provided" or imply the user gave you this content. If the excerpts
don't cover what's needed to answer fully, say so plainly and directly as
your own knowledge gap (e.g. "the standard doesn't specify..." or "I don't
have enough detail on..."), not as a complaint about the user's input.
Keep the style professional and concise.

Context:
{context}
"""),
    ('placeholder', '{chat_history}'),
    ("human", "{standalone_question}"),
])

CHECKLIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ISO 15189 internal audit checklist generator, drawing
on your own knowledge of the standard via the retrieved excerpts below (your
own reference material, not something the user supplied). Convert them into a
practical, numbered checklist of concise yes/no compliance questions, grouped
into sections if the content is long. Use only this material — never outside
knowledge. If it doesn't cover enough to build a meaningful checklist, say so
plainly as your own knowledge gap, not as a complaint about the user's input.

Context:
{context}
"""),
    ('placeholder', '{chat_history}'),
    ("human", "{standalone_question}"),
])

SOP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ISO 15189 SOP author, drawing on your own knowledge of
the standard via the retrieved excerpts below (your own reference material,
not something the user supplied). Convert them into a polished Standard
Operating Procedure with these sections: Purpose, Scope, Responsibilities,
Procedure, and References. Use only this material — never outside knowledge.
If it doesn't cover enough to produce a meaningful SOP, say so plainly as
your own knowledge gap, not as a complaint about the user's input.

Context:
{context}
"""),
    ('placeholder', '{chat_history}'),
    ("human", "{standalone_question}"),
])

_GENERATION_PROMPTS = {
    "chitchat": CHITCHAT_PROMPT,
    "general": GENERAL_PROMPT,
    "checklist": CHECKLIST_PROMPT,
    "sop": SOP_PROMPT,
}


async def analyze_query(question: str, chat_history: list[dict]) -> QueryAnalysis:
    chain = ANALYSIS_PROMPT | get_llm().with_structured_output(QueryAnalysis)
    return await chain.ainvoke({"input": question, "chat_history": chat_history})


def retrieve_context(standalone_question: str) -> list[Document]:
    return hybrid_retrieve(standalone_question)


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def build_citations(documents: list[Document]) -> list[dict]:
    seen = set()
    citations = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None  # PyPDFLoader pages are 0-indexed
        key = (source, page_number)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"source": source, "page": page_number})
    return citations


def get_generation_chain(intent: str) -> Runnable:
    prompt = _GENERATION_PROMPTS.get(intent, GENERAL_PROMPT)
    return prompt | get_llm() | StrOutputParser()
