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
        "without the chat history. If it already stands alone, leave it as is.\n"
        "2. Classify the user's intent: 'general' for a direct question or "
        "explanation, 'checklist' if they want an audit/compliance checklist, "
        "'sop' if they want a Standard Operating Procedure document."
    )),
    ('placeholder', '{chat_history}'),
    ("human", "{input}"),
])

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ISO 15189 expert assistant. Answer strictly using the
retrieved context below — ground every claim in it, and never use outside
knowledge. If the context is insufficient to answer, say so plainly.
Keep the style professional and concise.

Context:
{context}
"""),
    ('placeholder', '{chat_history}'),
    ("human", "{standalone_question}"),
])

CHECKLIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ISO 15189 internal audit checklist generator. Convert
the retrieved context below into a practical, numbered checklist of concise
yes/no compliance questions, grouped into sections if the content is long.
Use only the retrieved context — never outside knowledge.

Context:
{context}
"""),
    ('placeholder', '{chat_history}'),
    ("human", "{standalone_question}"),
])

SOP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ISO 15189 SOP author. Convert the retrieved context
below into a polished Standard Operating Procedure with these sections:
Purpose, Scope, Responsibilities, Procedure, and References. Use only the
retrieved context — never outside knowledge.

Context:
{context}
"""),
    ('placeholder', '{chat_history}'),
    ("human", "{standalone_question}"),
])

_GENERATION_PROMPTS = {
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
