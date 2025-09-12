from config import groq_api_key, mistral_api_key
import os, re
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from sqldb_utils import insert_application_logs
from sqldb_utils import get_chat_history
from chromadb_utils import get_chroma
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentFinish
import logging
import asyncio

logger  = logging.getLogger(__name__)

def get_llm():
    try:
        return ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=groq_api_key,
        temperature = 0.0,
        streaming = True
        )
    except Exception as e:
        logging.error(f'Groq Initialization failed: {e}')
        return ChatMistralAI(
            model="mistral-small-3.1", 
            temperature=0.0,
            streaming = True
        )

template = ChatPromptTemplate.from_messages(
    [
        ("system",'''
        Given a chat history and a latest question which might need context from the chat history, 
        formulate a standalone question that can be understood without the chat history.
        DO NOT answer the question. Only reformulate the question if needed or leave as is
        '''),
        ('placeholder', '{chat_history}'),
        ("human", "{input}"),
    ]
)

vectorDB = get_chroma()
retriever = vectorDB.as_retriever(search_kwargs = {'k' : 2})
history_aware_retriever = create_history_aware_retriever(
    llm=get_llm(),
    retriever=retriever,
    prompt=template
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an ISO 15189 expert assistant. Your job is to answer questions and create outputs 
strictly using the retrieved context provided from the RAG pipeline.  

Rules for using retrieved documents:
- ALWAYS ground your answers in the retrieved ISO 15189 content. Do not make up or assume anything.  
- If the retrieved context is insufficient, clearly say you cannot find relevant information.  
- NEVER use outside knowledge beyond what is retrieved.  
- When formatting answers, keep the style professional, concise, and aligned with ISO 15189 standards.  

Output requirements:
- If the user asks for a general explanation: provide a direct, well-structured answer using only retrieved content.  
- If the user asks for a checklist: convert the retrieved information into a clear, actionable checklist format.  
- If the user asks for an SOP: structure the answer into **Purpose, Scope, Responsibilities, Procedure, and References**.  
- Always ensure the final text is clean and ready for the user, without mentioning retrieval steps or tool usage.  

If unsure, provide the most relevant retrieved information as-is, formatted clearly.  

Answer the user questions based on the following context: {context}
"""),
    ('placeholder', '{chat_history}'),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(
    llm=get_llm(),
    prompt=qa_prompt
)

retrieval_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=qa_chain
)

model = get_llm()
def make_rag_answer_tool(session_id: str):
    @tool('rag_answer')
    def rag_answer(question: str):
        """
        Answer ISO 15189 questions using the RAG pipeline.
        Args:
          question: the user's question
        Returns:
          answer text string
        """
        chat_history = get_chat_history(session_id)
        result = retrieval_chain.invoke(
          {
            'input': question,
            'chat_history': chat_history
          }
        )
        answer = result['answer'] if isinstance(result, dict) else str(result)

        return {'output': answer}
    return rag_answer

def make_create_checklist(session_id: str):
    @tool('create_checklist')
    def create_checklist(question: str):
        """
        Converts retrieved ISO 15189 text into a practical checklist 
        using LLM + retrieval pipeline.
        """
        chat_history = get_chat_history(session_id)
        retrieved = retrieval_chain.invoke(
            {"input": question, "chat_history": chat_history}
        )

        retrieved_text = (
            retrieved.get("answer") 
            if isinstance(retrieved, dict) 
            else str(retrieved)
        )
        
        prompt = f"""
        You are an ISO 15189 Internal Audit Checklist generator.  
        Your task is to convert the following standard text into a practical checklist.  

        Guidelines:  
        - Write concise yes/no style questions.  
        - Focus on compliance, documentation, staff competency, and process adherence.  
        - Number the questions.  
        - Group into logical sections if the content is long.

        Context:  
        {retrieved_text}  

        Checklist:
        """
        raw_output = model.invoke(prompt)

        # make sure we only store string content
        checklist_text = raw_output.content if hasattr(raw_output, "content") else str(raw_output)

        return {'output': checklist_text}
    return create_checklist

@tool('final_answer')
def final_answer(answer: str):
    """
    Provide the final answer to the user. Use this tool IMMEDIATELY after getting information from rag_answer or create_checklist.
    Args:
      answer: the final answer to provide to the user (clean text without mentioning tools or actions)
    Returns:
      the final answer
    """
    return answer

@tool('format_sop')
def format_sop(raw_text: str) -> str:
    """
    Takes raw draft content (e.g., from rag_answer) and formats it into a 
    polished, structured SOP according to ISO 15189 style.
    Always include Purpose, Scope, Responsibilities, Procedure, and References sections.
    """
    llm = ChatMistralAI(model="mistral-tiny", temperature=0)
    response = llm.invoke(f"Format the following draft into a professional SOP:\n\n{raw_text}")
    return response.content



class DummyHandler(BaseCallbackHandler):
    def __init__(self):
        self.captured_output = ""   # For internal tool output capture
        self.final_answer_output = ""  # Only what should go to UI
        self.found_answer = False
        self.final_answer_called = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        tool_name = kwargs.get("tool_name")
        # Only append tokens from final_answer tool for streaming
        if tool_name == "final_answer":
            self.final_answer_output += token
        else:
            self.captured_output += token

    def on_tool_end(self, output, **kwargs) -> None:
        tool_name = kwargs.get("tool_name")
        
        # Mark if a final_answer tool was called
        if tool_name == "final_answer":
            self.final_answer_called = True
            if isinstance(output, dict) and "output" in output:
                self.final_answer_output += output["output"]
            else:
                self.final_answer_output += str(output)
        else:
            # Capture other tool outputs (rag_answer / format_sop)
            if isinstance(output, dict) and "output" in output:
                self.captured_output += output["output"]
            else:
                self.captured_output += str(output)

def get_streaming_llm(callbacks = None):
    try:
        return ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=groq_api_key,
        temperature = 0.0,
        streaming = True,
        max_retries=1,
        request_timeout=100,
        callbacks = callbacks or []
        )
    except Exception as e:
        logging.error(f'Groq Initialization failed: {e}')
        return ChatMistralAI(
            model="mistral-small-3.1", 
            temperature=0.0,
            streaming = True,
            max_retries=1,
            request_timeout=100,
            callbacks = callbacks or []
        )


def get_chat_agent(session_id: str, handler):
    rag_tool = make_rag_answer_tool(session_id)
    checklist_tool = make_create_checklist(session_id)

    streaming_model = get_streaming_llm(callbacks = [handler])

    agent = initialize_agent(
        tools=[rag_tool, checklist_tool, final_answer, format_sop],
        llm=streaming_model,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=False,
        max_iterations=3,
        early_stopping_method="generate",
        max_execution_time=60,
        handle_parsing_errors=True,
        agent_kwargs={
            'prefix': """You are an ISO 15189 expert assistant.

Tool usage rules:
- ALWAYS call the `rag_answer` tool first for any question related 
  to ISO 15189, laboratory standards, quality management, clauses, or measurement procedures.
- Use the `create_checklist` tool ONLY when the user explicitly asks 
  for a checklist or audit checklist.
- After retrieving raw content with `rag_answer`, use the `format_sop` tool 
  whenever the user’s request is an SOP or requires professional formatting.
- Use the `final_answer` tool to produce the clean, user-facing output, NOT JUST A SUMMARY
- NEVER answer from your own knowledge base. Use only content from the tools.
- NEVER return an answer without calling the correct tools.
- Do not use `create_checklist` for SOPs or general explanations.
- If a user asks for a PDF or Word document, politely explain that you only provide text output.
- Always decide the best tool according to the request.
- If you are unsure, default to `rag_answer`.

CRITICAL formatting rules:
1. You MUST call tools in this order for SOP requests:
   rag_answer → format_sop → final_answer
2. For non-SOP requests: rag_answer → final_answer
3. NEVER include the words "Thought:", "Action:", "Action Input:", "Observation:", tool names, or intermediate reasoning in your final output.
4. Provide ONLY the HELPFUL, formatted answer directly.
5. STOP immediately after giving the final answer.  
"""
        },
        callbacks = [handler]
    )
    return agent