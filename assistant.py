from typing import Optional
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db_url = "postgresql://ai_owner:B9iIwFyus4VO@ep-restless-block-a1e1oiah.ap-southeast-1.aws.neon.tech/ai?sslmode=require"

def get_groq_assistant(
    llm_model: str = "llama3-70b-8192",
    embeddings_model: str = "text-embedding-3-small",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq RAG Assistant."""

    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    embeddings_table = (
        "groq_rag_documents_ollama" if embeddings_model == "nomic-embed-text" else "groq_rag_documents_openai"
    )

    return Assistant(
        name="groq_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="groq_rag_assistant", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=embedder,
            ),
            num_documents=2,
        ),
        # Slytle different instructions results in different grading or grading style.
        description="You are an AI called 'GroqRAG' and your task is to grade student answers based on model answers.",
        instructions=[
            "You will always take two PDF files as input: Model Answer (best answers) and Student Answer.",
            "Don't give marks to the model answers file only use it as a refrance",
            "You should give a grade to each question on the student answer based on the model answer.",
            "Use the model answer as the reference for grading.",
            "A student who provides the meaning of an answer but uses different words and mentions the entire information given in the model answer will receive full marks.",
            "A student who provides incomplete or irrelevant information will lose marks based on the quality and completeness of their answer.",
            "Use a consistent marking technique so that The same answers should always receive the same marks."
            "A question with no answer should receive zero marks."
        ],
        add_references_to_prompt=False,
        markdown=True,
        add_chat_history_to_messages=True,
        num_history_messages=4,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )