import streamlit as st
from phi.assistant import Assistant
from phi.document.reader.pdf import PDFReader
from phi.utils.log import logger
from assistant import get_groq_assistant
import io
import os

# environment variables
os.environ['GROQ_API_KEY'] = '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'


st.set_page_config(
    page_title="Test Corrector Model"
)
st.title("Test Corrector Model")
st.markdown("##### Upload Model Answer and Student Answer PDFs to get the grades")

def restart_assistant():
    st.session_state["assistant"] = None
    st.session_state["assistant_run_id"] = None
    st.rerun()

def main():
    # Get LLM model
    llm_model = st.sidebar.selectbox("Select LLM", options=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"])
    embeddings_model = st.sidebar.selectbox("Select Embeddings", options=["nomic-embed-text", "text-embedding-3-small"])

    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        restart_assistant()

#type annotation in Python. It indicates that the variable assistant is expected to be an instance of the Assistant class.
    assistant: Assistant 
    if "assistant" not in st.session_state or st.session_state["assistant"] is None:
        logger.info(f"---*--- Creating {llm_model} Assistant ---*---")
        assistant = get_groq_assistant(llm_model=llm_model, embeddings_model=embeddings_model)
        st.session_state["assistant"] = assistant
    else:
        assistant = st.session_state["assistant"]

    try:
        st.session_state["assistant_run_id"] = assistant.create_run()
    except Exception:
        st.warning("Could not create assistant, is the database running?")
        return

    # Upload model answer PDF
    model_answer_pdf = st.file_uploader("Upload Model Answer PDF", type="pdf")
    model_answers = []
    if model_answer_pdf:
        reader = PDFReader()
        model_documents = reader.read(io.BytesIO(model_answer_pdf.read()))
        model_answers = [doc.content for doc in model_documents]

    # Upload student answer PDF
    student_answer_pdf = st.file_uploader("Upload Student Answer PDF", type="pdf")
    student_answers = []
    if student_answer_pdf:
        reader = PDFReader()
        student_documents = reader.read(io.BytesIO(student_answer_pdf.read()))
        student_answers = [doc.content for doc in student_documents]

    # Grade answers
    if st.button("Grade Answers"):
        if model_answers and student_answers:
            grades = []
            # for model_answer, student_answer in zip(model_answers, student_answers):
            prompt = f"Grade the following student answer based on the model answer:\n\nModel Answer: {[doc.content for doc in model_documents]}\n\nStudent Answer: {[doc.content for doc in student_documents]}"
            response_generator = assistant.run(prompt)
            response = ''.join(list(response_generator))
            grades.append(response)
            for i, grade in enumerate(grades, 1):
                st.write(f"{grade}")
        else:
            st.warning("Please upload both Model Answer PDF and Student Answer PDF")

main()