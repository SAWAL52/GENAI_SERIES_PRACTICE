
import os
import uuid
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🩺",
    layout="wide"
)


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        streaming=True,
        api_key=os.getenv("GROQ_API_KEY")
    )


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()

    return str(content).strip()


def extract_patient_information(vector_store, llm):
    queries = [
        "Patient name, age, gender, date of birth and patient identification number",
        "Patient diagnosis, admission date and medical history",
        "Patient symptoms, clinical findings and treatment"
    ]

    documents = []

    for query in queries:
        documents.extend(
            vector_store.similarity_search(query, k=5)
        )

    unique_documents = {}

    for document in documents:
        unique_documents[document.page_content] = document

    context = "\n\n".join(
        f"Page {doc.metadata.get('page', 0) + 1}: {doc.page_content}"
        for doc in unique_documents.values()
    )

    prompt = f"""
Extract the basic patient information from this medical report.

Return these fields:
- Patient Name
- Age
- Gender
- Date of Birth
- Patient ID
- Primary Diagnosis
- Admission Date

Use only explicitly documented information.
Do not confuse the patient with doctors, relatives or other people.
Do not guess or invent missing information.
For missing fields, write "Not mentioned".

Medical report:
{context}
"""

    response = llm.invoke(prompt)
    return get_response_text(response)


def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text found in the PDF.")

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"medical_{uuid.uuid4().hex}",
        persist_directory="./vector_db"
    )


def get_history():
    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in st.session_state.messages[-8:]
    )


def rewrite_question(question, history, llm):
    prompt = f"""
Rewrite the user's latest question as a standalone question.

Use the conversation history to understand references such as
"it", "that disease", "this condition", and "the patient".

Do not answer the question.
Do not change its meaning.

Conversation:
{history}

Latest question:
{question}

Return only the rewritten question.
"""

    response = llm.invoke(prompt)
    return get_response_text(response)


def check_relevance(question, history, report_text, llm):
    prompt = f"""
You are a relevance checker for a medical PDF chatbot.

Decide whether the question is related to the uploaded medical report.

RELATED includes:
- Questions about the patient and information in the report.
- Questions about diagnoses, diseases, symptoms, medicines and test results
  documented in the report.
- Follow-up questions about a topic previously discussed from the report.
- Requests to explain a medical term or disease mentioned in the report.

UNRELATED includes questions about politics, sports, entertainment,
current affairs or other topics unrelated to the report.

Conversation history:
{history}

User question:
{question}

Report content:
{report_text[:12000]}

Return exactly RELATED or UNRELATED.
"""

    response = llm.invoke(prompt)
    result = get_response_text(response).upper()

    return result.strip().startswith("RELATED")


def retrieve_context(question):
    documents = st.session_state.vector_store.similarity_search(
        question,
        k=6
    )

    return "\n\n".join(
        f"Page {doc.metadata.get('page', 0) + 1}:\n{doc.page_content}"
        for doc in documents
    )


def generate_answer(question, context, history, llm):
    system_prompt = """
                    You are a helpful medical PDF assistant.

                    Your job is to answer questions using the uploaded PDF.

                    CONVERSATION MEMORY RULES:

                    1. Always consider the complete conversation history.
                    2. Remember patient names, diagnoses, symptoms,
                       medications, treatments and other subjects
                       mentioned earlier.
                    3. Understand references such as it, its, that disease,
                       this condition, the patient, that medication and
                       those symptoms using previous messages.
                    4. Never ask the user to repeat information that is
                       already available in conversation history.
                    5. Resolve follow-up questions using previous messages.

                    PATIENT INFORMATION RULES:

                    6. For questions about the patient's name, age, gender,
                       date of birth or identity, use the extracted patient
                       information returned by retriever_tool.
                    7. Do not confuse the patient's name with the doctor's,
                       relative's or hospital's name.
                    8. If patient information is unavailable, clearly say
                       that it was not found.
                    9. Never invent patient identification details.

                    RETRIEVAL RULES:

                    10. Always use retriever_tool before answering questions
                        about the uploaded PDF.
                    11. When the user asks a follow-up question, identify
                        the relevant subject from conversation history.
                    12. Convert incomplete follow-up questions into complete
                        search queries.
                    13. Use the retrieved PDF information to answer.
                    14. If information is not available in the PDF, clearly
                        distinguish general medical knowledge from
                        information documented in the PDF.
                    15. Never invent patient-specific medical details.

                    ANSWERING RULES:

                    16. Explain medical terminology in simple language.
                    17. Give clear and structured answers.
                    18. Distinguish documented patient information from
                        general medical explanations.
                    19. For potentially life-threatening conditions,
                        recommend prompt professional medical assessment.
                    20. Do not replace professional medical advice.
                    21. Never reply anything not in the context of given pdf.
                    """

    prompt = f"""
{system_prompt}

CONVERSATION HISTORY:
{history}

EXTRACTED PATIENT INFORMATION:
{st.session_state.patient_info}

USER QUESTION:
{question}

RETRIEVED PDF CONTENT:
{context}

Answer the user's question using the rules above.
"""

    response = llm.invoke(prompt)
    return get_response_text(response)


def process_question(question):
    llm = get_llm()
    history = get_history()

    standalone_question = rewrite_question(
        question,
        history,
        llm
    )

    is_related = check_relevance(
        standalone_question,
        history,
        st.session_state.report_text,
        llm
    )

    if not is_related:
        return (
            "I can only answer questions related to the uploaded medical "
            "report. Please ask something about the patient or the "
            "information documented in the PDF."
        )

    context = retrieve_context(standalone_question)

    if not context.strip():
        return (
            "I couldn't find enough information about this in the "
            "uploaded medical report."
        )

    return generate_answer(
        standalone_question,
        context,
        history,
        llm
    )


def reset_application():
    st.session_state.messages = []
    st.session_state.vector_store = None
    st.session_state.patient_info = None
    st.session_state.documents = []
    st.session_state.report_text = None
    st.session_state.processed_file = None
    st.session_state.uploader_version += 1


if not os.getenv("GROQ_API_KEY"):
    st.error("Please configure GROQ_API_KEY in your .env file.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "patient_info" not in st.session_state:
    st.session_state.patient_info = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "report_text" not in st.session_state:
    st.session_state.report_text = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


llm = get_llm()

st.title("🩺 MediScan AI")
st.caption("Medical Report Chatbot")

with st.sidebar:
    st.header("Medical Report")

    uploaded_file = st.file_uploader(
        "Upload your medical PDF",
        type=["pdf"],
        key=f"pdf_{st.session_state.uploader_version}"
    )

    if st.button("Start Again", use_container_width=True):
        reset_application()
        st.rerun()

    if st.session_state.vector_store is not None:
        st.success("Report uploaded successfully")

    st.divider()

    st.caption(
        "MediScan AI provides document-based information and does not "
        "replace professional medical advice."
    )


if uploaded_file is not None:
    file_signature = uploaded_file.getvalue()

    if file_signature != st.session_state.processed_file:
        temp_path = None

        try:
            with st.spinner("Reading medical report..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:
                    temp_file.write(file_signature)
                    temp_path = temp_file.name

                try:
                    documents = PyPDFLoader(temp_path).load()
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

                if not documents:
                    raise ValueError("No readable pages found in the PDF.")

                vector_store = create_vector_store(documents)

                patient_info = extract_patient_information(
                    vector_store,
                    llm
                )

                report_text = "\n\n".join(
                    doc.page_content for doc in documents
                )

                st.session_state.documents = documents
                st.session_state.vector_store = vector_store
                st.session_state.patient_info = patient_info
                st.session_state.report_text = report_text
                st.session_state.processed_file = file_signature
                st.session_state.messages = []

            st.success("Medical report processed successfully.")

        except Exception as error:
            st.error(f"Error processing PDF: {error}")
            st.stop()


if st.session_state.vector_store is not None:

    st.subheader("Patient Information")

    st.markdown(st.session_state.patient_info)

    st.divider()

    st.subheader("Chat with MediScan AI")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about the medical report..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = process_question(question)
                except Exception as error:
                    answer = f"An error occurred: {error}"

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info("Upload a medical PDF from the sidebar to get started.")
