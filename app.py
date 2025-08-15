import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="🌾 AgroScan Chatbot", page_icon="🌾")

# --- TITLE / INTRO ---
st.title("🌾 Agro RAG Chatbot")
st.write("👋 Hello! I’m your Crop Advisor bot. Select a crop below and ask me anything about it.")

# --- API KEY ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- LLM (Groq API) ---
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# --- EMBEDDINGS ---
embeddings = HuggingFaceEmbeddings()

# --- KNOWLEDGE BASE ---
knowledge_texts = {
    "Cassava": """Cassava is a major root crop widely cultivated in Nigeria for its starchy tubers, which serve as a staple food and industrial raw material. 
Climate & Soil Requirements: Prefers warm, humid tropical climates (25–29°C), sandy-loam soils (pH 5.5–6.5), rainfall 1,000–1,500 mm.
Planting: Stem cuttings (20–25 cm) on ridges/mounds, spacing 1m × 1m, weed regularly.
Pests/Diseases: Cassava mealybug, green mite, cassava mosaic disease, bacterial blight.
Harvest: 9–12 months after planting, process tubers within 48 hours.
Market: Processed into garri, fufu, starch, ethanol, and feed. Nigeria is world’s largest producer.""",

    "Yam": """Yam is a staple food crop in Nigeria, culturally significant and economically valuable. Nigeria produces over 70% of the world’s yams.
Climate & Soil Requirements: Tropical climates (25–30°C), deep loamy soils rich in organic matter, rainfall 1,200–1,500 mm.
Planting: Tuber setts or small whole tubers, spacing 1m × 1m or 1.2m × 1.2m, use stakes for vines.
Pests/Diseases: Yam beetles, nematodes, anthracnose, yam mosaic virus, tuber rots.
Harvest: 8–12 months after planting, store in ventilated yam barns.
Market: Consumed boiled, pounded, fried, roasted, also exported fresh/processed.""",

    "Maize": """Maize is an important cereal crop in Nigeria for food, livestock feed, and industry. Grown nationwide, adaptable to many climates.
Climate & Soil Requirements: Thrives at 18–27°C, fertile well-drained soils (pH 5.5–7.0), rainfall 500–1,200 mm depending on variety.
Planting: Direct seeding, spacing 75 cm × 25 cm, apply NPK fertilizer, weed early.
Pests/Diseases: Stem borers, armyworms, maize streak virus, rust, leaf blight.
Harvest: Dry husks, hard kernels; dry grains to 12–13% moisture.
Market: Consumed fresh or processed (pap, flour, feed), used in breweries and food industries."""
}

# --- CONVERT KNOWLEDGE TO DOCUMENTS AND FAISS ---
faiss_dict = {}
for crop, text in knowledge_texts.items():
    docs = [Document(page_content=text, metadata={"source": f"{crop}_KB"})]
    faiss_dict[crop] = FAISS.from_documents(docs, embeddings)

# --- PROMPT TEMPLATE ---
prompt_template = """
Use the following pieces of context to answer the question.
If you don’t know the answer from the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- CHAT HISTORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "input_box" not in st.session_state:
    st.session_state.input_box = ""

# --- CROP SELECTION ---
selected_crop = st.selectbox("Select a crop:", list(knowledge_texts.keys()))
st.markdown(f"**Short Summary of {selected_crop}:**")
st.write(knowledge_texts[selected_crop].split("\n")[0])  # first line only

# --- RETRIEVAL QA CHAIN FOR SELECTED CROP ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_dict[selected_crop].as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# --- USER INPUT ---
st.text_input("💬 Ask a question:", key="input_box")

# --- SEND BUTTON ---
if st.button("Send"):
    user_input = st.session_state.input_box
    if user_input.strip() != "":
        with st.spinner("Thinking..."):
            answer = qa_chain.run(user_input)
        
        # Insert user question at the top
        st.session_state.chat_history.insert(0, ("User", user_input))
        
        # Insert bot greeting first if not greeted yet
        if not st.session_state.greeted:
            greeting = "👋 Hello! I’m your Crop Advisor bot. How can I help you today?"
            st.session_state.chat_history.insert(0, ("Bot", greeting))
            st.session_state.greeted = True

        # Insert bot answer right after greeting
        st.session_state.chat_history.insert(0, ("Bot", answer))
        
        # Clear input box
        st.session_state.input_box = ""

# --- DISPLAY CHAT HISTORY (newest at top, grows downward) ---
for speaker, message in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"<div style='background-color:#D1E7DD;padding:8px;border-radius:8px;margin-bottom:5px'><b>User:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#F8D7DA;padding:8px;border-radius:8px;margin-bottom:5px'><b>Bot:</b> {message}</div>", unsafe_allow_html=True)
