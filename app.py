import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# --- Setup page ---
st.set_page_config(page_title="Crop Advisor", page_icon="🌾", layout="centered")

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Crop data ---
crop_docs = {
    "Cassava": [
        Document(page_content="Cassava is a root crop grown widely in Nigeria.", metadata={"source": "AgroDoc1"}),
        Document(page_content="Cassava can be processed into garri, fufu, and other foods.", metadata={"source": "AgroDoc2"}),
        Document(page_content="Cassava grows well in well-drained sandy-loam soils.", metadata={"source": "AgroDoc3"})
    ],
    "Yam": [
        Document(page_content="Yam is a staple food in West Africa.", metadata={"source": "AgroDoc1"}),
        Document(page_content="Yam can be boiled, roasted, or pounded into pounded yam.", metadata={"source": "AgroDoc2"}),
        Document(page_content="Yam is grown in mounds or ridges to allow tuber expansion.", metadata={"source": "AgroDoc3"})
    ],
    "Maize": [
        Document(page_content="Maize is a cereal crop grown across Nigeria.", metadata={"source": "AgroDoc1"}),
        Document(page_content="Maize can be processed into pap, a popular Nigerian breakfast meal.", metadata={"source": "AgroDoc2"}),
        Document(page_content="Maize thrives in fertile, well-drained soils with good sunlight.", metadata={"source": "AgroDoc3"})
    ]
}

# --- Crop summaries ---
crop_summaries = {
    "Cassava": "Cassava is a root crop widely cultivated in Nigeria. It can be processed into garri, fufu, and starch, and grows well in sandy-loam soils.",
    "Yam": "Yam is a staple food in West Africa, eaten boiled, roasted, or pounded. It grows best in mounds or ridges with loose soil.",
    "Maize": "Maize is a versatile cereal crop grown across Nigeria, used for food and animal feed. It thrives in fertile soils with good sunlight."
}

# --- Build FAISS indexes ---
faiss_indexes = {crop: FAISS.from_documents(docs, embeddings) for crop, docs in crop_docs.items()}

# --- Intro message ---
st.markdown("## 👋 Hello! I’m your Crop Advisor bot")
st.markdown("Select a crop below and ask me anything about it.")

# --- Crop selection ---
crop_choice = st.selectbox("🌱 Choose a crop", list(crop_docs.keys()), index=None, placeholder="Select a crop...")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = None

# --- When crop is selected ---
if crop_choice:
    st.session_state.selected_crop = crop_choice
    st.markdown(f"### 📄 {crop_choice} Summary")
    st.write(crop_summaries[crop_choice])

    # --- Chat interface ---
    st.markdown("---")
    st.markdown("### 💬 Ask me anything about this crop")

    # Show previous conversation
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**👤 You:** {text}")
        else:
            st.markdown(f"**🤖 Bot:** {text}")

    # Question input
    user_question = st.text_input("Type your question here...", key="question_input")

    if user_question:
        # Store user question
        st.session_state.chat_history.append(("user", user_question))

        # Retrieve answer from FAISS
        faiss_index = faiss_indexes[crop_choice]
        result = faiss_index.similarity_search(user_question, k=1)

        if result:
            answer = result[0].page_content
        else:
            answer = "Sorry, I don’t have information on that right now. Please check back later."

        # Store bot answer
        st.session_state.chat_history.append(("bot", answer))

        # Clear input
        st.experimental_rerun()
