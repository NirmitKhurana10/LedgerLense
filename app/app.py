import streamlit as st
from services.document_uploader import handle_uploaded_file
from chains.chains import get_chain
from retrievers.retrievers import get_retriever


st.set_page_config(page_title="LedgerLens - 10-K RAG", layout="wide")
st.title("📊 LedgerLens - SEC 10-K Insights")

st.sidebar.header("📂 Upload SEC 10-K Files")
uploaded_files = st.sidebar.file_uploader("Upload only .txt 10-K files", type=["txt"], accept_multiple_files=True)

search_method = st.sidebar.radio("Search Type", options=["similarity", "mmr"])
retriever = get_retriever(search_type=search_method)
rag_chain = get_chain(retriever)


if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        chunks = handle_uploaded_file(uploaded_files)
        st.sidebar.success(f"✅ Uploaded and stored {len(uploaded_files)} files!")

st.header("🔍 Your personal Financial Guide")
user_query = st.text_input("Ask any question related to 10-K Files for AMZN, TSLA, MSFT, GOOGL, AAPL, META or any you have uploaded...")

if user_query:
    with st.spinner("Generating response..."):
        response = rag_chain.invoke(user_query)
        st.subheader("📄 Answer")
        st.write(response)



