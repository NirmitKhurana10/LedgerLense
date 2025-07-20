from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from helpers.helpers import extract_clean_text_from_xbrl
from services.pinecone_client import vectorStore
from uuid import uuid4
import json
import os
import tempfile




def handle_uploaded_file(uploaded_files):
    output_file = "data/chunks/all_chunks.json"
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

    for file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(file.read())
            tmpfilePath = tmpfile.name

            if file.name.endswith(".txt"):
                cleaned_text = extract_clean_text_from_xbrl(tmpfilePath)

                doc = Document(
                page_content=cleaned_text,
                metadata={
                    "ticker": file.name.split(".")[0],
                    "source_file": tmpfilePath
                    }
                )

                chunks = splitter.split_documents([doc])
                all_chunks.extend(chunks)
                
            else:
                print("❌ Please upload only.txt files")


    
    # Save all chunks to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps({
                "text": chunk.page_content,
                "metadata": chunk.metadata
            }) + "\n")

    batch_size = 100

# Loop through all_chunks in batches
    for i in range(0, len(all_chunks), batch_size):
        chunk_batch = all_chunks[i:i + batch_size]
        uuid_batch = [str(uuid4()) for _ in chunk_batch]

        try:
            vectorStore.add_documents(documents=chunk_batch, ids=uuid_batch)
        except Exception as e:
            print(f"❌ Error uploading batch {i // batch_size + 1}: {e}")
    
    return all_chunks


# Step       Functionality
# 1️⃣         Accepts .txt file uploads (via uploaded_files)
# 2️⃣         Saves each file temporarily, cleans XBRL HTML using your extract_clean_text_from_xbrl()
# 3️⃣         Converts each cleaned file into a LangChain Document
# 4️⃣         Splits the document into overlapping chunks (using RecursiveCharacterTextSplitter)
# 5️⃣         Aggregates all chunks into a master list
# 6️⃣         Saves the chunks into a data/chunks/all_chunks.json file for future use
# 7️⃣         Uploads those chunks to Pinecone vector DB in batches (w/ UUIDs for traceability)


    