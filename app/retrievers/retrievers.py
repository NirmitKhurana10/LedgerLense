from services.pinecone_client import vectorStore

def get_retriever(search_type='similarity'):

    '''
    Returns a retriever from vectorStore using the specified search strategy.
    Supported types:
      - 'similarity': Top-K similarity search
      - 'mmr': Maximal Marginal Relevance (diversified search)
    '''
     
    if search_type == "similarity":
        return vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
        )
    else:
        return vectorStore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,        # final number of docs to return
            "fetch_k": 20,  # initial pool size
            "lambda": 0.5   # mix relevance (0–1)
            }
        )
