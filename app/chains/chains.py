from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs) 

prompt_template = PromptTemplate.from_template("""
You are a financial analyst. Use the following SEC 10-K filing excerpts to answer the question. Be precise and cite relevant numbers. If you don't know the answer, just say that you don't know and provide suggestions to the related query that you can actually find answer to. For any tables required, give a proper table format.

Context:
{context}

Question:
{question}

Answer:
""")

def get_chain(retriever):
    return (
    {"context": retriever | format_docs, "question":RunnablePassthrough()}
     | prompt_template
     | llm
     | StrOutputParser()
)
