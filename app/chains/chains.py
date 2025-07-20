from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from templates.prompt_template import prompt_template

llm = llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs) 

def get_chain(retriever):
    return (
    {"context": retriever | format_docs, "question":RunnablePassthrough()}
     | prompt_template
     | llm
     | StrOutputParser()
)


# ✅ Why Use RunnablePassthrough?

# You need to send both:
# 	•	context: built by the retriever from the query
# 	•	question: the raw query text itself

# So instead of splitting it manually, LangChain lets you build a dictionary input pipeline using:

# RunnablePassthrough() just returns the input as-is, and it’s used here to make sure your original question reaches the prompt template unchanged.

# This is a clean, declarative way to say:

# “Use the same input in two places: one goes through the retriever and becomes context, and the other just flows through untouched as question.”

# LangChain internally maps this like:

# Key      :     Value
# contex   :     The result of running the query through retriever_flat + format_docs
# question :     The raw string you passed: "What were Amazon's total assets in 2023?"