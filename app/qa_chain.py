from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from config import OLLAMA_BASE_URL, LLM_MODEL, TOP_K

PROMPT_TEMPLATE = """Use the following context to answer the question. If the answer is not
in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
