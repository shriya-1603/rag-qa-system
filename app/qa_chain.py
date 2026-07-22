import os
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from config import OLLAMA_BASE_URL, LLM_MODEL, TOP_K

PROMPT_TEMPLATE = """You are a helpful assistant designed to answer questions about the provided documents.

Guidelines:
1. Thoroughly analyze the Context below.
2. Answer the Question directly and clearly using the information in the Context.
3. If the answer is not explicitly stated but there is related information in the context, synthesize a helpful response using that related information and explain the connection.
4. If the context does not contain the answer at all, explain what topics are covered in the context and provide a helpful answer using your general knowledge, while clearly stating that the information is from your general knowledge and not the uploaded documents.
5. Provide detailed context and explanations to make the answer easy to understand for someone who has not read the documents.

Context:
{context}

Question: {question}

Answer:"""


def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=api_key,
            temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama
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
