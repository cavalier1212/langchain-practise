import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


if __name__ == "__main__":
    print("hi")
    pdf_path = os.getenv("PDF_PATH")
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    # OpenAI embedding models.
    embeddings = OpenAIEmbeddings()
    # Return VectorStore initialized from documents and embeddings.
    vectorstore = FAISS.from_documents(docs, embeddings)
    # 在本地存取
    vectorstore.save_local("faiss_index_react")
    # 本地反序列化
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    """
    retrieval_qa_chat_prompt
    ---
    SYSTEM
        Answer any use questions based solely on the context below:
    
    <context>
        {context}
    </context>
    
    PLACEHOLDER
        chat_history
    HUMAN
        {input}
    """
    # 組成 chain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        ChatOpenAI(model="gpt-4o"), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
