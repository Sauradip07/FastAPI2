import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import random
from typing import List

class RAGApp:
    def __init__(self, urls: List[str] = None):
        # Initialize LLM and other components
        self.llm = Ollama(model="mistral")
        self.embeddings_llm = OllamaEmbeddings(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter()
        
        self.urls = urls or []
        self.documents = []
        self.load_documents()
        
    def load_documents(self):
        # Load documents from URLs
        for url in self.urls:
            loader = WebBaseLoader(url)
            self.documents.extend(loader.load())
        
        # Load PDFs from /data folder
        pdf_folder = "/data"
        if os.path.exists(pdf_folder):
            for filename in os.listdir(pdf_folder):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(pdf_folder, filename)
                    loader = PyPDFLoader(pdf_path)
                    self.documents.extend(loader.load())
        
        documents = self.text_splitter.split_documents(self.documents)
        self.vector_index = FAISS.from_documents(documents, self.embeddings_llm)
        self.retriever = self.vector_index.as_retriever()
        self.prompt = ChatPromptTemplate.from_template("""
        Please respond to the following inquiry by prioritizing the provided context, while also incorporating relevant knowledge you possess. 
        If the context does not offer sufficient clarity or the subject matter is beyond your expertise, clearly state that 
        you are not informed on this specific topic and unable to provide a definitive answer.
        <context>
        {context}
        </context>
        Question: {input}
        """)
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)

    def get_answer(self, question: str) -> str:
        relevant_docs = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        response = self.document_chain.run({
            "input": question,
            "context": [Document(page_content=context)]
        })
        
        return response

    def generate_synthetic_data(self, num_samples: int = 100):
        # Generate synthetic question-answer pairs
        synthetic_data = []
        for _ in range(num_samples):
            question = self.generate_random_question()
            answer = self.get_answer(question)
            synthetic_data.append({"question": question, "answer": answer})
        return synthetic_data

    def generate_random_question(self):
        # Generate a random legal question (this is a simple example, you may want to expand this)
        topics = ["constitution", "business law", "personal law", "tax law", "criminal law"]
        question_starters = ["What is", "How does", "Can you explain", "What are the implications of", "Is it legal to"]
        return f"{random.choice(question_starters)} {random.choice(topics)}?"

    def fine_tune(self, synthetic_data):
        # Fine-tune the model using synthetic data
        # Note: This is a placeholder. The actual implementation will depend on the specific fine-tuning method you choose.
        print("Fine-tuning the model with synthetic data...")
        # Implement your fine-tuning logic here
        print("Fine-tuning complete.")

    def add_url(self, url: str):
        self.urls.append(url)
        loader = WebBaseLoader(url)
        new_docs = loader.load()
        self.documents.extend(new_docs)
        self.update_vector_index(new_docs)

    def update_vector_index(self, new_docs):
        split_docs = self.text_splitter.split_documents(new_docs)
        self.vector_index.add_documents(split_docs)

def upload_pdf(file_content: bytes, filename: str):
    # Ensure the /data directory exists
    os.makedirs("/data", exist_ok=True)
    
    # Save the file to the /data directory
    destination_path = os.path.join("/data", filename)
    with open(destination_path, "wb") as dest_file:
        dest_file.write(file_content)
    
    print(f"PDF uploaded successfully: {destination_path}")
    return destination_path

# Initialize RAGApp (without hard-coded URLs)
rag_app = RAGApp()

# Example usage:
# rag_app.add_url("https://example.com/some-legal-document")
# uploaded_pdf_path = upload_pdf(file_content_from_frontend, "document.pdf")
# rag_app.load_documents()  # Reload documents to include the new PDF