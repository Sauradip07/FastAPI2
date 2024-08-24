from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

class RAGApp:
    def __init__(self):
        # Initialize LLM and other components
        self.llm = Ollama(model="mistral")
        self.embeddings_llm = OllamaEmbeddings(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter()

        # Load documents
        urls = [
            "https://indiankanoon.org/doc/1218090/",
            "https://restthecase.com/knowledge-bank/small-business-laws-in-india-every-business-owner-should-know",
            "https://www.india.gov.in/information-central-board-direct-taxes",
            "https://www.nextias.com/blog/constitution-of-india/",
            "https://en.wikipedia.org/wiki/Constitution_of_India",
            "https://loksabhadocs.nic.in/Refinput/Research_notes/English/04122019_153433_1021204140.pdf",
            "https://cjp.org.in/personal-laws-vis-a-vis-fundamental-rights-part-iii-of-the-constitution/",
            "https://www.lawaudience.com/personal-laws-and-fundamental-rights/",
            "https://www.indiafilings.com/learn/essential-legal-documents-for-startup/",
            "https://carajput.com/blog/legal-documents-required-for-running-business/",
            "https://www.archives.gov/files/about/laws/basic-laws-book-2016.pdf",
            "https://socialsciences.exeter.ac.uk/media/universityofexeter/schoolofhumanitiesandsocialsciences/law/pdfs/The_Common_Law_in_India.pdf",
            "https://dopt.gov.in/sites/default/files/Revised_AIS_Rule_Vol_I_Rule_01.pdf",
        ]

        docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())

        documents = self.text_splitter.split_documents(docs)
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
        relevant_docs = self.retriever.get_relevant_documents(question)  # Use appropriate method
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        response = self.document_chain.run({
            "input": question,
            "context": [Document(page_content=context)]
        })
        
        return response

# Initialize RAGApp once
rag_app = RAGApp()