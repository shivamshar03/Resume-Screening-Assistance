
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS


#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# iterate over files in 
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


#Function to push data to Vector Store - Pinecone here
def push_to_faiss(embedding_model, docs, faiss_index_path = "./faiss_index"):
    # Create FAISS index from documents
    faiss_index = FAISS.from_documents(docs, embedding_model)

    # Save index to local disk
    faiss_index.save_local(faiss_index_path)

    return faiss_index



# Pull data from existing FAISS index on disk
def pull_from_faiss(embedding_model, faiss_index_path = "./faiss_index"):
    faiss_index = FAISS.load_local(faiss_index_path, embeddings=embedding_model,allow_dangerous_deserialization=True)

    return faiss_index



#Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,k,embeddings,unique_id):
    index = pull_from_faiss(embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    return similar_docs


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = ChatGroq(model_name="llama3-8b-8192",temperature=0.9)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    print("load_summarize_chain.name :",chain.name,"load_summarize_chain  ", chain)
    summary = chain.run([current_doc])

    return summary




    