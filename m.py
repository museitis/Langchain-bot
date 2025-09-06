import os
import warnings
import time
import glob
import json
from pypdf import PdfReader
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv
import redis
import numpy as np


class Chatbot:
    
    def __init__(self):
        warnings.filterwarnings("ignore")
        load_dotenv()

        MODEL = "llama2"
        self.model = Ollama(model=MODEL)
        
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        self.cache_file = "cache.json"
        self.load_cache()
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Initialize the chain directly to prevent errors
        self.chain = self.get_conversational_chain()

    def load_cache(self):
        # First, set a default empty cache
        self.cache = {}
        # Check if the file exists and is not empty
        if os.path.exists(self.cache_file) and os.path.getsize(self.cache_file) > 0:
            with open(self.cache_file, 'r') as f:
                try:
                    # Try to load the JSON data
                    self.cache = json.load(f)
                except json.JSONDecodeError:
                    # If the file is corrupted, default to an empty cache
                    print("Warning: cache.json is corrupted. Starting with a new cache.")
                    self.cache = {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def list_files_in_folder(self, folder_path):
        return glob.glob(os.path.join(folder_path, '*.pdf'))

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    # Ensure text is not None before concatenating
                    text += page.extract_text() or ""
            except Exception as e:
                print(f"Error reading {pdf}: {e}")
        return text

    def get_text_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(text)

    def get_vectorstore(self, text_chunks):
        vectors = self.embeddings.embed_documents(text_chunks)
        for i, (chunk, vector) in enumerate(zip(text_chunks, vectors)):
            # The 'vector' variable is already a list, so we remove .tolist()
            vector_json = json.dumps(vector) 
            
            # Use Redis Hashes to store related data together
            self.redis_client.hset(f"doc:{i}", mapping={
                "text": chunk,
                "vector": vector_json
            })
        return len(vectors)

    def get_conversational_chain(self):
        template = """
        Answer the question based on the context below. If you can't
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        return load_qa_chain(self.model, chain_type="stuff", prompt=prompt)

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve_similar_documents(self, user_query, top_k=5):
        # NOTE: This is a brute-force search and is very inefficient for large datasets.
        # For production, use Redis's built-in vector search capabilities (e.g., HNSW index).
        print("Retrieving similar documents...")
        query_vector = self.embeddings.embed_query(user_query)
        
        similarities = []
        # Scan for document hash keys
        for key in self.redis_client.scan_iter("doc:*"):
            stored_vector_json = self.redis_client.hget(key, "vector")
            if stored_vector_json:
                # Convert stored JSON list back to NumPy array
                stored_vector = np.array(json.loads(stored_vector_json))
                similarity = self.cosine_similarity(query_vector, stored_vector)
                similarities.append((similarity, key))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        docs = []
        for _, key in similarities[:top_k]:
            text = self.redis_client.hget(key, "text").decode('utf-8')
            docs.append(Document(page_content=text))
            
        return docs

    def user_input(self, user_question, use_cache):
        query_vector = self.embeddings.embed_query(user_question)
        
        # Corrected Cache Logic
        if use_cache and self.cache: 
            for question, cache_item in self.cache.items():
                if 'vector' in cache_item:
                    cached_vector = np.array(cache_item['vector'])
                    similarity = self.cosine_similarity(np.array(query_vector), cached_vector)
                    if similarity > 0.95:
                        print(f"Found similar question in cache (similarity: {similarity:.2f}). Using cached response.")
                        return cache_item['response']
            
        # Document retrieval for every new question
        start_time = time.time()
        similar_docs = self.retrieve_similar_documents(user_question)
        end_time = time.time()
        print(f"Elapsed Time in document retrieval: {end_time - start_time:.4f} seconds")
        
        # The chain is now guaranteed to be a valid object
        response = self.chain.invoke(
            {"input_documents": similar_docs, "question": user_question}
        )

        # CORRECTED LINE: Extract only the text from the response before caching
        final_answer = response.get('output_text', '')
        self.cache[user_question] = {'response': final_answer, 'vector': query_vector}
        self.save_cache()
        
        return response # Return the full response object for immediate use

    def upload_docs(self):
        # Use a relative path for the documents folder
        folder_path = 'pdf_file' 
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' not found. Please create it and add your PDF files.")
            return

        files = self.list_files_in_folder(folder_path)
        if not files:
            print(f"No PDF files found in '{folder_path}'.")
            return
            
        print(f"Found files: {files}")
        text = self.get_pdf_text(files) 
        print("Extracted text from PDFs.")
        
        chunks = self.get_text_chunks(text)
        print(f"Created {len(chunks)} text chunks.")

        num_vectors = self.get_vectorstore(chunks)
        print(f"Stored {num_vectors} vectors into Redis.")

    def main(self):
        # Check if Redis has documents, if not, ingest them.
        if not self.redis_client.keys('doc:*'):
            print("No documents found in Redis. Starting the ingestion process...")
            self.upload_docs()

        while True:
            user_question = input("\nHEY!!, I am Wise Bud, Please enter your question (or type 'exit'): ")
            if user_question.lower() == "exit":
                break
            
            cache_input = input("Enable Cache? (Y/N, default N): ")
            use_cache = cache_input.lower() == 'y'

            if user_question:
                start_time = time.time()
                response = self.user_input(user_question, use_cache)
                
                # Use .get() for safe dictionary access
                answer = response.get('output_text', 'Sorry, I could not generate a response.').strip()
                
                print(f"\nAnswer: {answer}")
                end_time = time.time()
                print(f"Elapsed Time for response generation: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.main()