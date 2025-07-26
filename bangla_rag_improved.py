import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
    BitsAndBytesConfig,
)
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from rich import print as rprint
from rich.panel import Panel
from tqdm import tqdm
import warnings
import re
import shutil

warnings.filterwarnings("ignore")

CACHE_DIR = "./models"

class BanglaRAGChain:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_model_id = "llama3.2:1b"  
        self.embed_model_id = "nomic-embed-text" 
        self.k = 3
        self.max_new_tokens = 256
        self.chunk_size = 500
        self.chunk_overlap = 100
        self.text_path = "output/best_bengali.txt"
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        
        self._llm = None
        self._retriever = None
        self._db = None
        self._documents = []
        self._chain = None

    def load_quick(self):
      
        rprint(Panel("[bold green]Quick loading Bengali RAG with Ollama...", expand=False))
        
        try:
            # Create documents
            self._create_document()
            
            # Create vector DB with Ollama embeddings
            self._update_chroma_db()
            
            # Initialize components
            self._get_retriever()
            self._get_llm()
            self._create_chain()
            
            rprint(Panel("Bengali RAG with Ollama loaded successfully!", expand=False))
            return True
            
        except Exception as e:
            rprint(Panel(f"Loading failed: {e}", expand=False))
            return False

    def _create_document(self):
      
        try:
            rprint(Panel("Processing Bengali text...", expand=False))
            
            with open(self.text_path, "r", encoding="utf-8") as file:
                text_content = file.read()
            
            
            character_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "।", ".", "!", "?", " "],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            
            self._documents = character_splitter.split_text(text_content)
            
            rprint(Panel(f"Created {len(self._documents)} chunks", expand=False))
            
        except Exception as e:
            rprint(Panel(f"Document creation failed: {e}", expand=False))
            raise

    def _update_chroma_db(self):
        """Create vector database with Bengali-optimized embeddings"""
        try:
            rprint(Panel("[bold green]Creating embeddings with Bengali SBERT...", expand=False))
            
           
            db_path = "chroma_bangla_db"
            if os.path.exists(db_path):
                rprint(Panel("[bold yellow]Clearing existing database for dimension compatibility...", expand=False))
                shutil.rmtree(db_path)
            
            try:
                # Try Bengali-optimized embeddings first
                embeddings = HuggingFaceEmbeddings(
                    model_name="l3cube-pune/bengali-sentence-similarity-sbert",
                    model_kwargs={"device": self._device},
                    encode_kwargs={"normalize_embeddings": True}
                )
                rprint(Panel("Using Bengali SBERT embeddings!", expand=False))
            except Exception as e:
                rprint(Panel(f"Bengali SBERT failed: {e}", expand=False))
                rprint(Panel("Falling back to Ollama embeddings...", expand=False))
                
                # Fallback to Ollama embeddings
                embeddings = OllamaEmbeddings(
                    model=self.embed_model_id,
                    base_url="http://localhost:11434"
                )
                rprint(Panel("Using Ollama embeddings as fallback!", expand=False))
            
            self._db = Chroma.from_texts(
                texts=self._documents, 
                embedding=embeddings,
                persist_directory=db_path
            )
            
            rprint(Panel("Vector database created!", expand=False))
            
        except Exception as e:
            rprint(Panel(f"Vector DB failed: {e}", expand=False))
            raise

    def _get_retriever(self):
        """Initialize retriever"""
        self._retriever = self._db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": self.k}
        )
        rprint(Panel("Retriever ready!", expand=False))

    def _get_llm(self):
        """Initialize Ollama LLM"""
        try:
            rprint(Panel("Loading Ollama LLM...", expand=False))
            
            # Try different Ollama models
            models_to_try = [
                "llama3.2:1b",
                "gemma2:2b", 
                "llama3.2:3b"
            ]
            
            for model_name in models_to_try:
                try:
                    rprint(Panel(f"[bold blue]Trying {model_name}...", expand=False))
                    
                    self._llm = Ollama(
                        model=model_name,
                        base_url="http://localhost:11434",
                        temperature=self.temperature,
                        num_predict=self.max_new_tokens,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repeat_penalty=1.2
                    )
                    
                    # Test the model
                    test_response = self._llm("Hello")
                    if test_response:
                        rprint(Panel(f"Using {model_name}!", expand=False))
                        return
                        
                except Exception as model_error:
                    rprint(Panel(f"{model_name} failed: {model_error}", expand=False))
                    continue
            
            # If all Ollama models fail, fallback to HuggingFace
            rprint(Panel("All Ollama models failed, using HuggingFace fallback...", expand=False))
            self._load_huggingface_fallback()
            
        except Exception as e:
            rprint(Panel(f"LLM initialization failed: {e}", expand=False))
            raise

    def _load_huggingface_fallback(self):
        """Fallback to HuggingFace if Ollama fails"""
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
        
        model_id = "microsoft/DialoGPT-medium"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir="./models",
        )
        
        config = GenerationConfig(
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            generation_config=config,
        )
        
        self._llm = HuggingFacePipeline(pipeline=pipe)
        rprint(Panel("HuggingFace fallback ready!", expand=False))

    def _create_chain(self):
        """Create the RAG chain"""
        try:
            rprint(Panel("Creating RAG chain...", expand=False))
            
            # Improved Bengali-aware prompt template
            template = """আপনি একজন সহায়ক AI সহায়ক। দেওয়া প্রসঙ্গের ভিত্তিতে প্রশ্নের সংক্ষিপ্ত ও সঠিক উত্তর দিন। যদি প্রসঙ্গে উত্তর না থাকে, তাহলে "আমি জানি না" বলুন।

প্রসঙ্গ: {context}

প্রশ্ন: {question}

সংক্ষিপ্ত উত্তর:"""
            
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            
           
            def format_docs(docs):
                if not docs:
                    return "কোন প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"
                return "\n\n".join([doc.page_content[:200] for doc in docs])  # Limit context length
            
            self._chain = (
                {
                    "context": self._retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self._llm
                | StrOutputParser()
            )
            
            rprint(Panel("RAG chain created!", expand=False))
            
        except Exception as e:
            rprint(Panel(f"Chain creation failed: {e}", expand=False))
            raise

    def get_stats(self):
        """Get system statistics for health check"""
        return {
            "device": str(self._device),
            "qa_chain_ready": self._chain is not None,
            "documents_loaded": len(self._documents),
            "retriever_ready": self._retriever is not None,
            "llm_ready": self._llm is not None
        }

    def get_response(self, question):
        """Get response from the RAG chain"""
        if not self._chain:
            raise Exception("RAG chain not initialized")
        
        try:
            # Use the chain to get response
            answer = self._chain.invoke(question)
            
            # Clean up the answer
            if isinstance(answer, str):
                answer = answer.strip()
                # Remove common prefixes
                prefixes_to_remove = ["উত্তর:", "সংক্ষিপ্ত উত্তর:", "Answer:", "Response:"]
                for prefix in prefixes_to_remove:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()
                        break
                
                # Limit answer length to avoid rambling
                if len(answer) > 300:
                    sentences = answer.split('।')
                    answer = sentences[0] + '।' if sentences else answer[:200]
            else:
                answer = str(answer)
            
            # Get context from retriever
            context = []
            if self._retriever:
                try:
                    docs = self._retriever.get_relevant_documents(question)
                    context = [doc.page_content for doc in docs]
                except Exception as e:
                    rprint(Panel(f"Context retrieval warning: {e}", expand=False))
                    context = []
            
            return answer, context
            
        except Exception as e:
            rprint(Panel(f"[red]Error getting response: {e}", expand=False))
            return f"দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে।", []

# Test the system
if __name__ == "__main__":
    rag = BanglaRAGChain()
    
    if rag.load_quick():
        print("\n" + "="*50)
        print("Bengali RAG System Ready!")
        print("="*50)
        
        # Test query
        query = "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
        answer, context = rag.get_response(query)
        
        print(f"প্রশ্ন: {query}")
        print(f"উত্তর: {answer}")
        print(f"প্রসঙ্গ: {context[:200]}...")









