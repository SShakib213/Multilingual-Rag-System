# Bengali RAG System

## Setup Guide
bash
pip install -r requirements.txt
# Install Ollama and pull models:
ollama pull llama3.2:1b
ollama pull nomic-embed-text
python bangla_rag_improved.py
python app.py  # For web interface


## Tools & Libraries
- **PDF Extraction**: PyMuPDF, pdfplumber, pymupdf4llm
- **RAG Framework**: LangChain, ChromaDB
- **Embeddings**: Bengali SBERT, Ollama embeddings
- **LLM**: Ollama (llama3.2:1b)
- **Web**: FastAPI

## Sample Queries
Q: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
A: ১. i, ii।

Q: : কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
A: (১) ৮৩। অপর্ির্চতাগইনিজা- ৬। 'তক'। ৭। নির্দেশ, সার্থক (এ. -ও) ৮। (ধ) ৫। হিউরিয়ন (iii). 'আপনা-অনু'। (iv) ৬। 'ইসার্থ, দারি'- (vi) এ. -ও


## API Documentation
- `POST /query/` - Submit questions
- `GET /health` - System status

## Answers to Questions

**1 What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**
Used multiple methods (PyMuPDF, pdfplumber, pymupdf4llm) and selected best based on Bengali character count. Faced severe formating issues. Couldnt properly extract the pdf into txt and thus got gibberish txt file.

**2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**
I chose a hybrid character-based chunking strategy with Bengali-aware separators using LangChain's RecursiveCharacterTextSplitter. This works well because, it is aware of bengali sperators, the key innovation is using "।" (Bengali full stop) as a primary separator, which respects Bengali sentence boundaries better than English punctuation. The RecursiveCharacterTextSplitter tries to split at paragraph breaks (\n\n) first, then sentences (।, .), then words ( ), maintaining semantic coherence.


**3 What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**
I used l3cube-pune/bengali-sentence-similarity-sbert as the primary embedding model with Ollama nomic-embed-text as fallback. I chose the Bengali SBERT model because it's specifically trained on Bengali text for semantic similarity tasks, making it superior to multilingual models for understanding Bengali grammar, syntax, and cultural context. The model captures meaning through transformer-based contextual embeddings that create dense vector representations at the sentence level, enabling semantic matching even when different words are used to express similar concepts. With normalized embeddings and cosine similarity, it effectively retrieves relevant Bengali text chunks based on semantic understanding rather than just keyword matching, which is crucial for literary text comprehension.

**4 How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**
I store all the text chunks as vector embeddings in ChromaDB, which is like a smart database for finding similar content. When you ask a question, I convert your question into the same type of vector using the Bengali embedding model, then ChromaDB finds the chunks that are most similar to your question using cosine similarity - basically measuring how "close" the meanings are in vector space. I chose cosine similarity because it's great at understanding semantic relationships rather than just matching exact words, and ChromaDB because it's simple to use, works offline, and integrates well with LangChain. The system retrieves the top few most relevant chunks and feeds them to the language model to generate an answer, so you get responses based on the most contextually appropriate parts of the text.

**5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**
I use Bengali-specific embeddings that understand meaning, not just exact words, so if you ask about "কল্যাণীর বয়স" it can find chunks about Kalyani's age even if worded differently. For vague questions like "এটা কী?" the system gets confused because vague queries don't match well with specific content - it either returns irrelevant chunks or says "আমি জানি না" when nothing seems relevant enough. The system works best with clear, specific questions about actual story content.

**6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?**
Unfortunately no, the results seems completely irrelevant. The main problem according to me is the text extraction and chunking. I found it difficult to find better text extraction method and thus failed to implement this project properly. So, got irrelevent answer because of poor text extraction. 
