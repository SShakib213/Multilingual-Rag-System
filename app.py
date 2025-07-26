from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from bangla_rag_improved import BanglaRAGChain  # Use improved version
import uvicorn
import logging
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multilingual RAG System (Local)",
    description="A RAG system using local Transformers models for Bengali and English",
    version="2.0.0"
)

# Global variable for RAG system
rag = None
initialization_time = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag, initialization_time
    
    start_time = time.time()
    
    try:
        logger.info("🚀 Initializing improved Bengali RAG system...")
        
        rag = BanglaRAGChain()  # Use improved class
        success = rag.load_quick()  # Use quick load method
        
        if success:
            initialization_time = time.time() - start_time
            logger.info(f"✅ RAG system initialized successfully in {initialization_time:.2f} seconds")
        else:
            logger.error("❌ Failed to initialize RAG system")
            rag = None
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        rag = None

class QueryRequest(BaseModel):
    question: str
    include_sources: bool = False
    include_stats: bool = False

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list = None
    conversation_history: list = None
    stats: dict = None
    processing_time: float = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multilingual RAG System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 10px 0; }
            input[type="text"] { width: 70%; padding: 10px; margin: 5px; }
            button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; }
            .answer { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .ready { background: #d4edda; color: #155724; }
            .not-ready { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>🤖 Multilingual RAG System (Local)</h1>
        <div class="container">
            <h3>System Status</h3>
            <div id="status" class="status">Checking...</div>
            <button onclick="checkStatus()">Refresh Status</button>
        </div>
        
        <div class="container">
            <h3>Ask a Question</h3>
            <input type="text" id="question" placeholder="Enter your question in Bengali or English..." />
            <button onclick="askQuestion()">Ask</button>
            <div id="answer" class="answer" style="display:none;"></div>
        </div>

        <script>
            async function checkStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    const statusDiv = document.getElementById('status');
                    
                    if (data.status === 'healthy') {
                        statusDiv.className = 'status ready';
                        statusDiv.innerHTML = `
                            ✅ System Ready<br>
                            Device: ${data.device}<br>
                            Models: ${data.models_loaded ? 'Loaded' : 'Loading...'}<br>
                            Initialization Time: ${data.initialization_time || 'N/A'}s
                        `;
                    } else {
                        statusDiv.className = 'status not-ready';
                        statusDiv.innerHTML = '❌ System Not Ready - Check logs';
                    }
                } catch (error) {
                    document.getElementById('status').innerHTML = '❌ Error checking status';
                }
            }

            async function askQuestion() {
                const question = document.getElementById('question').value;
                if (!question.trim()) return;

                const answerDiv = document.getElementById('answer');
                answerDiv.style.display = 'block';
                answerDiv.innerHTML = '🤔 Thinking...';

                try {
                    const response = await fetch('/query/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            question: question, 
                            include_sources: true,
                            include_stats: true 
                        })
                    });

                    const data = await response.json();
                    answerDiv.innerHTML = `
                        <strong>Q:</strong> ${data.question}<br><br>
                        <strong>A:</strong> ${data.answer}<br><br>
                        <small>Processing time: ${data.processing_time}s</small>
                    `;
                } catch (error) {
                    answerDiv.innerHTML = '❌ Error: ' + error.message;
                }
            }

            // Check status on load
            checkStatus();
            
            // Allow Enter key to submit
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') askQuestion();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag:
        stats = rag.get_stats()
        return {
            "status": "healthy",
            "device": stats["device"],
            "models_loaded": stats["qa_chain_ready"],
            "initialization_time": f"{initialization_time:.2f}" if initialization_time else None,
            **stats
        }
    else:
        return {
            "status": "unhealthy",
            "device": "unknown",
            "models_loaded": False,
            "initialization_time": None
        }

@app.post("/query/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not rag:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized. Check logs for setup issues."
        )
    
    start_time = time.time()
    
    try:
        answer, context = rag.get_response(request.question)  # Use improved method
        processing_time = time.time() - start_time
        
        response = QueryResponse(
            question=request.question,
            answer=answer,
            processing_time=round(processing_time, 2)
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-form/")
async def query_rag_form(question: str = Form(...)):
    """Form-based endpoint"""
    request = QueryRequest(question=question, include_sources=True)
    return await query_rag(request)

if __name__ == "__main__":
    print("🚀 Starting Multilingual RAG System with Local Models")
    print("📦 First run may take time to download models...")
    print("🌐 Web interface will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
