from rag_pipeline import BanglaRAG
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import json

def calculate_similarity_score(expected, generated):
    """Calculate semantic similarity between expected and generated answers"""
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    expected_embedding = model.encode([expected])
    generated_embedding = model.encode([generated])
    
    similarity = cosine_similarity(expected_embedding, generated_embedding)[0][0]
    return similarity

def evaluate_groundedness(answer, retrieved_docs):
    """Check if answer is grounded in retrieved documents"""
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    answer_embedding = model.encode([answer])
    doc_embeddings = model.encode([doc.page_content for doc in retrieved_docs])
    
    similarities = cosine_similarity(answer_embedding, doc_embeddings)[0]
    max_similarity = np.max(similarities)
    
    return max_similarity

def run_comprehensive_eval():
    """Run comprehensive evaluation of the RAG system"""
    test_cases = [
        {
            "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "expected": "শুম্ভুনাথ",
            "language": "Bengali"
        },
        {
            "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "expected": "মামাকে",
            "language": "Bengali"
        },
        {
            "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "expected": "১৫ বছর",
            "language": "Bengali"
        }
    ]

    rag = BanglaRAG()
    rag.build_vectorstore("output/cleaned_text.txt")

    results = []
    total_relevance = 0
    total_groundedness = 0
    total_similarity = 0

    print("=== RAG System Evaluation ===\n")

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected = test_case["expected"]
        
        # Retrieve documents
        docs = rag.retrieve(question)
        
        # Generate answer
        answer = rag.generate_answer(question)
        
        # Calculate metrics
        similarity_score = calculate_similarity_score(expected, answer)
        groundedness_score = evaluate_groundedness(answer, docs)
        relevance_score = int(expected.lower() in answer.lower())
        
        # Store results
        result = {
            "test_case": i,
            "question": question,
            "expected": expected,
            "generated": answer,
            "relevance_score": relevance_score,
            "similarity_score": round(similarity_score, 3),
            "groundedness_score": round(groundedness_score, 3),
            "language": test_case["language"]
        }
        results.append(result)
        
        # Accumulate totals
        total_relevance += relevance_score
        total_similarity += similarity_score
        total_groundedness += groundedness_score
        
        # Print individual results
        print(f"Test Case {i}:")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print(f"Generated: {answer}")
        print(f"Relevance: {relevance_score}/1")
        print(f"Similarity: {similarity_score:.3f}")
        print(f"Groundedness: {groundedness_score:.3f}")
        print("-" * 50)

    # Calculate averages
    num_tests = len(test_cases)
    avg_relevance = total_relevance / num_tests
    avg_similarity = total_similarity / num_tests
    avg_groundedness = total_groundedness / num_tests

    print("\n=== Overall Results ===")
    print(f"Average Relevance: {avg_relevance:.3f}")
    print(f"Average Similarity: {avg_similarity:.3f}")
    print(f"Average Groundedness: {avg_groundedness:.3f}")
    
    # Save results to file
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "individual_results": results,
            "summary": {
                "avg_relevance": avg_relevance,
                "avg_similarity": avg_similarity,
                "avg_groundedness": avg_groundedness,
                "total_test_cases": num_tests
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to evaluation_results.json")

if __name__ == "__main__":
    run_comprehensive_eval()
