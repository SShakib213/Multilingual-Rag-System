import os
import fitz  # PyMuPDF
import pdfplumber
import pymupdf4llm

def extract_direct_text(pdf_path, output_path):
    """Try direct PDF text extraction first (fastest)"""
    print("⚡ Trying direct PDF text extraction...")
    
    try:
        doc = fitz.open(pdf_path)
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                all_text.append(text.strip())
        
        combined_text = "\n\n".join(all_text)
        bengali_chars = sum(1 for c in combined_text if '\u0980' <= c <= '\u09FF')
        
        if bengali_chars > 100:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            print(f"✅ Direct extraction: {bengali_chars} Bengali chars")
            return True
        else:
            print("❌ Direct extraction failed - need OCR")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def extract_with_pdfplumber(pdf_path, output_path):
    """Extract text using pdfplumber"""
    print("🔧 Using pdfplumber for Bengali extraction...")
    
    try:
        all_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text(
                    x_tolerance=3,
                    y_tolerance=3,
                    layout=True,
                    x_density=7.25,
                    y_density=13
                )
                
                if text and text.strip():
                    all_text.append(text.strip())
        
        combined_text = "\n\n".join(all_text)
        bengali_chars = sum(1 for c in combined_text if '\u0980' <= c <= '\u09FF')
        
        if bengali_chars > 50:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            print(f"✅ pdfplumber: {bengali_chars} Bengali chars")
            return True
        else:
            print("❌ pdfplumber failed")
            return False
            
    except Exception as e:
        print(f"❌ pdfplumber error: {e}")
        return False

def extract_with_pymupdf4llm(pdf_path, output_path):
    """Extract with pymupdf4llm"""
    print("🔧 Using pymupdf4llm...")
    
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        if md_text and md_text.strip():
            cleaned_text = md_text.replace('#', '').replace('*', '').replace('_', '')
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            bengali_chars = sum(1 for c in cleaned_text if '\u0980' <= c <= '\u09FF')
            print(f"✅ pymupdf4llm: {bengali_chars} Bengali chars")
            return True
        else:
            print("❌ pymupdf4llm failed")
            return False
            
    except Exception as e:
        print(f"❌ pymupdf4llm error: {e}")
        return False

def find_best_extraction(pdf_path):
    """Try multiple extraction methods and pick the best one"""
    
    print("🔍 Finding best extraction method for Bengali PDF...")
    
    methods = [
        ("Direct PyMuPDF", "output/direct_test.txt", extract_direct_text),
        ("PDFPlumber", "output/pdfplumber_test.txt", extract_with_pdfplumber),
        ("PyMuPDF4LLM", "output/pymupdf4llm_test.txt", extract_with_pymupdf4llm),
    ]
    
    best_method = None
    best_score = 0
    best_file = None
    
    for method_name, output_file, extract_func in methods:
        print(f"\n🧪 Testing {method_name}...")
        
        try:
            success = extract_func(pdf_path, output_file)
            
            if success and os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Score based on Bengali content and readability
                bengali_chars = sum(1 for c in content if '\u0980' <= c <= '\u09FF')
                total_chars = len(content)
                bengali_ratio = bengali_chars / max(total_chars, 1)
                
                # Simple readability check
                words = content.split()
                avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
                
                score = bengali_chars * bengali_ratio * (1 if 2 < avg_word_length < 15 else 0.5)
                
                print(f"📊 {method_name} Score: {score:.1f} (Bengali chars: {bengali_chars})")
                
                if score > best_score:
                    best_score = score
                    best_method = method_name
                    best_file = output_file
            
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
    
    if best_method:
        print(f"\n🏆 Best method: {best_method}")
        print(f"📁 Best file: {best_file}")
        
        # Copy best result to final output
        final_output = "output/best_bengali.txt"
        if os.path.exists(best_file):
            with open(best_file, 'r', encoding='utf-8') as src:
                content = src.read()
            with open(final_output, 'w', encoding='utf-8') as dst:
                dst.write(content)
            
            print(f"✅ Final output saved to: {final_output}")
            print(f"📄 Sample:\n{content[:400]}...")
            
            return final_output
    else:
        print("❌ All extraction methods failed!")
        return None

if __name__ == "__main__":
    pdf_file = "data/HSC26-Bangla1st-Paper.pdf"
    best_file = find_best_extraction(pdf_file)
    
    if best_file:
        print(f"\n🎉 Use this file for your RAG: {best_file}")
