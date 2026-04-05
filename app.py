from flask import Flask, render_template, request
import pdfplumber
import requests
import os
import time
from google import genai  # Latest 2026 SDK

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not HF_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing HF_TOKEN or GEMINI_API_KEY in environment variables")

# ✅ SUMMARY API (KEEPING UNCHANGED AS REQUESTED)
SUMMARY_API = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

# ✅ GEMINI CLIENT (NEW FOR QUESTIONS)
# Model: gemini-3-flash is the 2026 standard for speed/free tier
client = genai.Client(api_key=GEMINI_API_KEY)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# -----------------------------
# PDF EXTRACTION
# -----------------------------
def extract_text(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file.stream) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        print(f"Extraction Error: {e}")
    return text

# -----------------------------
# SUMMARY (UNCHANGED LOGIC)
# -----------------------------
def summarize_text(text):
    # Using BART via Hugging Face Router
    payload = {"inputs": text[:1200]}
    try:
        response = requests.post(SUMMARY_API, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data[0].get("summary_text", "Summary not available")
        return f"Summary Error {response.status_code}"
    except Exception as e:
        return f"Summary Exception: {str(e)}"

# -----------------------------
# QUESTION GENERATION (NEW GEMINI IMPLEMENTATION)
# -----------------------------
def generate_questions(text, limit=5):
    # We use Gemini here because it handles logic/formatting much better
    prompt = (
        f"Generate {limit} revision questions based on this text. "
        "Rules: Output ONLY the questions, one per line, ending with '?'. No numbering.\n\n"
        f"Text: {text[:10000]}"
    )
    
    try:
        # 2026 SDK Syntax
        response = client.models.generate_content(
            model="gemini-3-flash", 
            contents=prompt
        )
        
        # Split by newline and filter for valid lines
        raw_lines = response.text.strip().split('\n')
        questions = [line.strip() for line in raw_lines if '?' in line and len(line) > 10]
        
        return questions[:limit] if questions else ["No questions generated"]
    except Exception as e:
        return [f"Gemini API Error: {str(e)}"]

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    pdf = request.files.get("pdf")
    if not pdf or pdf.filename == "":
        return "No file uploaded"

    text = extract_text(pdf)
    if not text.strip():
        return "Could not extract text from PDF"

    # Call both AI functions
    summary = summarize_text(text)
    questions = generate_questions(text, limit=5)

    return render_template(
        "result.html",
        summary=summary,
        questions=questions
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)