from flask import Flask, render_template, request
import pdfplumber
import requests
import os
import time

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it in your terminal/environment.")

# ✅ MANDATORY ROUTER API (2026 Standard)
SUMMARY_API = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

# ✅ GEMMA 2 (The one you accepted the license for)
QG_API = "https://router.huggingface.co/hf-inference/models/google/gemma-2-9b-it"

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
        print(f"PDF Extraction Error: {e}")
    return text

# -----------------------------
# SAFE API CALL (Handles 503 Waking Up)
# -----------------------------
def call_hf_api(url, payload):
    # Try up to 3 times in case the model is "Cold" (Loading)
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json(), None
            
            # If 503, the model is loading. Wait and retry.
            if response.status_code == 503:
                print(f"Model is loading... attempt {attempt + 1}/3")
                time.sleep(10)
                continue
                
            return None, f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return None, str(e)
            
    return None, "The AI model is taking too long to wake up. Please try again in 30 seconds."

# -----------------------------
# SUMMARY (BART)
# -----------------------------
def summarize_text(text):
    payload = {
        "inputs": text[:1200],
        "parameters": {"max_length": 150, "min_length": 40}
    }

    data, error = call_hf_api(SUMMARY_API, payload)

    if error or not data:
        return "Summary generation failed. The model might be offline."

    if isinstance(data, list) and "summary_text" in data[0]:
        return data[0]["summary_text"]

    return "Summary not available"

# -----------------------------
# QUESTION GENERATION (GEMMA 2)
# -----------------------------
def generate_questions(text, limit=5):
    # Prompt tuned for Gemma-2-9b-it
    prompt = (
        f"Context: {text[:1000]}\n\n"
        f"Task: Generate exactly {limit} high-quality exam revision questions based on the text above. "
        "Each question must end with a '?'. Do not provide answers. Do not include numbering. "
        "Output each question on a new line."
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    data, error = call_hf_api(QG_API, payload)

    if error:
        return [f"Error: {error}"]

    # Gemma returns a list of dicts
    output = ""
    if isinstance(data, list) and len(data) > 0:
        output = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        output = data.get("generated_text", "")

    # Clean the output string into a list of questions
    questions = []
    for line in output.split("\n"):
        line = line.strip()
        if "?" in line and len(line) > 15:
            # Strip common AI artifacts
            clean_q = line.lstrip("0123456789. -*•")
            questions.append(clean_q)

    return questions[:limit] if questions else ["Model could not generate formatted questions. Try another PDF."]

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
        return "Please upload a valid PDF file."

    text = extract_text(pdf)

    if not text.strip():
        return "Could not read any text from that PDF. It might be an image-only scan."

    summary = summarize_text(text)
    questions = generate_questions(text, limit=5)

    return render_template(
        "result.html",
        summary=summary,
        questions=questions
    )

@app.route("/health")
def health():
    return "Server is healthy"

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    # Standard port for local dev or Render/Railway/Heroku deployment
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)