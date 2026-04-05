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
    raise ValueError("HF_TOKEN not found in environment variables")

# ✅ Summary remains on the Router (Works fine for BART)
SUMMARY_API = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

# ✅ QG switched to Direct Inference API to avoid 404
# Using a highly reliable specialized QG model
QG_API = "https://api-inference.huggingface.co/models/mrm8488/t5-base-finetuned-question-generation-ap"

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
# SAFE API CALL
# -----------------------------
def call_hf_api(url, payload):
    try:
        # Standard request
        response = requests.post(url, headers=headers, json=payload)

        # Handle 503 (Model Loading) - Common on Direct API
        if response.status_code == 503:
            return None, "Model is currently loading on Hugging Face. Please refresh in 20 seconds."

        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}: {response.text}"

        return response.json(), None

    except Exception as e:
        return None, str(e)


# -----------------------------
# SUMMARY (UNCHANGED LOGIC)
# -----------------------------
def summarize_text(text):
    payload = {
        "inputs": text[:1200]
    }

    data, error = call_hf_api(SUMMARY_API, payload)

    if error or not data:
        return "Summary not available"

    if isinstance(data, list) and "summary_text" in data[0]:
        return data[0]["summary_text"]

    return "Summary not available"


# -----------------------------
# QUESTION GENERATION (UPDATED FOR T5)
# -----------------------------
def generate_questions(text, limit=5):
    # Specialized QG models expect 'context: ' prefix
    prompt = f"context: {text[:1000]}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.6,
            "do_sample": True
        }
    }

    data, error = call_hf_api(QG_API, payload)

    if error:
        return [f"Error: {error}"]
    
    if not data or not isinstance(data, list):
        return ["No questions generated"]

    # T5 models usually output one string with questions separated by '?'
    output = data[0].get("generated_text", "")
    
    # Split the output into individual questions
    raw_questions = output.split("?")
    questions = []

    for q in raw_questions:
        q = q.strip()
        if len(q) > 10:
            # Clean up any leftover numbering or bullet points
            clean_q = q.lstrip("-•1234567890. ")
            questions.append(clean_q + "?")

    return questions[:limit] if questions else ["No questions generated"]


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

    summary = summarize_text(text)
    questions = generate_questions(text, limit=5)

    return render_template(
        "result.html",
        summary=summary,
        questions=questions
    )


@app.route("/health")
def health():
    return "OK"


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)