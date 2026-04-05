from flask import Flask, render_template, request
import pdfplumber
import nltk
import requests
import os

app = Flask(__name__)

# -----------------------------
# SETUP
# -----------------------------
nltk.download('punkt')

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

# ✅ STABLE ROUTER MODELS (CONFIRMED WORKING)
SUMMARY_API = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
QG_API = "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# -----------------------------
# PDF EXTRACTION
# -----------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file.stream) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


# -----------------------------
# SAFE API CALL
# -----------------------------
def call_hf_api(url, payload):
    try:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}: {response.text}"

        try:
            return response.json(), None
        except:
            return None, "Invalid JSON response"

    except Exception as e:
        return None, str(e)


# -----------------------------
# SUMMARY
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
# QUESTION GENERATION (FIXED)
# -----------------------------
def generate_questions(text, limit=5):

    prompt = f"""
Generate {limit} high-quality exam-level revision questions from the text below.

Rules:
- Only questions
- Each must end with ?
- No answers
- No explanation

Text:
{text[:1500]}
"""

    payload = {
        "inputs": prompt
    }

    data, error = call_hf_api(QG_API, payload)

    if error or not data:
        return [f"Error: {error}"]

    # -----------------------------
    # Extract output safely
    # -----------------------------
    output = data[0].get("generated_text", "")

    # -----------------------------
    # Clean questions
    # -----------------------------
    questions = []

    for line in output.split("\n"):
        line = line.strip()

        if len(line) > 10:
            line = line.lstrip("-•1234567890. ")

            if "?" in line:
                questions.append(line)

    # fallback
    if not questions:
        questions = [
            q.strip("- ").strip()
            for q in output.split("\n")
            if len(q.strip()) > 10
        ]

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