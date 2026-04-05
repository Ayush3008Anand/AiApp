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

SUMMARY_API = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"
QG_API = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
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
# SUMMARY (UNCHANGED BUT CLEANED)
# -----------------------------
def summarize_text(text):
    try:
        payload = {"inputs": text[:1200]}
        res = requests.post(SUMMARY_API, headers=headers, json=payload)
        data = res.json()

        if isinstance(data, list):
            return data[0]["summary_text"]

    except:
        pass

    return "Summary not available"


# -----------------------------
# HIGH-QUALITY QUESTION GENERATION (AI-BASED)
# -----------------------------
def generate_questions(text, limit=5):
    try:
        prompt = f"""
You are an expert teacher.

Read the text and generate {limit} important exam-style revision questions.

Rules:
- Only questions
- No answers
- No numbering explanation
- Each question should be meaningful and clear

Text:
{text[:1500]}

Output:
"""

        payload = {
            "inputs": prompt
        }

        response = requests.post(QG_API, headers=headers, json=payload)

        data = response.json()

        # DEBUG (VERY IMPORTANT)
        print("HF RESPONSE:", data)

        # -------------------------
        # CASE 1: Normal response
        # -------------------------
        if isinstance(data, list):
            output = data[0].get("generated_text", "")

        # -------------------------
        # CASE 2: Dict error response
        # -------------------------
        elif isinstance(data, dict) and "error" in data:
            return [f"API Error: {data['error']}"]

        else:
            return ["Unexpected model response"]

        # -------------------------
        # CLEAN OUTPUT
        # -------------------------
        questions = []

        for line in output.split("\n"):
            line = line.strip()

            if len(line) > 10 and "?" in line:
                questions.append(line)

        # fallback if no "?" format
        if not questions:
            questions = [q.strip("- ") for q in output.split("\n") if len(q.strip()) > 10]

        return questions[:limit] if questions else ["No questions generated"]

    except Exception as e:
        return [f"Error: {str(e)}"]


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
        return "Could not extract text"

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