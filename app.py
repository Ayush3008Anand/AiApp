from flask import Flask, render_template, request
import pdfplumber
import nltk
import random
import requests
import os

app = Flask(__name__)

# -----------------------------
# NLTK SETUP
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')


# -----------------------------
# HUGGINGFACE API CONFIG
# -----------------------------
API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"


headers = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"
}


# -----------------------------
# PDF TEXT EXTRACTION
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
# SUMMARIZATION
# -----------------------------
def summarize_text(text):
    payload = {"inputs": text[:1000]}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, list):
            return result[0]["summary_text"]
        else:
            return "❌ Error generating summary"
    except Exception as e:
        return f"❌ API Error: {str(e)}"


# -----------------------------
# MCQ GENERATION
# -----------------------------
def generate_mcqs(text, num_questions=5):
    sentences = nltk.sent_tokenize(text)
    mcqs = []

    for _ in range(num_questions):
        if len(sentences) < 3:
            break

        sentence = random.choice(sentences)
        words = sentence.split()

        if len(words) < 6:
            continue

        answer = random.choice(words)
        question = sentence.replace(answer, "_____")

        options = [answer]

        while len(options) < 4:
            rand_sentence = random.choice(sentences)
            rand_word = random.choice(rand_sentence.split())

            if rand_word not in options and len(rand_word) > 2:
                options.append(rand_word)

        random.shuffle(options)

        mcqs.append({
            "question": question,
            "options": options,
            "answer": answer
        })

    return mcqs


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        if "pdf" not in request.files:
            return "❌ No file uploaded"

        pdf = request.files["pdf"]

        if pdf.filename == "":
            return "❌ Empty file"

        text = extract_text(pdf)

        if len(text.strip()) == 0:
            return "❌ Could not extract text"

        summary = summarize_text(text)
        mcqs = generate_mcqs(summary)

        return render_template("result.html", summary=summary, mcqs=mcqs)

    except Exception as e:
        return f"🔥 Error: {str(e)}"


@app.route("/health")
def health():
    return "OK", 200


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)