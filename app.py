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
nltk.download('punkt')
nltk.download('punkt_tab')


# -----------------------------
# HUGGINGFACE API CONFIG
# -----------------------------
API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"


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
    payload = {
        "inputs": text[:1000]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        print("Status Code:", response.status_code)
        print("Response:", response.text)

        result = response.json()

        # ✅ Case 1: Model loading
        if isinstance(result, dict) and "error" in result:
            return f"⚠️ Model loading / API error: {result['error']}"

        # ✅ Case 2: Success
        if isinstance(result, list):
            return result[0]["summary_text"]

        return "❌ Unexpected response format"

    except Exception as e:
        return f"❌ API Error: {str(e)}"
# -----------------------------
# MCQ GENERATION
# -----------------------------
def shorten_sentence(sentence, max_words=10):
    words = sentence.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


from collections import Counter
import random
import nltk

def shorten_sentence(sentence, max_words=12):
    words = sentence.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def get_key_terms(text):
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and len(w) > 3]
    return Counter(words)


def generate_mcqs(text, num_questions=5):
    sentences = nltk.sent_tokenize(text)

    if len(sentences) < 2:
        return []

    freq = get_key_terms(text)
    mcqs = []
    attempts = 0

    while len(mcqs) < num_questions and attempts < 60:
        attempts += 1

        sentence = random.choice(sentences)
        words = nltk.word_tokenize(sentence)

        words = [w for w in words if w.isalpha() and len(w) > 3]

        if len(words) < 6:
            continue

        # -----------------------------
        # PICK BEST ANSWER (NOT RANDOM)
        # -----------------------------
        scored = [(w, freq[w.lower()]) for w in words]
        scored.sort(key=lambda x: x[1], reverse=True)

        answer = scored[0][0]

        # skip useless words
        if answer.lower() in ["this", "that", "there", "their", "which", "would"]:
            continue

        # -----------------------------
        # SHORT QUESTION (EXAM STYLE)
        # -----------------------------
        question = sentence.replace(answer, "_____")
        question = shorten_sentence(question, 14)

        # -----------------------------
        # SMART DISTRACTORS
        # -----------------------------
        pool = list(freq.keys())

        distractors = [
            w for w in pool
            if w.lower() != answer.lower()
            and w.isalpha()
            and len(w) > 3
        ]

        # prefer similar frequency words (more realistic MCQs)
        distractors = sorted(distractors, key=lambda w: abs(freq[w] - freq[answer.lower()]))

        options = distractors[:3]

        while len(options) < 3:
            options.append(random.choice(pool))

        options = options[:3] + [answer]
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
        mcqs = generate_mcqs(text)

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