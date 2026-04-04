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


def generate_mcqs(text, num_questions=5):
    sentences = nltk.sent_tokenize(text)
    mcqs = []

    if len(sentences) < 2:
        return []

    # build frequency map
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and len(w) > 3]

    from collections import Counter
    freq = Counter(words)

    attempts = 0

    while len(mcqs) < num_questions and attempts < 50:
        attempts += 1

        sentence = random.choice(sentences)
        sent_words = [w for w in sentence.split() if w.isalpha()]

        if len(sent_words) < 5:
            continue

        # pick most important word in sentence
        candidates = [(w, freq[w.lower()]) for w in sent_words if len(w) > 3]

        if not candidates:
            continue

        answer = max(candidates, key=lambda x: x[1])[0]

        # avoid very common garbage words
        if answer.lower() in ["the", "this", "that", "and", "for", "with"]:
            continue

        # SHORT QUESTION
        question = sentence.replace(answer, "_____")
        question = shorten_sentence(question, 10)

        # distractors from same text (context-based)
        distractors = list(set(words))
        distractors = [w for w in distractors if w.lower() != answer.lower()]
        random.shuffle(distractors)
        distractors = distractors[:3]

        # fallback safety
        while len(distractors) < 3:
            distractors.append(random.choice(words))

        options = distractors + [answer]
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