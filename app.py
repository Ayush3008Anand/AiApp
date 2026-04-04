from flask import Flask, render_template, request
import pdfplumber
from transformers import pipeline
import nltk
import random
import os

# Download tokenizer (only once)
nltk.download("punkt")

app = Flask(__name__)

# -----------------------------
# LAZY LOAD MODEL (IMPORTANT)
# -----------------------------
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6"  # lighter + faster
        )
    return summarizer


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


# -----------------------------
# CHUNK TEXT
# -----------------------------
def split_text(text, max_words=400):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))

    return chunks


# -----------------------------
# SUMMARIZATION PIPELINE
# -----------------------------
def summarize_text(text):
    summarizer = get_summarizer()

    chunks = split_text(text)
    partial_summaries = []

    for chunk in chunks:
        try:
            out = summarizer(
                chunk,
                max_length=120,
                min_length=40,
                do_sample=False
            )
            partial_summaries.append(out[0]["summary_text"])
        except:
            continue

    combined = " ".join(partial_summaries)

    final_summary = summarizer(
        combined[:2000],
        max_length=150,
        min_length=50,
        do_sample=False
    )[0]["summary_text"]

    return final_summary


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

        # generate distractors
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
    pdf = request.files["pdf"]

    text = extract_text(pdf)

    if len(text.strip()) == 0:
        return "❌ Could not extract text from PDF"

    summary = summarize_text(text)
    mcqs = generate_mcqs(summary)

    return render_template("result.html", summary=summary, mcqs=mcqs)


# -----------------------------
# HEALTH CHECK (IMPORTANT)
# -----------------------------
@app.route("/health")
def health():
    return "OK", 200


# -----------------------------
# RUN APP (LOCAL ONLY)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)