from flask import Flask, render_template, request
import nltk
import re
import os
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

app = Flask(__name__)
app.name = "AI_TEXT_Summarizer"

# Download required resources once
nltk.download('punkt_tab')
nltk.download('stopwords')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text")

        if not text or len(text.split()) < 10:
            return render_template(
                "index.html",
                summary="Please enter a longer text for summarization."
            )

        clean_text = re.sub(r'\[[0-9]*\]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text)

        sentences = sent_tokenize(clean_text)

        stop_words = set(stopwords.words('english'))
        words = word_tokenize(clean_text.lower())

        filtered_words = [
            word for word in words if word.isalnum() and word not in stop_words
        ]

        word_frequencies = {}
        for word in filtered_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

        max_frequency = max(word_frequencies.values())

        for word in word_frequencies:
            word_frequencies[word] /= max_frequency

        sentence_score = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies and len(sentence.split()) < 30:
                    sentence_score[sentence] = sentence_score.get(
                        sentence, 0
                    ) + word_frequencies[word]

        if len(sentence_score) == 0:
            Summary = "Unable to generate summary. Please enter more text."
        else:
            Summary_Sentence = heapq.nlargest(
                2, sentence_score, key=sentence_score.get
            )
            Summary = " ".join(Summary_Sentence)

        print("SUMMARY GENERATED:", Summary)

        return render_template("index.html",summary=Summary,text=text)

    except Exception as e:
        return f"Error: {e}"
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0',debug=False,port=port)