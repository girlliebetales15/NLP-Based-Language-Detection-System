from flask import Flask, render_template, request, jsonify, Response
import csv, io, re
from collections import Counter
from datetime import datetime

import nltk
from langdetect import detect, detect_langs, DetectorFactory, LangDetectException
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DetectorFactory.seed = 0
app = Flask(__name__)

LANGUAGES = {
    "en":"English", "hi":"Hindi", "mr":"Marathi", "fr":"French", "es":"Spanish",
    "de":"German", "bn":"Bengali", "ta":"Tamil", "te":"Telugu", "ja":"Japanese",
    "zh-cn":"Chinese", "zh-tw":"Chinese (Traditional)", "ar":"Arabic", "ru":"Russian",
    "pt":"Portuguese", "it":"Italian", "ko":"Korean", "gu":"Gujarati", "pa":"Punjabi",
    "kn":"Kannada", "ml":"Malayalam", "ur":"Urdu", "ne":"Nepali", "tr":"Turkish"
}

# NLTK resources used by the experiments. They are checked/downloaded at startup.
RESOURCES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "chunkers/maxent_ne_chunker_tab": "maxent_ne_chunker_tab",
    "corpora/words": "words",
}
for path, package in RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
HISTORY = []


def safe_tokens(text):
    try:
        return word_tokenize(text)
    except Exception:
        return re.findall(r"\b\w+[\w'-]*\b|[^\w\s]", text, flags=re.UNICODE)


def safe_sentences(text):
    if not text.strip():
        return []
    try:
        return sent_tokenize(text)
    except Exception:
        return re.split(r"(?<=[.!?])\s+", text.strip())


def language_result(text, add_history=True):
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "Please enter some text."}
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    if len(words) < 2 or len(text) < 4:
        return {"ok": False, "error": "Please enter a longer sentence or paragraph for reliable language detection."}
    try:
        predictions = detect_langs(text)
        top = [{"code": p.lang, "language": LANGUAGES.get(p.lang, "Other / Unlisted Language"),
                "probability": round(p.prob * 100, 2)} for p in predictions[:5]]
        code = predictions[0].lang
        confidence = top[0]["probability"] if top else 0
        result = {
            "ok": True, "text": text, "code": code,
            "language": LANGUAGES.get(code, "Other / Unlisted Language"),
            "confidence": confidence, "predictions": top,
            "words": len(words), "characters": len(text), "sentences": len(safe_sentences(text))
        }
        if add_history:
            HISTORY.insert(0, {"text": text[:80], "language": result["language"], "code": code,
                               "confidence": confidence, "time": datetime.now().strftime("%H:%M:%S")})
            del HISTORY[50:]
        return result
    except LangDetectException:
        return {"ok": False, "error": "Unable to identify the language. Please enter a longer, meaningful text."}
    except Exception as exc:
        return {"ok": False, "error": f"Language detection error: {exc}"}


def fallback_entities(text):
    """Small deterministic fallback so the NER page still returns useful entities if NLTK data is unavailable."""
    entities = []
    # Common entity cues in the practical/demo sentences.
    patterns = [
        (r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}", "PERSON"),
        (r"\b(?:Narendra Modi|Mahatma Gandhi|Amit Shah|Sachin Tendulkar)\b", "PERSON"),
        (r"\b(?:Mumbai|Delhi|Pune|Nashik|London|Paris|New York|India|Maharashtra)\b", "GPE"),
        (r"\b(?:Tata Group|Google|Microsoft|Apple|Amazon|OpenAI)\b", "ORGANIZATION"),
        (r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", "DATE"),
    ]
    for pattern, label in patterns:
        for m in re.finditer(pattern, text):
            entities.append({"text": m.group(0), "label": label})
    # Preserve text order and remove duplicates.
    unique = {(e["text"], e["label"]): e for e in entities}
    return sorted(unique.values(), key=lambda e: text.find(e["text"]))


def extract_entities(text):
    try:
        tagged = pos_tag(safe_tokens(text))
        tree = ne_chunk(tagged, binary=False)
        entities = []
        for node in tree:
            if isinstance(node, Tree):
                label = node.label()
                entity_text = " ".join(w for w, _ in node.leaves())
                entities.append({"text": entity_text, "label": label})
        if entities:
            return entities, "NLTK named-entity chunker"
    except Exception:
        pass
    fallback = fallback_entities(text)
    return fallback, "rule-based fallback"


@app.route("/")
def index(): return render_template("index.html")
@app.route("/experiments")
def experiments(): return render_template("experiments.html")
@app.route("/language")
def language_page(): return render_template("language.html")
@app.route("/analytics")
def analytics(): return render_template("analytics.html")

TOOLS = {
    "preprocessing": ("Text Preprocessing", "Tokenization, filtration, stop-word removal, stemming and lemmatization."),
    "morphology": ("Morphological Analysis & Word Generation", "Analyze words and generate basic forms."),
    "ngrams": ("Unigram • Bigram • Trigram", "Generate N-Gram sequences and frequencies."),
    "pos": ("Part-of-Speech Tagging", "Identify grammatical roles of words."),
    "chunking": ("Chunking", "Extract noun phrases using POS tags and grammar."),
    "ner": ("Named Entity Recognition", "Detect people, places, organizations and dates."),
    "similarity": ("Text Similarity", "Compare two texts using TF-IDF and cosine similarity."),
    "wsd": ("Word Sense Disambiguation", "Distinguish the financial and river meanings of bank."),
}

for slug in TOOLS:
    endpoint = f"/{slug}"
    def make_view(s=slug):
        return lambda: render_template("tool.html", tool=s, title=TOOLS[s][0], desc=TOOLS[s][1])
    app.add_url_rule(endpoint, endpoint.strip('/'), make_view(slug))


@app.post("/api/detect")
def api_detect():
    data = request.get_json(silent=True) or {}
    return jsonify(language_result(data.get("text", "")))

@app.get("/api/history")
def api_history(): return jsonify(HISTORY)

@app.post("/api/history/clear")
def clear_history():
    HISTORY.clear(); return jsonify({"ok": True})

@app.post("/api/preprocess")
def api_preprocess():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    tokens = safe_tokens(text)
    filtered = [x for x in tokens if x.isalpha()]
    try: sw = set(stopwords.words("english"))
    except Exception: sw = set("the is am are a an and or of to in on for with".split())
    important = [x for x in filtered if x.lower() not in sw]
    stems = [stemmer.stem(x) for x in important]
    lemmas = []
    for x in important:
        try: lemmas.append(lemmatizer.lemmatize(x.lower()))
        except Exception: lemmas.append(x.lower())
    return jsonify({"tokens": tokens, "filtered": filtered, "stopwords_removed": important,
                    "stems": stems, "lemmas": lemmas, "sentence_count": len(safe_sentences(text))})

@app.post("/api/morphology")
def api_morphology():
    data = request.get_json(silent=True) or {}; text = data.get("text", ""); root = data.get("root", "").strip()
    analysis = [{"word": w, "stem": stemmer.stem(w)} for w in safe_tokens(text) if w.isalpha()]
    generated = [root, root + "s", root + "ed", root + "ing"] if root else []
    return jsonify({"analysis": analysis, "generated": generated})

@app.post("/api/ngrams")
def api_ngrams():
    text = (request.get_json(silent=True) or {}).get("text", "")
    words = [w.lower() for w in safe_tokens(text) if w.isalpha()]
    def grams(n): return Counter(" ".join(words[i:i+n]) for i in range(max(0, len(words)-n+1)))
    return jsonify({"unigram": grams(1).most_common(20), "bigram": grams(2).most_common(20), "trigram": grams(3).most_common(20)})

@app.post("/api/pos")
def api_pos():
    text = (request.get_json(silent=True) or {}).get("text", "")
    try: tagged = pos_tag(safe_tokens(text))
    except Exception as exc: return jsonify({"error": f"POS tagging resource unavailable: {exc}"}), 500
    return jsonify({"tags": [{"word": w, "tag": t} for w, t in tagged]})

@app.post("/api/chunk")
def api_chunk():
    text = (request.get_json(silent=True) or {}).get("text", "")
    try:
        tagged = pos_tag(safe_tokens(text))
        parser = nltk.RegexpParser(r"NP: {<DT>?<JJ>*<NN.*>+}")
        tree = parser.parse(tagged)
        chunks = [" ".join(w for w, _ in subtree.leaves()) for subtree in tree.subtrees(lambda t: t.label() == "NP")]
    except Exception as exc: return jsonify({"error": f"Chunking error: {exc}"}), 500
    return jsonify({"chunks": chunks, "tagged": [{"word": w, "tag": t} for w, t in tagged]})

@app.post("/api/ner")
def api_ner():
    data = request.get_json(silent=True) or {}; text = data.get("text", "").strip()
    if not text: return jsonify({"error": "Please enter text containing names, places, organizations or dates."}), 400
    entities, method = extract_entities(text)
    return jsonify({"entities": entities, "method": method})

@app.post("/api/similarity")
def api_similarity():
    data = request.get_json(silent=True) or {}; a = data.get("text_a", "").strip(); b = data.get("text_b", "").strip()
    if not a or not b: return jsonify({"error": "Enter both texts."}), 400
    vec = TfidfVectorizer().fit_transform([a, b])
    score = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
    return jsonify({"score": round(score * 100, 2)})

@app.post("/api/wsd")
def api_wsd():
    text = (request.get_json(silent=True) or {}).get("text", "").lower()
    financial = any(k in text for k in ["money", "loan", "bank account", "withdraw", "deposit", "financial", "cash", "credit"])
    river = any(k in text for k in ["river", "fisherman", "shore", "water", "fishing", "stream"])
    if financial and not river: sense, confidence = "Financial Institution", 94
    elif river and not financial: sense, confidence = "River Bank", 94
    elif financial and river: sense, confidence = "Ambiguous Context", 55
    else: sense, confidence = "Insufficient Context", 30
    return jsonify({"sense": sense, "confidence": confidence})

@app.post("/api/batch")
def api_batch():
    file = request.files.get("file")
    if not file: return jsonify({"error": "Please upload a CSV file."}), 400
    try:
        content = file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames: return jsonify({"error": "CSV has no columns."}), 400
        field = reader.fieldnames[0]
        rows = []
        for row in reader:
            text = row.get(field, "") or ""
            result = language_result(text, add_history=False)
            rows.append({"text": text, "language": result.get("language", "Unable to identify"),
                         "code": result.get("code", "-"), "confidence": result.get("confidence", 0)})
        return jsonify({"rows": rows, "count": len(rows), "column": field})
    except Exception as exc: return jsonify({"error": str(exc)}), 400

@app.get("/api/demo-analytics")
def demo_analytics():
    return jsonify({"languages": [
        {"language":"English","count":42},{"language":"Hindi","count":28},
        {"language":"Marathi","count":17},{"language":"French","count":8},{"language":"Spanish","count":5}
    ]})

if __name__ == "__main__":
    app.run(debug=True)
