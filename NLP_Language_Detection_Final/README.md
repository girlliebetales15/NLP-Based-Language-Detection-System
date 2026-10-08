# NLP-Based Language Detection System

A Flask web application for the NLP mini-project and practical experiments. It includes language detection, text preprocessing, morphology, N-grams, POS tagging, chunking, NER, TF-IDF similarity, WSD, analytics and CSV batch detection.

## Run in VS Code (Windows)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

The first run may download the required NLTK data packages. Internet access is needed for that first download if the packages are not already installed.

## NER test sentence

`Prime Minister Narendra Modi visited Mumbai on 5 September 2026 to inaugurate a new technology centre developed by Tata Group.`

The application first tries NLTK's named-entity chunker. If the NLTK entity model is unavailable, it uses a small deterministic fallback for common demo entities so the page does not silently fail.
