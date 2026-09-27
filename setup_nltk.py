"""Download the NLTK data that crawler.py and analysis.py need (run once)."""
import nltk

for package in ["stopwords", "wordnet", "omw-1.4",
                "averaged_perceptron_tagger_eng", "punkt_tab"]:
    nltk.download(package)
