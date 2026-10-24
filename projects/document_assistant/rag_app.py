#!/usr/bin/env python3
"""
Document Question-Answering RAG Assistant
Answers user questions strictly from company_knowledge_base.json with source citations.
"""

import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    kb_path = "datasets/company_knowledge_base.json"
    if not os.path.exists(kb_path):
        kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform([f"{d['title']} {d['content']}" for d in docs]).toarray()

    for q in ["What is the refund policy?", "When is delivery free?"]:
        qv = vec.transform([q]).toarray()[0]
        sims = mat @ qv
        best = int(np.argmax(sims))
        print(f"Q: {q}
A: {docs[best]['content']} [{docs[best]['doc_id']}]
")

if __name__ == "__main__":
    main()
