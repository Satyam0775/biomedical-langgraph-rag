import json
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
import os

# Download punkt tokenizer (only first time)
nltk.download("punkt")

# Paths
INPUT_PATH = "data/processed/cleaned_pubmed.csv"
OUTPUT_PATH = "data/processed/semantic_chunks.json"

# Chunking configuration
CHUNK_SIZE = 5        # number of sentences per chunk
CHUNK_OVERLAP = 2     # overlapping sentences


def semantic_chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into semantically meaningful chunks
    using sentence boundaries.
    """
    sentences = sent_tokenize(text)

    chunks = []
    start = 0

    while start < len(sentences):
        end = start + chunk_size
        chunk_sentences = sentences[start:end]
        chunk_text = " ".join(chunk_sentences)

        if len(chunk_text.strip()) > 0:
            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    all_chunks = []
    chunk_id = 0

    for doc_id, row in df.iterrows():
        text = row["text"]
        chunks = semantic_chunk_text(text)

        for ch in chunks:
            all_chunks.append(
                {
                    "chunk_id": f"semantic_{chunk_id}",
                    "text": ch,
                    "source_doc_id": int(doc_id),
                    "chunk_type": "semantic"
                }
            )
            chunk_id += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Semantic chunking completed")
    print(f"📦 Total chunks created: {len(all_chunks)}")
    print(f"💾 Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
