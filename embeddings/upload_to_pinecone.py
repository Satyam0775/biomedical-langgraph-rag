from embeddings.embed_chunks import embed_chunks
from vector_store.pinecone_client import get_pinecone_index


def main():
    print("🔗 Connecting to Pinecone...")
    index = get_pinecone_index()

    print("📥 Loading embeddings from agentic chunks...")
    ids, embeddings, metadata = embed_chunks()

    print(f"📦 Total vectors to upload: {len(ids)}")

    # Prepare vectors
    vectors = []
    for i in range(len(ids)):
        vectors.append({
            "id": ids[i],
            "values": embeddings[i].tolist(),
            "metadata": metadata[i]
        })

    # Upload in batches (safe for large data)
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i:i + BATCH_SIZE])
        print(f"⬆️ Uploaded {i + BATCH_SIZE} / {len(vectors)}")

    print("✅ All embeddings uploaded to Pinecone successfully!")


if __name__ == "__main__":
    main()
