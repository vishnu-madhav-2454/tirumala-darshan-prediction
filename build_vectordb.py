"""
Build ChromaDB vector database from TTD corpus and trip data.
Uses sentence-transformers for embeddings (runs on CPU, ~90MB model).
"""
import os, json, re, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("build_vectordb")

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS   = os.path.join(BASE, "ttd_corpus.txt")
KB_JSON  = os.path.join(BASE, "ttd_knowledge_base.json")
TRIP_JSON = os.path.join(BASE, "tirumala_trip_data.json")
DB_DIR   = os.path.join(BASE, "vectordb")


def _chunk_corpus(text: str, max_tokens=400, overlap=60) -> list[dict]:
    """Split corpus into overlapping chunks by markdown sections."""
    sections = re.split(r'\n---\n|\n## ', text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 30:
            continue
        # Extract a title from the first line
        lines = sec.split('\n')
        title = lines[0].strip('#').strip() if lines else "General"
        # If section is short enough, keep as one chunk
        words = sec.split()
        if len(words) <= max_tokens:
            chunks.append({"text": sec, "title": title, "source": "corpus"})
        else:
            # Split into paragraphs and combine up to max_tokens
            paragraphs = re.split(r'\n\n+', sec)
            buf, buf_title = [], title
            word_count = 0
            for p in paragraphs:
                p_words = len(p.split())
                if word_count + p_words > max_tokens and buf:
                    chunks.append({"text": "\n\n".join(buf), "title": buf_title, "source": "corpus"})
                    # Keep last paragraph as overlap
                    buf = [buf[-1]] if overlap > 0 and len(buf) > 1 else []
                    word_count = len(buf[0].split()) if buf else 0
                buf.append(p)
                word_count += p_words
                # Try to grab sub-title
                for line in p.split('\n'):
                    if line.startswith('###'):
                        buf_title = line.strip('#').strip()
            if buf:
                chunks.append({"text": "\n\n".join(buf), "title": buf_title, "source": "corpus"})
    return chunks


def _chunks_from_kb(kb: dict) -> list[dict]:
    """Convert knowledge base QA pairs into chunks."""
    chunks = []
    for cat in kb.get("categories", []):
        cat_name = cat.get("name", "General")
        for qa in cat.get("qa", []):
            text = f"Category: {cat_name}\nQ: {qa['q']}\nA: {qa['a']}"
            chunks.append({"text": text, "title": cat_name, "source": "knowledge_base"})
    return chunks


def _chunks_from_trip(td: dict) -> list[dict]:
    """Convert trip data JSON into searchable chunks."""
    chunks = []

    # Hotels
    for group_name, hotels in td.get("hotels", {}).items():
        for h in hotels:
            price = h.get("price_range", {})
            text = (
                f"Hotel: {h['name']}\n"
                f"Location: {group_name.replace('_', ' ').title()}\n"
                f"Type: {h.get('type', 'N/A')}\n"
                f"Category: {h.get('category', 'N/A')}\n"
                f"Price: ₹{price.get('min', 'N/A')}-₹{price.get('max', 'N/A')}/night\n"
                f"Rating: {h.get('rating', 'N/A')}\n"
                f"Amenities: {', '.join(h.get('amenities', []))}\n"
                f"Coordinates: {h.get('lat', '')}, {h.get('lng', '')}"
            )
            chunks.append({"text": text, "title": h["name"], "source": "hotels"})

    # Attractions
    for a in td.get("attractions", []):
        text = (
            f"Attraction: {a['name']}\n"
            f"Type: {a.get('type', 'N/A')}\n"
            f"Description: {a.get('description', '')}\n"
            f"Visit Duration: {a.get('visit_duration_hours', 'N/A')} hours\n"
            f"Timings: {a.get('timings', 'N/A')}\n"
            f"Entry Fee: ₹{a.get('entry_fee', 0)}\n"
            f"Tips: {a.get('tips', '')}\n"
            f"Coordinates: {a.get('lat', '')}, {a.get('lng', '')}"
        )
        chunks.append({"text": text, "title": a["name"], "source": "attractions"})

    # Restaurants
    for r in td.get("restaurants", []):
        text = (
            f"Restaurant: {r['name']}\n"
            f"Cuisine: {r.get('cuisine', 'N/A')}\n"
            f"Price per Person: ₹{r.get('price_per_person', 'N/A')}\n"
            f"Timings: {r.get('timings', 'N/A')}\n"
            f"Location: {r.get('location', 'N/A')}\n"
            f"Description: {r.get('description', '')}"
        )
        chunks.append({"text": text, "title": r["name"], "source": "restaurants"})

    # Transport
    transport = td.get("transport", {})
    for mode in ["by_air", "by_rail", "by_road"]:
        data = transport.get("how_to_reach_tirupati", {}).get(mode, {})
        if data:
            text = f"Transport Mode: {mode.replace('_', ' ').title()}\n{json.dumps(data, indent=2, ensure_ascii=False)}"
            chunks.append({"text": text[:800], "title": f"Transport - {mode}", "source": "transport"})

    # Local transport
    local = transport.get("tirupati_to_tirumala", {})
    if local:
        text = f"Tirupati to Tirumala Transport:\n{json.dumps(local, indent=2, ensure_ascii=False)}"
        chunks.append({"text": text[:800], "title": "Tirupati to Tirumala", "source": "transport"})

    # Sevas
    for s in td.get("sevas", []):
        text = (
            f"Seva: {s['name']}\n"
            f"Cost: ₹{s['cost']}\n"
            f"Time: {s['time']}\n"
            f"Description: {s['description']}\n"
            f"Booking: {'Online' if s.get('online_booking') else 'Counter'}"
        )
        chunks.append({"text": text, "title": s["name"], "source": "sevas"})

    # Festivals
    for f in td.get("festivals", []):
        text = (
            f"Festival: {f.get('name', 'Unknown')}\n"
            f"Month: {f.get('month', 'N/A')}\n"
            f"Duration: {f.get('duration', f.get('duration_days', 'N/A'))}\n"
            f"Description: {f.get('description', f.get('significance', ''))}"
        )
        chunks.append({"text": text, "title": f["name"], "source": "festivals"})

    # Daily costs
    for tier, info in td.get("daily_costs_estimate", {}).items():
        text = f"Budget Tier: {tier.title()}\nPer Day: ₹{info.get('per_day_inr', 'N/A')}\nIncludes: {info.get('includes', 'N/A')}"
        chunks.append({"text": text, "title": f"Budget - {tier}", "source": "costs"})

    # Tips
    for i, tip in enumerate(td.get("tips", []), 1):
        chunks.append({"text": f"Travel Tip #{i}: {tip}", "title": f"Tip {i}", "source": "tips"})

    return chunks


def build():
    """Main build: chunk data → embed → store in ChromaDB."""
    import chromadb
    from chromadb.utils import embedding_functions

    log.info("📚 Loading data sources...")
    all_chunks = []

    # 1. Corpus
    if os.path.exists(CORPUS):
        with open(CORPUS, "r", encoding="utf-8") as f:
            corpus_text = f.read()
        corpus_chunks = _chunk_corpus(corpus_text)
        all_chunks.extend(corpus_chunks)
        log.info("  ✅ Corpus: %d chunks", len(corpus_chunks))

    # 2. Knowledge Base QA
    if os.path.exists(KB_JSON):
        with open(KB_JSON, "r", encoding="utf-8") as f:
            kb = json.load(f)
        kb_chunks = _chunks_from_kb(kb)
        all_chunks.extend(kb_chunks)
        log.info("  ✅ Knowledge Base: %d chunks", len(kb_chunks))

    # 3. Trip Data
    if os.path.exists(TRIP_JSON):
        with open(TRIP_JSON, "r", encoding="utf-8") as f:
            td = json.load(f)
        trip_chunks = _chunks_from_trip(td)
        all_chunks.extend(trip_chunks)
        log.info("  ✅ Trip Data: %d chunks", len(trip_chunks))

    log.info("📊 Total chunks: %d", len(all_chunks))

    # 4. Build ChromaDB with sentence-transformers
    log.info("🧠 Initializing embedding model (all-MiniLM-L6-v2)...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Persistent storage
    if os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.create_collection(
        name="ttd_knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Add chunks in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[{"title": c["title"], "source": c["source"]} for c in batch],
            ids=[f"chunk_{i+j}" for j in range(len(batch))]
        )
        log.info("  Added batch %d-%d / %d", i, i+len(batch), len(all_chunks))

    log.info("✅ Vector DB built at %s with %d documents", DB_DIR, collection.count())
    log.info("   Collection count: %d", collection.count())

    # Quick test
    results = collection.query(query_texts=["How to book darshan tickets?"], n_results=3)
    log.info("🔍 Test query: 'How to book darshan tickets?'")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        log.info("   [%s] %s", meta["source"], doc[:100])

    return collection


if __name__ == "__main__":
    build()
