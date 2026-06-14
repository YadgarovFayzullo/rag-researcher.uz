from core.clients import get_embedding, get_supabase

supabase = get_supabase()


def search(query, limit=5):
    vector = get_embedding(query)
    result = supabase.rpc("search_articles", {
        "query_embedding": vector,
        "query_text": query,
        "match_count": limit,
    }).execute()
    return result.data


if __name__ == "__main__":
    queries = [
        "biologiyada innovatsion metodlar",
        "инновационные методы обучения",
        "innovative teaching methods",
    ]
    for q in queries:
        print(f"\n--- {q} ---")
        for r in search(q, limit=3):
            print(f"{r['similarity']:.3f} | {r['title'][:60]}")
