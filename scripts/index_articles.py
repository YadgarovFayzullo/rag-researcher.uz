import fitz
import httpx

from core.clients import get_embedding, get_supabase

supabase = get_supabase()


def extract_text_from_pdf(url):
    response = httpx.get(url, timeout=30)
    doc = fitz.open(stream=response.content, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])


def build_text_for_embedding(article):
    parts = [
        article.get("title") or "",
        article.get("title_foreign") or "",
        article.get("authors") or "",
        article.get("annotation") or "",
        article.get("annotation_foreign") or "",
        article.get("keywords") or "",
        article.get("keywords_foreign") or "",
    ]
    return "\n".join([p for p in parts if p.strip()])


def index_articles():
    response = supabase.table("articles").select(
        "id, title, title_foreign, authors, annotation, "
        "annotation_foreign, keywords, keywords_foreign, pdf"
    ).is_("embedding", "null").execute()

    articles = response.data
    print(f"Найдено статей для индексации: {len(articles)}")

    for i, article in enumerate(articles):
        try:
            text = build_text_for_embedding(article)
            if article.get("pdf"):
                text = text + "\n" + extract_text_from_pdf(article["pdf"])[:4000]
            if not text.strip():
                print(f"[{i + 1}] Статья {article['id']} — нет текста, пропускаем")
                continue

            embedding = get_embedding(text)
            supabase.table("articles").update(
                {"embedding": embedding}
            ).eq("id", article["id"]).execute()
            print(f"[{i + 1}/{len(articles)}] OK: {article.get('title', '')[:60]}")
        except Exception as e:
            print(f"[{i + 1}] Ошибка на статье {article['id']}: {e}")
            continue


if __name__ == "__main__":
    index_articles()
