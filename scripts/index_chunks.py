import fitz
import httpx
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.clients import get_embedding, get_qdrant, get_supabase
from core.config import settings

qdrant = get_qdrant()
supabase = get_supabase()
COLLECTION = settings.QDRANT_COLLECTION

NOISE_PATTERNS = [
    "inter education & global study",
    "научно-теоретический и методический журнал",
    "scientific theoretical and methodological journal",
    "issn",
    "© intereduglobalstudy",
    "intereduglobalstudy.com",
    "original paper",
]


def is_noise(text):
    text_lower = text.lower()
    return any(p in text_lower for p in NOISE_PATTERNS)


def setup_collection():
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=settings.EMBED_DIM, distance=Distance.COSINE),
    )
    print("Коллекция создана")


def extract_pdf_text(url):
    try:
        response = httpx.get(url, timeout=30)
        doc = fitz.open(stream=response.content, filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception:
        return ""


def detect_sections(text):
    sections = {
        "introduction": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "other": "",
    }
    keywords = {
        "introduction": ["introduction", "kirish", "введение", "maqsad", "цель"],
        "methods": [
            "materials and methods", "metodlar", "материалы и методы",
            "metod", "methodology", "tadqiqot metodlari",
        ],
        "results": ["result", "natija", "результат", "findings", "натижалар"],
        "discussion": ["discussion", "muhokama", "обсуждение", "tahlil"],
        "conclusion": ["conclusion", "xulosa", "заключение", "вывод", "хулоса"],
    }

    current_section = "other"
    for line in text.split("\n"):
        line_lower = line.lower().strip()
        for section, keys in keywords.items():
            if any(k in line_lower for k in keys) and len(line_lower) < 50:
                current_section = section
                break
        sections[current_section] += line + "\n"

    return {k: v.strip() for k, v in sections.items() if v.strip()}


def chunk_article(article):
    chunks = []

    meta = "\n".join(filter(None, [
        article.get("title"),
        article.get("title_foreign"),
        article.get("authors"),
        article.get("keywords"),
        article.get("keywords_foreign"),
    ]))
    if meta and not is_noise(meta):
        chunks.append({"section": "metadata", "content": meta})

    annotation = "\n".join(filter(None, [
        article.get("annotation"),
        article.get("annotation_foreign"),
    ]))
    if annotation and len(annotation) > 300 and not is_noise(annotation):
        chunks.append({"section": "annotation", "content": annotation})

    if article.get("pdf"):
        pdf_text = extract_pdf_text(article["pdf"])
        if pdf_text:
            for section, content in detect_sections(pdf_text).items():
                if section == "other":
                    continue
                if len(content) > 150 and not is_noise(content):
                    chunks.append({"section": section, "content": content[:3000]})

    return chunks


def index_chunks():
    setup_collection()

    all_articles = []
    offset = 0
    while True:
        res = supabase.table("articles").select(
            "id, title, title_foreign, authors, annotation, "
            "annotation_foreign, keywords, keywords_foreign, pdf"
        ).range(offset, offset + 999).execute()
        all_articles.extend(res.data)
        if len(res.data) < 1000:
            break
        offset += 1000

    print(f"Статей: {len(all_articles)}")

    points = []
    point_id = 0
    for i, article in enumerate(all_articles):
        try:
            for chunk in chunk_article(article):
                points.append(PointStruct(
                    id=point_id,
                    vector=get_embedding(chunk["content"]),
                    payload={
                        "article_id": article["id"],
                        "title": article.get("title"),
                        "section": chunk["section"],
                        "content": chunk["content"][:500],
                    },
                ))
                point_id += 1
            print(f"[{i + 1}/{len(all_articles)}] {article.get('title', '')[:50]}")
        except Exception as e:
            print(f"Ошибка {article['id']}: {e}")

    for i in range(0, len(points), 100):
        qdrant.upsert(collection_name=COLLECTION, points=points[i:i + 100])

    print(f"\nГотово. Всего чанков: {point_id}")


if __name__ == "__main__":
    index_chunks()
