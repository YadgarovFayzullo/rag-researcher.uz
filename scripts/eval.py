import asyncio

asyncio.set_event_loop(asyncio.new_event_loop())

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import answer_relevancy, faithfulness  # noqa: E402

from core.rag import ask  # noqa: E402

test_questions = [
    "biologiyada innovatsion o'qitish metodlari qanday?",
    "maktabgacha ta'limda bolalarni rivojlantirish usullari?",
    "ingliz tilini o'qitishda kommunikativ yondashuv nima?",
    "raqamli texnologiyalar ta'limda qanday qo'llaniladi?",
    "o'qituvchilarning kasbiy kompetensiyasi nima?",
]

print("Генерируем ответы...")
data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for q in test_questions:
    answer, sources = ask(q)
    from core.rag import search_chunks

    contexts = [c.payload["content"] for c in search_chunks(q)]
    data["question"].append(q)
    data["answer"].append(answer)
    data["contexts"].append(contexts)
    data["ground_truth"].append("")
    print(f"✓ {q[:150]}")

dataset = Dataset.from_dict(data)

print("\nЗапускаем RAGAS eval...")
from langchain_community.embeddings import OllamaEmbeddings  # noqa: E402
from langchain_community.llms import Ollama  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

from core.config import settings  # noqa: E402

llm = LangchainLLMWrapper(Ollama(model=settings.LLM_MODEL))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=settings.EMBED_MODEL))

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings,
)

print("\n=== RAGAS Results ===")
print(results)
