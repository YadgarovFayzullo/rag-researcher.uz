# RAG API

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-DC244C)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Сервис retrieval-augmented генерации по научным статьям: семантический поиск по
чанкам в Qdrant + генерация ответа через OpenAI-совместимый LLM. Многоязычный
(узбекский / русский / английский), с трейсингом качества через Langfuse.

**Стек:** Python 3.12 · FastAPI · Qdrant · Supabase (pgvector) · Ollama /
OpenAI-совместимый LLM · Langfuse · Docker Compose.

## Возможности

- **Многоязычный поиск** — узбекский, русский, английский в одном индексе.
- **Чанкинг по секциям** — текст статьи разбивается на 7 типов чанков:
  `metadata`, `annotation`, `introduction`, `methods`, `results`,
  `discussion`, `conclusion` (распознавание по ключевым словам на 3 языках).
- **Фильтрация шума** — отсев колонтитулов, ISSN и служебных строк журнала.
- **Фильтр по секции** — поиск можно ограничить, напр. только `methods`.
- **Векторный поиск** — `nomic-embed-text`, 768 измерений, косинусное расстояние,
  порог отсечения по score.
- **Оценка качества** — LangFuse-evaluator'ы поверх трейсов (answer relevancy,
  hallucination, correctness); RAGAS-скрипт для оффлайн-прогона включён.
- **Observability** — трейсинг каждого запроса в Langfuse.
- **Готов к проду** — Docker, healthcheck, liveness/readiness, non-root, env-конфиг.

## Статистика

| Показатель | Значение |
|-----------|----------|
| Проиндексировано статей | 1 321 |
| Всего чанков в Qdrant | 2 936 |
| Языков в индексе | 3 (uz / ru / en) |
| Типов чанков | 7 |
| Размерность эмбеддингов | 768 |
| REST-эндпоинтов | 4 (`/ask`, `/search`, `/health`, `/ready`) |

**Качество ответов** (оценка через LangFuse-evaluator'ы):

| Метрика | Значение |
|---------|----------|
| Answer relevancy | 0.65 |
| Hallucination | 0.0 |
| Correctness | 1.0 |

> RAGAS (`scripts/eval.py`) на локальной модели не довёл прогон до конца, поэтому
> метрики качества собирались LangFuse-evaluator'ами поверх трейсов.

**Задержка `/ask`** (замер на MacBook Pro, Apple M5, 16 GB RAM, 1 ТБ SSD; Ollama через Metal):

| Этап | Время |
|------|-------|
| Эмбеддинг запроса | 0.03–0.5 с |
| Поиск в Qdrant | < 0.01 с |
| Генерация (`qwen3:8b`) | 15–39 с |
| Итого (warm) | ~25 с в среднем |

> Сам RAG-пайплайн (эмбеддинг + поиск) укладывается в **< 0.6 с** — всё время
> съедает генерация локальной reasoning-модели `qwen3:8b` на встроенном GPU.
> На выделенном GPU-сервере или хостед-LLM задержка снижается в разы; backend
> меняется через `.env` без правки кода.

## Архитектура

```
              ┌──────────────┐
HTTP  ───────▶│  core/api.py │  FastAPI (uvicorn, 2 воркера)
              │   /ask /search│
              └──────┬───────┘
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
   ┌──────────────┐      ┌──────────────┐
   │   Qdrant     │      │  LLM backend │  Ollama (self-hosted)
   │ (вектора)    │      │ OpenAI-совм. │  ИЛИ облачный провайдер
   └──────────────┘      └──────────────┘
           ▲
           │ индексация (scripts/)
   ┌──────────────┐
   │  Supabase    │  исходные статьи + pgvector
   └──────────────┘
```

| Компонент | Файл | Назначение |
|-----------|------|-----------|
| Конфиг | `core/config.py` | все настройки из env, один источник правды |
| Клиенты | `core/clients.py` | LLM / эмбеддинги / Qdrant / Supabase + `get_embedding` |
| RAG-ядро | `core/rag.py` | `search_chunks`, `ask` (+ трейсинг Langfuse) |
| API | `core/api.py` | HTTP-эндпоинты `/ask`, `/search`, `/health`, `/ready` |
| Скрипты | `scripts/` | индексация и оффлайн-оценка (не в рантайме API) |

**Backend LLM — код-агностик.** По умолчанию Ollama; чтобы перейти на облачный
OpenAI-совместимый провайдер, поменяйте только `LLM_BASE_URL` / `LLM_API_KEY` /
`LLM_MODEL` в `.env` — код не трогается.

## Запуск (docker-compose)

```bash
cp .env.example .env          # заполните секреты
docker compose up -d --build  # api + qdrant
```

С self-hosted Ollama (нужен достаточно мощный хост, для скорости — GPU):

```bash
docker compose --profile local-llm up -d --build
# подтянуть модели внутрь контейнера ollama:
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull nomic-embed-text
# в .env: LLM_BASE_URL=http://ollama:11434/v1 (EMBED_BASE_URL аналогично)
```

API поднимется на `http://localhost:8000` (`/docs` — Swagger UI).

## Эндпоинты

```bash
curl http://localhost:8000/health        # liveness
curl http://localhost:8000/ready         # readiness (проверяет Qdrant)

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "innovatsion o'\''qitish metodlari qanday?"}'

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "teaching methods", "limit": 5, "section": "methods"}'
```

## Индексация

```bash
docker compose exec api python -m scripts.index_chunks    # чанки → Qdrant
docker compose exec api python -m scripts.index_articles  # эмбеддинги → Supabase
docker compose exec api python -m scripts.clean_chunks    # удалить коллекцию
```

## Оценка качества (оффлайн)

Тяжёлые зависимости вынесены в отдельный файл и не входят в продакшен-образ:

```bash
pip install -r requirements-eval.txt
python -m scripts.eval
```

> На локальной LLM прогон RAGAS может зависать. Текущие метрики качества
> собирались LangFuse-evaluator'ами поверх трейсов реальных запросов.

## Локальная разработка без Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn core.api:app --reload
```

## Лицензия

[MIT](LICENSE) © Fayzullo Yadgarov
