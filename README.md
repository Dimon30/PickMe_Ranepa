# PickMe — Бот приёмной комиссии РАНХиГС

Telegram-бот для ответов на вопросы абитуриентов РАНХиГС. Использует LLM-агентов, векторный поиск и SQL для предоставления точной информации о программах и поступлении.

## Архитектура

```
Пользователь (Telegram)
        │
        ▼
   [Bot service]          aiogram, polling
        │ HTTP POST /ask {text, user_id}
        ▼
  [FastAPI service]       Python + Uvicorn
        │
        ▼
  [Response Builder]
   ┌────┴────┐
   │  Router │  ← LLM классифицирует намерение пользователя
   └────┬────┘
        │
   ┌────┴──────────────┐
   │                   │                   │
   ▼                   ▼                   ▼
[chitchat]           [rag]              [sql]
LLM отвечает    Векторный поиск     LLM извлекает
на приветствие  в Milvus → LLM      фильтр → SQL
                формирует ответ     → LLM форматирует
        │                   │                   │
        └───────────┬───────┘───────────────────┘
                    ▼
              [Redis] ← история диалога по user_id
```

## Применение LLM

Модель: `grok-3-mini` через xAI API.

В системе четыре точки использования LLM:

1. **Роутинг** (`router.py`) — классифицирует намерение пользователя: `sql`, `rag` или `chitchat`.
2. **Chitchat** (`response_builder.py`) — генерирует ответ на приветствия и off-topic сообщения.
3. **SQL-агент** (`sql_service.py`) — извлекает параметры фильтрации из вопроса, затем форматирует результаты из базы данных в человекочитаемый ответ.
4. **RAG-ответ** (`response_builder.py`) — формирует ответ на основе релевантных фрагментов из векторной базы (FAQ и глоссарий).

## NLP-компоненты

- **Эмбеддинги** — модель `paraphrase-multilingual-mpnet-base-v2` (768 измерений) из `sentence-transformers` для векторизации FAQ и терминов.
- **Семантический поиск** — векторная БД Milvus хранит две коллекции: `faq` и `terms`; при запросе ищутся ближайшие векторы.
- **Фильтрация нецензурной лексики** — токенизация текста и сверка со словарями русских стоп-слов.
- **История диалога** — Redis хранит последние сообщения каждого пользователя (по `user_id`). История подставляется в контекст LLM при генерации ответа, что позволяет боту учитывать предыдущие вопросы в разговоре. TTL — 24 часа, лимит — 10 пар сообщений.

## Стек технологий

| Слой | Технология |
|---|---|
| Бот | Python, aiogram 3 |
| API | FastAPI, Uvicorn |
| LLM | xAI API (grok-3-mini) |
| Эмбеддинги | sentence-transformers |
| Векторная БД | Milvus 2.4 |
| Реляционная БД | PostgreSQL 16 |
| Кэш / история | Redis 7 |
| ORM | SQLAlchemy |
| Инфраструктура | Docker, Docker Compose |

## Структура проекта

```
PickMe_Ranepa/
├── app/                    # FastAPI-приложение
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   ├── response_builder.py  # Главный пайплайн
│   │   ├── router.py            # Классификация намерений
│   │   ├── llm_service.py       # Groq API
│   │   ├── faq_service.py       # Поиск по Milvus
│   │   ├── sql_service.py       # SQL-запросы
│   │   ├── chat_history.py      # История диалогов (Redis)
│   │   └── preprocess.py        # Фильтрация нецензурной лексики
│   └── db/
│       ├── session.py
│       ├── models.py
│       └── repository.py
├── bot/
│   └── tg_bot.py               # Telegram-бот
├── data/
│   ├── all_program.xlsx         # Образовательные программы
│   ├── Database.xlsx            # FAQ
│   └── Database-2.xlsx          # Глоссарий
├── scripts/
│   └── ingest.py               # Загрузка данных в БД и Milvus
├── thursday_tg_bot.py          # Утилита для тестирования бота через MTProto
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Запуск

### Требования

- Docker и Docker Compose
- Токен Telegram-бота ([@BotFather](https://t.me/BotFather))
- API-ключ [xAI](https://console.x.ai)

### 1. Настройка окружения

```bash
cp .env.example .env
```

Заполни `.env`:

```env
BOT_TOKEN=<токен от @BotFather>
XAI_API_KEY=<ключ с console.x.ai>
```

### 2. Запуск всех сервисов

```bash
docker compose up -d --build
```

Запустятся: PostgreSQL, Redis, Milvus (+ etcd, MinIO), FastAPI, бот.

### 3. Загрузка данных (первый запуск)

После того как все сервисы поднялись, залей данные в БД и Milvus:

```bash
docker compose exec fastapi python scripts/ingest.py
```

### 4. Проверка

```bash
# Статус контейнеров
docker compose ps

# Healthcheck API
curl http://localhost:8000/health

# Логи бота
docker compose logs bot -f
```

### Остановка

```bash
docker compose down
```

---

### Тестирование через MTProto (`thursday_tg_bot.py`)

Скрипт для автоматической отправки тестовых вопросов боту через Telegram MTProto API (Telethon).
Требует `TELEGRAM_API_ID` и `TELEGRAM_API_HASH` в `.env` (получить на [my.telegram.org](https://my.telegram.org)).

```bash
pip install telethon python-dotenv
python thursday_tg_bot.py
```
