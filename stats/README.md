
# Stats folder for PickMe_Ranepa

Эта папка нужна для офлайн-оценки качества. Она **не меняет поведение бота для абитуриентов**.

## Что увидит пользователь

Обычный пользователь **ничего не увидит**.
Скрипты просто отправляют HTTP-запросы в твой `/ask` как технический клиент.

Что реально происходит:
- сервис FastAPI получает тестовые вопросы;
- ответы сохраняются в `stats/results/*.csv`;
- графики сохраняются в `stats/plots/*.png`.

Единственный побочный эффект: в таблицу истории чатов могут записаться технические eval-диалоги.
Чтобы не смешивать их с реальными пользователями, в `eval_live_api.py` используется отдельный диапазон `user_id` (`9000000000+`).
В Telegram это не показывается, бот никому сам не пишет.

## Что считается

### 1. End-to-end по живому API
Для всех трёх тестовых файлов:
- `cosine_similarity`
- `F1`
- `exact_match`
- `latency`

Файл: `stats/eval_live_api.py`

### 2. Retrieval для RAG-ветки
Для `test_database.xlsx` и `test_database2.xlsx`:
- `Recall@5`
- `MRR`

Файл: `stats/eval_retrieval_rag.py`

Логика строгая: gold source определяется по точному совпадению ответа с исходной базой `Database.xlsx` / `Database-2.xlsx`, после чего смотрим, на каком месте этот source вернулся из `faq_service.search_all(..., top_k=5)`.

### 3. Диагностика SQL-ветки
Для `test_all_program.xlsx`:
- `filter_column_program_rate`
- `Recall@5 proxy`
- `MRR proxy`

Файл: `stats/eval_sql_diagnostics.py`

Почему proxy: SQL-ветка у тебя не делает векторный retrieval как RAG, а сначала извлекает фильтр, потом фильтрует БД. Поэтому здесь мы отдельно проверяем, насколько часто LLM правильно понимает, что нужен `program`, и попадает ли gold-программа в top-5 кандидатов по извлечённому фильтру.

## Как встроить в проект

Положи папку `stats/` в корень проекта:

```bash
~/PickMe_Ranepa/stats/
```

Пример структуры:

```text
PickMe_Ranepa/
  app/
  bot/
  data/
  stats/
```

## Установка зависимостей

Внутри твоего окружения на сервере:

```bash
pip install pandas matplotlib scikit-learn rapidfuzz requests sentence-transformers openpyxl
```

Если всё это уже стоит через основной `requirements.txt`, просто докинь недостающее.

## Как запускать

Из корня проекта:

```bash
chmod +x stats/run_all.sh
API_URL=http://127.0.0.1:8000/ask bash stats/run_all.sh
```

Если ты запускаешь из docker-compose и API доступен по имени сервиса, можно так:

```bash
API_URL=http://fastapi:8000/ask bash stats/run_all.sh
```

## Какие файлы появятся

### Таблицы с результатами
- `stats/results/api_eval_all.csv`
- `stats/results/api_eval_summary.csv`
- `stats/results/retrieval_eval_all.csv`
- `stats/results/retrieval_eval_summary.csv`
- `stats/results/sql_diagnostics.csv`
- `stats/results/sql_diagnostics_summary.csv`
- `stats/results/SUMMARY.md`

### Графики
- `stats/plots/api_cosine_by_dataset.png`
- `stats/plots/api_f1_by_dataset.png`
- `stats/plots/api_latency_by_dataset.png`
- `stats/plots/api_latency_hist.png`
- `stats/plots/api_cosine_hist.png`
- `stats/plots/retrieval_recall_at5.png`
- `stats/plots/retrieval_mrr.png`
- `stats/plots/sql_recall_at5_proxy.png`
- `stats/plots/sql_mrr_proxy.png`

## Что я бы рассказывал на защите

1. Я разделил качество на два слоя:
   - end-to-end качество ответа;
   - качество retrieval/выбора данных.

2. Для RAG-ветки отдельно посчитал `Recall@5` и `MRR`, чтобы показать, находит ли система правильный источник до генерации.

3. Для живого API посчитал `cosine similarity`, `F1`, `exact match` и `latency`, чтобы показать, насколько ответ близок к эталону и сколько система отвечает по времени.

4. Для SQL-ветки добавил отдельную диагностику, потому что там механизм другой: сначала извлечение фильтра, потом работа по таблице.

## Что можно улучшить потом

Самый правильный следующий шаг — добавить в прод debug-логирование:
- `intent` роутера;
- top-k retrieval chunks;
- выбранный `filter_column` и `filter_value`;
- итоговый source_id.

Тогда можно будет строить ещё более строгую аналитику по этапам пайплайна.
