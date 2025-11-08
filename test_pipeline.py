'''
единый тестовый скрипт test_pipeline.py, который объединяет все три проверки:

Тест токенизатора (tokenize_ru),
Тест BM25 (SimpleBM25),
Интеграционный тест пайплайна ((rewrite → retrieve → generate)).

Каждый тест явно сообщает о прохождении или провале, а при ошибке — выводит детали.


# В PowerShell:
$env:USE_MOCK="1"; python test_pipeline.py

# В CMD:
set USE_MOCK=1 && python test_pipeline.py

# В Linux/macOS:
USE_MOCK=1 python test_pipeline.py
'''

# -*- coding: utf-8 -*-


import os
import sys

# Включаем mock-режим для безопасного тестирования без API
os.environ["USE_MOCK"] = "1"

# Импортируем функции из main.py
try:
    from main_4 import (
        tokenize_ru,
        SimpleBM25,
        rewrite_queries,
        hybrid_retrieve,
        answer_generation
    )
except ImportError as e:
    print(f"❌ ОШИБКА ИМПОРТА: {e}")
    sys.exit(1)


def test_tokenizer():
    """Тест: tokenize_ru корректно обрабатывает русские слова."""
    try:
        tokens = tokenize_ru("Как открыть вклад?")
        assert "вклад" in tokens, f"'вклад' отсутствует в {tokens}"
        print("✅ Тест токенизатора пройден")
        return True
    except Exception as e:
        print(f"❌ Тест токенизатора НЕ пройден: {e}")
        return False


def test_bm25():
    """Тест: BM25 правильно ранжирует релевантные документы."""
    try:
        docs = ["открыть вклад в банке", "как получить карту"]
        bm25 = SimpleBM25(docs)
        scores = bm25.get_scores("вклад")
        assert len(scores) == 2, "Неверное количество скоров"
        assert scores[0] > scores[1], f"Нерелевантный документ выше: {scores}"
        print("✅ Тест BM25 пройден")
        return True
    except Exception as e:
        print(f"❌ Тест BM25 НЕ пройден: {e}")
        return False


def test_pipeline():
    """Тест: полный пайплайн работает без исключений."""
    try:
        # 1. Переформулировка
        rewrites = rewrite_queries("Как открыть депозит?")
        assert isinstance(rewrites, list) and len(rewrites) > 0
        print(f"   → Переформулировка: {len(rewrites)} вариантов")

        # 2. Retrieval
        retrieved = hybrid_retrieve("вклад открыть")
        assert isinstance(retrieved, list)
        print(f"   → Найдено фрагментов: {len(retrieved)}")

        # 3. Генерация
        answer = answer_generation("Как пополнить карту?")
        assert isinstance(answer, str) and len(answer) > 0
        print(f"   → Сгенерирован ответ: {answer[:50]}...")

        print("✅ Тест пайплайна пройден")
        return True
    except Exception as e:
        print(f"❌ Тест пайплайна НЕ пройден: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ RAG-ПАЙПЛАЙНА (в mock-режиме)")
    print("USE_MOCK=1 — без вызовов API, без траты токенов")
    print("=" * 60)

    results = []
    results.append(test_tokenizer())
    results.append(test_bm25())
    results.append(test_pipeline())

    print("=" * 60)
    if all(results):
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ. Пайплайн готов к запуску.")
    else:
        print("⚠️ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ. Исправьте ошибки перед запуском.")
    print("=" * 60)