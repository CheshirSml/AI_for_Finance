# -*- coding: utf-8 -*-
"""
НЕЗАПУСКАЛ

RAG в режиме STRICT (для максимальной точности оценки LLM-судьёй).
Отключает: переформулировку, компрессию, fallback на grok.
Использует: только дословные фрагменты из train_data.csv.
"""

import os
import logging
import time
import pandas as pd
import re
import math
import requests
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# === Компоненты LangChain ===
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# === Логирование ===
logger = logging.getLogger("StrictRAG")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler("strict_rag.log", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# === Загрузка .env ===
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
PROXY_URL = "https://ai-for-finance-hack.up.railway.app/"
RERANKER_URL = "https://ai-for-finance-hack.up.railway.app/rerank"

# === Константы ===
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "openrouter/mistralai/mistral-small-3.2-24b-instruct"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " "]
TOP_K_INITIAL = 10
TOP_K_FINAL = 2

TRAIN_DATA_PATH = "./train_data.csv"   
QUESTIONS_PATH = "./questions.csv"     

LLM_INPUT_COST_PER_1M = 0.20
LLM_OUTPUT_COST_PER_1M = 0.20

total_llm_input_tokens = 0
total_llm_output_tokens = 0


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))


def simple_tokenize(text: str):
    return [w for w in re.findall(r"\b[\w%$]+\b", text.lower()) if len(w) > 1]


class SimpleBM25:
    def __init__(self, corpus: list):
        self.corpus = corpus
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self._initialize()

    def _initialize(self):
        ndocs = len(self.corpus)
        freqs = []
        doc_lens = []
        all_tokens = set()
        for doc in self.corpus:
            tokens = simple_tokenize(doc)
            doc_lens.append(len(tokens))
            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
                all_tokens.add(token)
            freqs.append(freq)
        self.doc_len = doc_lens
        self.avg_doc_len = sum(doc_lens) / ndocs
        self.doc_freqs = freqs
        for token in all_tokens:
            containing_docs = sum(1 for freq in freqs if token in freq)
            self.idf[token] = math.log((ndocs - containing_docs + 0.5) / (containing_docs + 0.5) + 1)

    def get_scores(self, query: str):
        query_tokens = simple_tokenize(query)
        scores = []
        k1, b = 1.5, 0.75
        for i, doc in enumerate(self.corpus):
            doc_len = self.doc_len[i]
            freq = self.doc_freqs[i]
            score = 0.0
            for token in query_tokens:
                if token not in freq:
                    continue
                tf = freq[token]
                idf = self.idf.get(token, 0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / self.avg_doc_len)
                score += idf * numerator / denominator
            scores.append(score)
        return np.array(scores)


def rerank_docs(query, documents, key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "model": "deepinfra/Qwen/Qwen3-Reranker-4B",
        "query": query,
        "documents": documents
    }
    response = requests.post(RERANKER_URL, headers=headers, json=payload)
    return response.json()


def load_and_chunk_documents(csv_path: str):
    logger.info("Загрузка и чанкинг базы знаний...")
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        text = row["text"]
        if pd.isna(text):
            continue
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
        )
        chunks = splitter.split_text(str(text))
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 30:
                documents.append(Document(page_content=chunk))
    logger.info(f"Создано {len(documents)} чанков.")
    return documents


class HybridRetriever:
    def __init__(self, faiss_retriever, bm25_retriever):
        self.faiss = faiss_retriever
        self.bm25 = bm25_retriever

    def retrieve_candidates(self, query, k=10):
        faiss_docs = self.faiss.invoke(query) if self.faiss else []
        faiss_texts = [d.page_content for d in faiss_docs]

        bm25_scores = self.bm25.get_scores(query) if self.bm25 else np.array([])
        if len(bm25_scores) > 0:
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:k]
            bm25_texts = [self.bm25.corpus[i] for i in top_bm25_idx]
        else:
            bm25_texts = []

        all_texts = list(dict.fromkeys(faiss_texts + bm25_texts))
        return all_texts[:20]


def answer_generation(question: str, hybrid_retriever, embedder_key, llm_key) -> str:
    global total_llm_input_tokens, total_llm_output_tokens

    # STRICT: без переформулировки
    candidates = hybrid_retriever.retrieve_candidates(question, k=TOP_K_INITIAL)
    if not candidates:
        return "Информация по данному вопросу отсутствует в базе знаний."

    # STRICT: без компрессии — берём 1–2 лучших фрагмента как есть
    rerank_result = rerank_docs(question, candidates, embedder_key)
    reranked = rerank_result.get("results", [])
    if not reranked or reranked[0].get("relevance_score", 0) < 0.3:
        return "Информация по данному вопросу отсутствует в базе знаний."

    top_docs = []
    for item in reranked[:TOP_K_FINAL]:
        if "index" in item:
            idx = item["index"]
            if 0 <= idx < len(candidates):
                top_docs.append(candidates[idx])
        else:
            top_docs.append(item.get("document") or item.get("text") or "")

    # STRICT: дословный контекст
    context_text = "\n\n".join([f"Фрагмент {i+1}:\n{ctx}" for i, ctx in enumerate(top_docs)])
    prompt = f"""Ответь ТОЛЬКО на основе приведённых фрагментов.
- Если информация есть — ответь дословно или близко к тексту.
- Если информации нет — напиши строго: "Информация по данному вопросу отсутствует в базе знаний."
- Не добавляй своих пояснений, интерпретаций или примеров.

Фрагменты:
{context_text}

Вопрос: {question}

Ответ:"""

    input_tokens = count_tokens(prompt)
    total_llm_input_tokens += input_tokens

    client = OpenAI(base_url=PROXY_URL, api_key=llm_key)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        answer = "Информация по данному вопросу отсутствует в базе знаний."

    output_tokens = count_tokens(answer)
    total_llm_output_tokens += output_tokens
    return answer


# === Основной запуск ===
if __name__ == "__main__":
    overall_start = time.time()
    logger.info(" Запуск RAG в режиме STRICT (для максимизации оценки LLM-судьи)")

    questions_df = pd.read_csv(QUESTIONS_PATH)
    if 'Вопрос' in questions_df.columns:
        questions_list = questions_df['Вопрос'].tolist()
    elif 'question_text' in questions_df.columns:
        questions_list = questions_df['question_text'].tolist()
    else:
        raise ValueError("Колонка с вопросами не найдена")
    logger.info(f"Вопросов: {len(questions_list)}")

    docs = load_and_chunk_documents(TRAIN_DATA_PATH)
    texts = [d.page_content for d in docs]

    logger.info("Инициализация FAISS...")
    embeddings = OpenAIEmbeddings(base_url=PROXY_URL, api_key=EMBEDDER_API_KEY, model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_INITIAL})

    logger.info("Инициализация BM25...")
    bm25_retriever = SimpleBM25(corpus=texts)

    hybrid_retriever = HybridRetriever(faiss_retriever, bm25_retriever)

    answer_list = []
    for q in tqdm(questions_list, desc="STRICT RAG"):
        try:
            ans = answer_generation(q, hybrid_retriever, EMBEDDER_API_KEY, LLM_API_KEY)
        except Exception as e:
            logger.error(f"Ошибка на '{q[:50]}...': {e}")
            ans = "Информация по данному вопросу отсутствует в базе знаний."
        answer_list.append(ans)

    questions_df['Ответы на вопрос'] = answer_list
    questions_df.to_csv('submission.csv', index=False, encoding='utf-8')
    logger.info(" submission.csv сохранён (STRICT RAG).")

    # Стоимость
    input_cost = (total_llm_input_tokens / 1_000_000) * LLM_INPUT_COST_PER_1M
    output_cost = (total_llm_output_tokens / 1_000_000) * LLM_OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost

    total_time = time.time() - overall_start
    summary = (
        f"\n{'='*60}\n"
        f" STRICT RAG ИТОГ\n"
        f" Время: {total_time:.2f} сек\n"
        f" Input: {total_llm_input_tokens:,}\n"
        f" Output: {total_llm_output_tokens:,}\n"
        f" Стоимость: ${total_cost:.4f}\n"
        f"{'='*60}"
    )
    logger.info(summary)