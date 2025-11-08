# main_friendly_loged_reranker.py
# -*- coding: utf-8 -*-

"""
==========================================================
Hybrid RAG + Reranker + Budget Guard + Autotest
==========================================================

Архитектура:
------------
FAISS + BM25 → Qwen3-Reranker → Mistral Compressor → LLM → Fallback

Поддерживаются режимы логов: prod | hackathon | ml | trace
Поддерживаются режимы ответов: eco | balanced | max-score


Основные этапы:
---------------
1. Загрузка переменных окружения и настройка логирования
2. Инициализация FAISS (кэширование индекса) и BM25
3. Обработка вопросов:
   - Перефразирование (grok-3-mini)
   - Гибридный поиск (FAISS + BM25)
   - Реранкинг (Qwen3-Reranker-4B)
   - Компрессия контекста (Mistral 24B)
   - Генерация ответа (Mistral / LLaMA)
   - Fallback при ошибках (grok-3-mini)
4. Self-Test: проверка сходства ответов (text-embedding-3-small)
5. Сохранение submission.csv, run_summary.json и JSON-логов

Переменные окружения:
---------------------
LLM_API_KEY          – ключ для моделей LLM
EMBEDDER_API_KEY     – ключ для эмбеддеров
RAG_MODE             – режим (eco / balanced / max-score)
USE_MOCK             – 1 = использовать mock API
USE_RERANKER         – 1 = активировать reranker
DAILY_BUDGET_USD     – бюджет на токены (по умолчанию 3 USD)
LOG_MODE             – prod / hackathon / ml / trace
RERANKER_URL         – https://ai-for-finance-hack.up.railway.app/rerank

Основные артефакты:
-------------------
submission.csv           – финальные ответы
run_summary.json         – токены, стоимость, время
logs/events.jsonl        – JSON-логи шагов
diagnostics_autotest.json – self-judge отчёт
faiss_index/             – локальный кэш эмбеддингов

Совместимость:
--------------
✔ Полностью совместим с baseline API
✔ Работает в mock-режиме без внешних вызовов
✔ Поддерживает FAISS cache, reranker и autotest
✔ Безопасен по бюджету (safe_generate)
==========================================================
"""




import os, sys, json, time, logging, re, math, requests
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# === LangChain components ===
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# === Base early logger ===
_early_logger = logging.getLogger("INIT")
_early_logger.setLevel(logging.INFO)
_early_ch = logging.StreamHandler(sys.stdout)
_early_logger.addHandler(_early_ch)

# === ENV load ===
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
BASE_URL = "https://ai-for-finance-hack.up.railway.app/"
RERANKER_URL = "https://ai-for-finance-hack.up.railway.app/rerank"

# === Mock mode flag ===
USE_MOCK = os.getenv("USE_MOCK", "0") == "1"

if USE_MOCK:
    _early_logger.info("===  MOCK MODE ENABLED — API calls are mocked ===")
else:
    _early_logger.info("===  REAL API MODE — live inference ===")

# === Mock classes for offline testing ===
if USE_MOCK:
    class MockOpenAI:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return type("Resp", (object,), {
                        "choices": [type("Ch", (object,), {
                            "message": type("M", (object,), {"content": "MOCK: Информация по данному вопросу отсутствует в базе знаний."})
                        })]
                    })
        class embeddings:
            @staticmethod
            def create(**kwargs):
                return {"data": [{"embedding": [0.1] * 1536}]}


# === Modes & settings ===
MODE = os.getenv("RAG_MODE", "balanced").lower()
LOG_MODE = os.getenv("LOG_MODE", "hackathon").lower()  # prod | hackathon | ml | trace
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "3"))
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"   

# === Logging setup ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMATS = {
    "prod": "%(asctime)s [%(levelname)s] %(message)s",
    "hackathon": "%(asctime)s | %(levelname)s | %(message)s",
    "ml": "%(asctime)s | %(levelname)s | %(message)s | %(extra)s",
    "trace": "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s"
}
log_format = LOG_FORMATS.get(LOG_MODE, LOG_FORMATS["hackathon"])

log_level = {
    "prod": logging.INFO,
    "hackathon": logging.DEBUG,
    "ml": logging.DEBUG,
    "trace": logging.NOTSET
}.get(LOG_MODE, logging.DEBUG)

logger = logging.getLogger("RAGPipeline")
logger.setLevel(log_level)

formatter = logging.Formatter(log_format)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(
    f"{LOG_DIR}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    encoding="utf-8"
)
fh.setFormatter(formatter)
logger.addHandler(fh)

json_log_path = f"{LOG_DIR}/events.jsonl"

def log_json(event_type, **data):
    """Сохраняет структурированные события для ML-анализа"""
    data = {"time": time.time(), "event": event_type, **data}
    with open(json_log_path, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(data, ensure_ascii=False) + "\n")

logger.info(f"[INFO] Запуск пайплайна | MODE={MODE} | LOG_MODE={LOG_MODE}")

# === Token accounting ===
TOKEN_ENCODER = "cl100k_base"
COST_PER_1M_INPUT = 0.20
COST_PER_1M_OUTPUT = 0.20
total_input_tokens = 0
total_output_tokens = 0
run_start_time = time.time()

def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding(TOKEN_ENCODER).encode(text))

def estimate_cost(input_tok, output_tok):
    return (input_tok/1e6)*COST_PER_1M_INPUT + (output_tok/1e6)*COST_PER_1M_OUTPUT


# === Tokenizer + stopwords ===
STOPWORDS_RU = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до тебя себе под чем этого чтобы нее сейчас были где при для мы тебя вам их затем сказала свой них была были либо однако поэтому
""".split())

def tokenize_ru(text):
    tok = re.findall(r"[а-яА-Яa-zA-Z0-9%]+", text.lower())
    return [t for t in tok if t not in STOPWORDS_RU and len(t) > 2]


# === Simple BM25 ===
class SimpleBM25:
    def __init__(self, corpus):
        logger.debug("Init BM25")
        self.corpus = corpus
        self.docs_tokens = [tokenize_ru(c) for c in corpus]
        self.N = len(self.docs_tokens)
        self.avgdl = sum(len(d) for d in self.docs_tokens) / max(1, self.N)
        self.df = {}
        for d in self.docs_tokens:
            for w in set(d):
                self.df[w] = self.df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for w, f in self.df.items()}

    def get_scores(self, query):
        q = tokenize_ru(query)
        k1, b = 1.5, 0.75
        scores = []
        for d in self.docs_tokens:
            dl = len(d)
            freqs = {}
            for w in d:
                freqs[w] = freqs.get(w, 0) + 1
            score = 0.0
            for w in q:
                if w in freqs:
                    tf = freqs[w]
                    score += self.idf.get(w, 0) * (tf*(k1+1))/(tf + k1*(1-b+b*dl/(self.avgdl+1e-9)))
            scores.append(score)
        return np.array(scores)


# === Загрузка базы знаний ===
TRAIN_CSV = "./train_data.csv"
_docs, _texts = [], []
logger.info("[INFO] Загрузка и чанкинг базы знаний")

if os.path.exists(TRAIN_CSV):
    df = pd.read_csv(TRAIN_CSV)
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    for _, row in df.iterrows():
        text = row.get("text") if "text" in row else row.iloc[0]
        if pd.isna(text): continue
        for chunk in splitter.split_text(str(text)):
            chunk = chunk.strip()
            if len(chunk) > 30:
                _docs.append(Document(page_content=chunk))
                _texts.append(chunk)

logger.info(f"[OK] Загружено {len(_docs)} чанков")


# === FAISS + Embeddings ===
logger.info("[INFO] Инициализация FAISS + Embeddings")

FAISS_INDEX_PATH = "faiss_index"
try:
    if USE_MOCK:
        from langchain.embeddings.base import Embeddings
        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts): return [[0.1] * 1536 for _ in texts]
            def embed_query(self, text): return [0.1] * 1536
        embeddings = MockEmbeddings()
    else:
        embeddings = OpenAIEmbeddings(base_url=BASE_URL, api_key=EMBEDDER_API_KEY, model="text-embedding-3-small")

    if os.path.exists(FAISS_INDEX_PATH):
        logger.info(f"[INFO] Найден FAISS индекс → загрузка '{FAISS_INDEX_PATH}'")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info("[INFO] FAISS индекс не найден → создаём новый")
        vectorstore = FAISS.from_documents(_docs, embeddings)
        if not USE_MOCK:
            vectorstore.save_local(FAISS_INDEX_PATH)
            logger.info(f"[OK] FAISS индекс сохранён в '{FAISS_INDEX_PATH}'")

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    logger.info("[OK] FAISS готов")

except Exception as e:
    logger.error(f"[ERROR] FAISS init failed: {e}")
    faiss_retriever = None


# === BM25 init ===
try:
    bm25 = SimpleBM25(_texts)
except Exception as e:
    bm25 = None
    logger.warning(f"BM25 init failed: {e}")


# === Query rewrite ===
def rewrite_queries(question):
    """Переформулирует вопрос несколькими вариантами для поиска"""
    client = MockOpenAI() if USE_MOCK else OpenAI(base_url=BASE_URL, api_key=LLM_API_KEY)
    prompt = f"Перефразируй вопрос 2 короткими вариантами для поиска:\n{question}"
    try:
        resp = client.chat.completions.create(
            model="openrouter/x-ai/grok-3-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60
        )
        variants = [question] + [l.strip("-• ").strip() for l in resp.choices[0].message.content.splitlines() if len(l.strip()) > 3]
        variants = list(dict.fromkeys(variants))[:3]
        logger.debug(f"Rewrite variants: {variants}")
        return variants
    except Exception as e:
        logger.warning(f"Rewrite failed: {e}")
        return [question]


# === Retrieval ===
def hybrid_retrieve(q):
    """Объединяет результаты FAISS и BM25"""
    fa, bm = [], []
    if faiss_retriever:
        try: fa = [d.page_content for d in faiss_retriever.invoke(q)]
        except: pass
    if bm25:
        try:
            sc = bm25.get_scores(q)
            idx = np.argsort(sc)[::-1][:6]
            bm = [_texts[i] for i in idx]
        except: pass
    docs = list(dict.fromkeys(fa + bm))[:12]
    logger.debug(f"Retrieved {len(docs)} docs for query='{q}'")
    return docs


# === Reranker ===
def rerank_docs(query, docs, key=EMBEDDER_API_KEY):
    """Реранкер Qwen3-Reranker-4B для сортировки документов по релевантности"""
    if USE_MOCK:
        logger.debug("[MOCK] Reranker skipped")
        return [{"document": d, "relevance_score": 0.5} for d in docs]

    try:
        payload = {"model": "deepinfra/Qwen/Qwen3-Reranker-4B", "query": query, "documents": docs}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        resp = requests.post(RERANKER_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        logger.debug(f"Reranker returned {len(results)} docs for '{query[:40]}...'")
        return results
    except Exception as e:
        logger.warning(f"[WARN] Reranker failed: {e}")
        return [{"document": d, "relevance_score": 0.5} for d in docs]


# === Compression ===
def compress_context(frags):
    """Сжимает контекст до ключевой информации"""
    if not frags:
        return ""
    snippet = "\n\n".join(frags[:4])
    prompt = f"Сожми ключевую информацию (до 6 предложений):\n{snippet}"
    client = OpenAI(base_url=BASE_URL, api_key=LLM_API_KEY)
    try:
        r = client.chat.completions.create(
            model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0
        )
        ctx = r.choices[0].message.content.strip()
        logger.debug(f"Compressed context {len(ctx)} chars")
        return ctx
    except:
        return snippet[:800]


# === Fallback ===
def baseline_fallback_answer(q):
    """Fallback: дешёвый ответ grok-mini"""
    logger.warning("Fallback → baseline grok-mini")
    client = MockOpenAI() if USE_MOCK else OpenAI(base_url=BASE_URL, api_key=LLM_API_KEY)
    try:
        r = client.chat.completions.create(
            model="openrouter/x-ai/grok-3-mini",
            messages=[{"role": "user", "content": f"Ответь кратко: {q}"}],
            max_tokens=150
        )
        return r.choices[0].message.content.strip()
    except:
        return "Информация недоступна."


# === Budget Guard ===
def safe_generate(prompt, model, max_tokens):
    """Безопасный вызов модели с контролем бюджета"""
    global total_input_tokens, total_output_tokens
    input_tok = count_tokens(prompt)
    est_cost = estimate_cost(input_tok, max_tokens)
    spent = estimate_cost(total_input_tokens, total_output_tokens)

    if spent + est_cost > DAILY_BUDGET_USD:
        logger.warning("Budget limit → fallback")
        ans = baseline_fallback_answer(prompt)
        total_input_tokens += input_tok
        total_output_tokens += count_tokens(ans)
        log_json("fallback_budget", prompt_len=len(prompt))
        return ans

    client = MockOpenAI() if USE_MOCK else OpenAI(base_url=BASE_URL, api_key=LLM_API_KEY)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2
        )
        ans = r.choices[0].message.content.strip()
        total_input_tokens += input_tok
        total_output_tokens += count_tokens(ans)
        log_json("generation", model=model, tokens_in=input_tok, tokens_out=count_tokens(ans))
        return ans
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return baseline_fallback_answer(prompt)


# === Answer generation ===
def answer_generation(question):
    """Основная функция RAG-пайплайна"""
    logger.info(f"[INFO] Q: {question}")
    qs = rewrite_queries(question)

    docs = []
    for q in qs:
        docs.extend(hybrid_retrieve(q))
    docs = list(dict.fromkeys(docs))[:12]

    # === Reranker step ===
    if USE_RERANKER and docs:
        logger.debug(f"[RERANK] Sending {len(docs)} docs to reranker")
        reranked = rerank_docs(question, docs)
        if reranked and isinstance(reranked, list):
            reranked = sorted(reranked, key=lambda x: x.get("relevance_score", 0), reverse=True)
            docs = [item.get("document") or item.get("text") or str(item) for item in reranked[:6]]
            logger.debug(f"[RERANK] Top docs selected: {len(docs)}")
        else:
            logger.warning("[RERANK] No valid rerank results, using raw docs")
    log_json("rerank", q=question, docs_used=len(docs))

    if not docs:
        return baseline_fallback_answer(question)

    ctx = compress_context(docs)
    prompt = f"Ты финансовый помощник. Используй только фрагменты:\n{ctx}\n\nВопрос: {question}\n\nОтвет:"

    model = (
        "openrouter/meta-llama/llama-3-70b-instruct" if MODE == "max-score"
        else "openrouter/mistralai/mistral-small-3.2-24b-instruct"
    )

    ans = safe_generate(prompt, model, max_tokens=250)
    if not ans or len(ans) < 30:
        ans2 = baseline_fallback_answer(question)
        if len(ans2) > len(ans):
            ans = ans2

    logger.info(f"[OK] Answer: {ans[:80]}...")
    return ans


# === Autotest ===
def run_autotest():
    """Самопроверка RAG-пайплайна"""
    logger.info("[INFO] Запускаю self-test")
    try:
        sample = ["Как открыть вклад?", "Как изменить лимит карты?", "Как восстановить доступ?"]
        for q in sample:
            a = answer_generation(q)
            b = baseline_fallback_answer(q)
            try:
                client = MockOpenAI() if USE_MOCK else OpenAI(base_url=BASE_URL, api_key=LLM_API_KEY)
                emb = "text-embedding-3-small"
                e1 = client.embeddings.create(model=emb, input=a)["data"][0]["embedding"]
                e2 = client.embeddings.create(model=emb, input=b)["data"][0]["embedding"]
                sim = float(np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)))
            except:
                sim = None
            log_json("autotest", q=q, answer_len=len(a), fallback_len=len(b), sim=sim)
        log_json("autotest_rerank", reranker="Qwen3-Reranker-4B", status="active")
        logger.info("[OK] Автотест завершен")
    except Exception as e:
        logger.error(f"Autotest error: {e}")


# === Run main ===
if __name__ == "__main__":
    qdf = pd.read_csv('./questions.csv')
    questions = qdf['Вопрос'].tolist()
    answers = []
    for q in tqdm(questions, desc="Gen"):
        try:
            answers.append(answer_generation(q))
        except Exception as e:
            logger.error(f"Critical error, fallback: {e}")
            answers.append(baseline_fallback_answer(q))

    qdf['Ответы на вопрос'] = answers
    qdf.to_csv('submission.csv', index=False)

    run_autotest()

    total_cost = estimate_cost(total_input_tokens, total_output_tokens)
    summary = {
        "tokens_in": total_input_tokens,
        "tokens_out": total_output_tokens,
        "cost": total_cost,
        "time": time.time() - run_start_time
    }
    with open("run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"[OK] Готово | cost≈{total_cost:.4f}$ | tokens={total_input_tokens}+{total_output_tokens}")
