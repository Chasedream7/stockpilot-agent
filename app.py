import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import streamlit as st
from bs4 import BeautifulSoup

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    nltk = None
    SentimentIntensityAnalyzer = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


FINVIZ_URL = "https://finviz.com/quote.ashx?t="
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
LOCAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TRACE_SCHEMA_VERSION = "1.0"
MAX_AGENT_STEPS = 10


class TraceLogger:
    """Append-only JSONL trace for replaying and evaluating one agent run."""

    def __init__(self, trace_dir=None, run_id=None):
        self.run_id = run_id or f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        base_dir = trace_dir or os.environ.get("STOCKPILOT_TRACE_DIR", "traces")
        self.trace_dir = Path(base_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"{self.run_id}.jsonl"
        self.started_at = time.perf_counter()
        self.events = []

    @staticmethod
    def _json_safe(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
            return value.isoformat()
        if isinstance(value, pd.DataFrame):
            return {
                "type": "dataframe",
                "rows": len(value),
                "columns": list(value.columns),
                "preview": TraceLogger._json_safe(value.head(5).reset_index().to_dict("records")),
            }
        if isinstance(value, dict):
            return {str(key): TraceLogger._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [TraceLogger._json_safe(item) for item in list(value)[:30]]
        return str(value)

    def record(
        self,
        event,
        *,
        step,
        agent,
        input_data=None,
        decision=None,
        tool_call=None,
        output=None,
        usage=None,
        duration_ms=None,
        error=None,
    ):
        entry = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
            "event": event,
            "step": step,
            "agent": agent,
            "input": self._json_safe(input_data),
            "decision": self._json_safe(decision),
            "tool_call": self._json_safe(tool_call),
            "output": self._json_safe(output),
            "usage": self._json_safe(usage or {}),
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
            "error": str(error) if error else None,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self.events.append(entry)
        return entry


def short_text(value, limit=700):
    """Keep trace prompts inspectable without turning each trace into a huge artifact."""
    if value is None:
        return None
    value = str(value)
    return value if len(value) <= limit else f"{value[:limit]}… [truncated]"

# These terms are only a no-key / local-embedding-failure fallback. Primary
# classification uses the local multilingual embedding model below.
RISK_KEYWORDS = {
    "监管/法律": [
        "regulation",
        "regulatory",
        "probe",
        "investigation",
        "lawsuit",
        "antitrust",
        "sec",
        "ban",
        "fine",
    ],
    "业绩/指引": [
        "earnings",
        "guidance",
        "revenue",
        "profit",
        "margin",
        "miss",
        "cut",
        "forecast",
        "outlook",
    ],
    "需求/竞争": [
        "demand",
        "competition",
        "rival",
        "market share",
        "slowdown",
        "pressure",
        "discount",
    ],
    "宏观/融资": [
        "rates",
        "inflation",
        "recession",
        "debt",
        "downgrade",
        "supply chain",
        "tariff",
    ],
}

RISK_CATEGORY_DESCRIPTIONS = {
    "监管/法律": "监管调查、反垄断、执法、诉讼、处罚、禁令、证券合规与法律责任风险",
    "业绩/指引": "财报、收入、利润率、盈利、业绩不及预期、管理层指引、预测下调与展望风险",
    "需求/竞争": "客户需求放缓、竞争加剧、竞品、市场份额流失、价格压力、折扣与产品替代风险",
    "宏观/融资": "利率、通胀、衰退、债务、信用、关税、供应链与宏观经济冲击风险",
}

DEFAULT_GOAL_PROFILE = {
    "key": "general_watch",
    "label": "综合持仓观察",
    "priority_categories": ["监管/法律", "业绩/指引", "需求/竞争", "宏观/融资", "价格波动"],
    "description": "综合查看新闻情绪、风险信号和市场快照。",
    "next_steps": [
        "把本次 memo 记录到投资日志，和下一次复盘结果做对比。",
        "优先核对 Risk Signals 中的新闻原文，再决定是否调整关注优先级。",
    ],
}

GOAL_PROFILES = [
    {
        "key": "risk_review",
        "label": "风险复盘",
        "keywords": ["风险", "负面", "下跌", "暴雷", "利空", "亏损", "risk", "downside", "negative"],
        "priority_categories": ["监管/法律", "业绩/指引", "宏观/融资", "价格波动"],
        "description": "优先找可能影响长线持仓的负面信号。",
        "next_steps": [
            "优先阅读负面标题和 Risk Signals，确认是否存在新风险而不是市场噪音。",
            "如果风险来自业绩或监管，补充财报、公告或监管文件后再更新投资日志。",
        ],
    },
    {
        "key": "hold_check",
        "label": "持仓续持观察",
        "keywords": ["持有", "持仓", "继续关注", "长线", "复盘", "hold", "long term", "watch"],
        "priority_categories": ["业绩/指引", "需求/竞争", "宏观/融资"],
        "description": "判断最近新闻是否改变原来的长期持仓假设。",
        "next_steps": [
            "把情绪变化和核心风险写入持仓日志，和原始买入逻辑做对照。",
            "如果没有出现高优先级风险，继续跟踪下一批新闻和财报信息。",
        ],
    },
    {
        "key": "catalyst_scan",
        "label": "利好催化观察",
        "keywords": ["机会", "利好", "上涨", "增长", "催化", "正面", "bullish", "growth", "upside"],
        "priority_categories": ["业绩/指引", "需求/竞争"],
        "description": "优先找需求、业绩、产品和市场份额相关的正面催化。",
        "next_steps": [
            "优先核对正面证据是否来自基本面改善，而不只是短期市场情绪。",
            "如果正面信号集中在业绩和需求，加入下一次估值或财报复盘清单。",
        ],
    },
    {
        "key": "earnings_review",
        "label": "财报/业绩观察",
        "keywords": ["财报", "业绩", "营收", "利润", "指引", "earnings", "revenue", "margin", "guidance"],
        "priority_categories": ["业绩/指引"],
        "description": "聚焦财报、营收、利润率和管理层指引相关信号。",
        "next_steps": [
            "把业绩相关标题和情绪变化单独记录，后续对照财报原文。",
            "重点确认市场关注的是一次性波动，还是长期盈利能力变化。",
        ],
    },
    {
        "key": "regulation_review",
        "label": "监管/法律观察",
        "keywords": ["监管", "诉讼", "调查", "法律", "合规", "regulation", "lawsuit", "probe", "legal"],
        "priority_categories": ["监管/法律"],
        "description": "聚焦监管、诉讼、调查和合规相关风险。",
        "next_steps": [
            "优先打开监管或诉讼相关新闻原文，确认事件的严重程度和时间线。",
            "如果同类信号连续出现，单独建立监管风险观察记录。",
        ],
    },
    {
        "key": "competition_review",
        "label": "需求/竞争观察",
        "keywords": ["竞争", "需求", "市场份额", "对手", "competition", "demand", "rival", "market share"],
        "priority_categories": ["需求/竞争"],
        "description": "聚焦需求变化、竞争压力和市场份额信号。",
        "next_steps": [
            "把需求和竞争相关标题与公司的长期增长假设做对照。",
            "如果竞争压力升高，下一步补充行业数据或竞品新闻。",
        ],
    },
]

HIGH_SIGNAL_PROFILE_RULES = {
    "regulation_review": [
        "监管",
        "反垄断",
        "诉讼",
        "调查",
        "合规",
        "法律",
        "sec",
        "公告",
        "regulation",
        "antitrust",
        "lawsuit",
        "investigation",
        "compliance",
        "filing",
    ],
    "earnings_review": [
        "财报",
        "业绩",
        "营收",
        "利润",
        "指引",
        "earnings",
        "revenue",
        "guidance",
        "margin",
    ],
    "competition_review": [
        "竞争",
        "需求",
        "市场份额",
        "对手",
        "competition",
        "demand",
        "rival",
        "market share",
    ],
}

POSITIVE_WORDS = {
    "accelerate",
    "accelerates",
    "beat",
    "beats",
    "bullish",
    "demand",
    "gain",
    "gains",
    "growth",
    "improve",
    "improves",
    "improving",
    "momentum",
    "outperform",
    "profit",
    "rally",
    "record",
    "rise",
    "rises",
    "rose",
    "stabilize",
    "strong",
    "stronger",
    "surge",
    "surges",
    "upgrade",
    "upgrades",
}

NEGATIVE_WORDS = {
    "bearish",
    "cautious",
    "competition",
    "cut",
    "debt",
    "decline",
    "declines",
    "downgrade",
    "falls",
    "fine",
    "investigation",
    "lawsuit",
    "loss",
    "macro",
    "miss",
    "mixed",
    "pressure",
    "probe",
    "questions",
    "regulatory",
    "risk",
    "risks",
    "slowdown",
    "slips",
    "uncertainty",
    "weak",
}


class LexiconSentimentAnalyzer:
    tool_name = "Built-in finance lexicon"

    def polarity_scores(self, text):
        tokens = re.findall(r"[a-z']+", text.lower())
        positive_hits = sum(token in POSITIVE_WORDS for token in tokens)
        negative_hits = sum(token in NEGATIVE_WORDS for token in tokens)
        signal = positive_hits + negative_hits

        if signal == 0:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        strength = min(1.0, signal / 4)
        compound = ((positive_hits - negative_hits) / signal) * strength
        pos = max(0.0, compound)
        neg = max(0.0, -compound)
        neu = max(0.0, 1 - pos - neg)
        return {
            "neg": round(neg, 3),
            "neu": round(neu, 3),
            "pos": round(pos, 3),
            "compound": round(compound, 3),
        }


def cache_resource(func):
    if hasattr(st, "cache_resource"):
        return st.cache_resource(func)
    return st.cache(allow_output_mutation=True)(func)


@cache_resource
def get_local_embedding_model():
    """Download once on the host, then reuse a free multilingual local encoder."""
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(LOCAL_EMBEDDING_MODEL)


def get_config_value(name, default=None):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_deepseek_client(api_key=None):
    """DeepSeek exposes an OpenAI-compatible Chat Completions API."""
    if OpenAI is None or not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def is_llm_capacity_error(error):
    """Whether an SDK error means this run should stop making paid API calls."""
    status_code = getattr(error, "status_code", None)
    error_type = type(error).__name__.lower()
    return status_code == 429 or "ratelimit" in error_type or "quota" in error_type


def llm_failure_notice(error):
    """A user-safe message; never echo provider error text or credentials."""
    if is_llm_capacity_error(error):
        return (
            "DeepSeek API 当前返回 429（请求限流或账户余额不足）。"
            "本轮已自动切换到规则基线，研究仍会完成；请检查 DeepSeek 账户余额、"
            "限流状态和所选模型，或等待限流窗口重置后重试。"
        )
    return "DeepSeek API 本轮不可用，已自动切换到规则基线，研究仍会完成。"


def disable_llm_for_run(state, error, trace, stage):
    """Trip a per-run circuit breaker after an API failure and leave an audit trail."""
    notice = llm_failure_notice(error)
    if state.get("llm_disabled_reason"):
        return state["llm_disabled_reason"]

    state["llm_disabled_reason"] = notice
    state.setdefault("warnings", []).append(notice)
    trace.record(
        "llm_runtime_disabled",
        step=state.get("tool_steps", 0),
        agent="Supervisor Agent",
        input_data={"stage": stage, "error_type": type(error).__name__},
        error=error,
    )
    return notice


@cache_resource
def get_sentiment_analyzer():
    if nltk is None or SentimentIntensityAnalyzer is None:
        return LexiconSentimentAnalyzer()

    nltk_data_dir = os.environ.get(
        "NLTK_DATA",
        os.path.join(tempfile.gettempdir(), "stockpilot_nltk_data"),
    )
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", download_dir=nltk_data_dir, quiet=True)

    try:
        analyzer = SentimentIntensityAnalyzer()
        analyzer.tool_name = "NLTK VADER"
        return analyzer
    except Exception:
        return LexiconSentimentAnalyzer()


def fetch_finviz_soup(ticker):
    url = FINVIZ_URL + ticker
    req = Request(
        url=url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=12) as response:
        html = response.read()
    return BeautifulSoup(html, "html.parser")


def sec_headers():
    """SEC asks automated clients to identify a contact address."""
    contact = get_config_value("STOCKPILOT_CONTACT_EMAIL", "research@example.com")
    return {
        "User-Agent": f"StockPilotX research prototype/0.1 {contact}",
    }


def fetch_json(url, headers, timeout=12):
    request = Request(url=url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_sec_filings(ticker, max_filings=8):
    """Return recent primary SEC filings for a US-listed ticker, with source URLs."""
    ticker = ticker.upper().strip()
    ticker_index = fetch_json(SEC_COMPANY_TICKERS_URL, sec_headers())
    company = next(
        (item for item in ticker_index.values() if item.get("ticker", "").upper() == ticker),
        None,
    )
    if not company:
        raise ValueError(f"SEC could not find a company mapping for ticker {ticker}.")

    cik = int(company["cik_str"])
    submissions = fetch_json(SEC_SUBMISSIONS_URL.format(cik=cik), sec_headers())
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filings = []
    supported_forms = {"10-K", "10-Q", "8-K", "20-F", "6-K"}
    for index, form in enumerate(forms):
        if form not in supported_forms:
            continue
        accession = recent.get("accessionNumber", [""])[index].replace("-", "")
        document = recent.get("primaryDocument", [""])[index]
        if not accession or not document:
            continue
        filings.append(
            {
                "form": form,
                "filed_at": recent.get("filingDate", [""])[index],
                "report_date": recent.get("reportDate", [""])[index],
                "items": recent.get("items", [""])[index],
                "document_url": SEC_ARCHIVES_URL.format(
                    cik=cik,
                    accession=accession,
                    document=document,
                ),
            }
        )
        if len(filings) >= max_filings:
            break

    if not filings:
        raise ValueError(f"SEC returned no recent supported filings for {ticker}.")

    return {
        "company_name": company.get("title", ticker),
        "cik": cik,
        "filings": filings,
        "source_note": "SEC EDGAR filings (primary source metadata)",
    }


def parse_snapshot(soup):
    tables = soup.find_all("table", class_="snapshot-table2")
    if not tables:
        return {}

    snapshot = {}
    for table in tables:
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
            for index in range(0, len(cells) - 1, 2):
                key = cells[index]
                value = cells[index + 1]
                if key:
                    snapshot[key] = value

    if "Change" not in snapshot and "Change %" in snapshot:
        snapshot["Change"] = snapshot["Change %"]

    return snapshot


def normalize_finviz_date(date_text):
    today = dt.date.today()
    if date_text == "Today":
        return today.strftime("%b-%d-%y")
    if date_text == "Yesterday":
        return (today - dt.timedelta(days=1)).strftime("%b-%d-%y")
    return date_text


def parse_news(news_table, max_headlines):
    if news_table is None:
        raise ValueError("FinViz news table was not found.")

    rows = []
    current_date = dt.date.today().strftime("%b-%d-%y")

    for row in news_table.find_all("tr"):
        link = row.find("a")
        timestamp = row.find("td")
        if not link or not timestamp:
            continue

        parts = timestamp.get_text(" ", strip=True).split()
        if len(parts) == 1:
            date_text = current_date
            time_text = parts[0]
        else:
            date_text = parts[0]
            current_date = date_text
            time_text = parts[1]

        normalized_date = normalize_finviz_date(date_text)
        published_at = pd.to_datetime(
            f"{normalized_date} {time_text}",
            errors="coerce",
        )
        if pd.isna(published_at):
            continue

        rows.append(
            {
                "datetime": published_at,
                "headline": link.get_text(" ", strip=True),
                "url": link.get("href", ""),
            }
        )

    if not rows:
        raise ValueError("No valid FinViz headlines were parsed.")

    return (
        pd.DataFrame(rows)
        .sort_values("datetime", ascending=False)
        .head(max_headlines)
        .reset_index(drop=True)
    )


def create_fallback_news(ticker, max_headlines):
    now = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    headlines = [
        f"{ticker} shares rise as analysts cite stronger demand and improving margins",
        f"{ticker} faces regulatory questions as investors weigh near-term risk",
        f"{ticker} earnings preview points to cautious guidance and mixed sentiment",
        f"Wall Street upgrades {ticker} after new product momentum accelerates",
        f"{ticker} slips as competition pressure and macro uncertainty remain in focus",
        f"Investors watch {ticker} cash flow, revenue growth and management commentary",
        f"{ticker} supplier data suggests demand may stabilize into the next quarter",
        f"Options traders price larger move for {ticker} ahead of key announcement",
    ]
    rows = [
        {
            "datetime": now - dt.timedelta(hours=index),
            "headline": headline,
            "url": "",
        }
        for index, headline in enumerate(headlines[:max_headlines])
    ]
    return pd.DataFrame(rows)


def score_news(news_df):
    vader = get_sentiment_analyzer()
    scores = news_df["headline"].apply(vader.polarity_scores).apply(pd.Series)
    scored = news_df.join(scores)
    scored = scored.rename(columns={"compound": "sentiment_score"})
    scored.attrs["sentiment_tool"] = getattr(vader, "tool_name", "Sentiment analyzer")

    def label(score):
        if score >= 0.12:
            return "Positive"
        if score <= -0.12:
            return "Negative"
        return "Neutral"

    scored["sentiment_label"] = scored["sentiment_score"].apply(label)
    return scored.set_index("datetime").sort_index()


def summarize_sentiment(scored_news):
    total = len(scored_news)
    label_counts = scored_news["sentiment_label"].value_counts().to_dict()
    avg_score = float(scored_news["sentiment_score"].mean()) if total else 0.0
    positive = int(label_counts.get("Positive", 0))
    negative = int(label_counts.get("Negative", 0))
    neutral = int(label_counts.get("Neutral", 0))
    bullishness = round(max(0, min(100, (avg_score + 1) * 50)))

    return {
        "total": total,
        "avg_score": avg_score,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "bullishness": bullishness,
        "negative_ratio": negative / total if total else 0,
    }


def parse_percent(raw_value):
    if not raw_value:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?%", raw_value)
    if not match:
        return None
    return float(match.group(0).replace("%", ""))


def infer_goal_profile(mission):
    text = mission.lower().strip()
    if not text:
        return DEFAULT_GOAL_PROFILE

    # High-signal topic words should beat generic words such as “风险” or
    # “复盘”.  This is especially important for short follow-ups like
    # “详细说说那个监管风险”: the user is narrowing the topic to regulation,
    # not starting another generic risk scan.
    for profile_key, signals in HIGH_SIGNAL_PROFILE_RULES.items():
        if any(signal in text for signal in signals):
            return profile_by_key(profile_key)

    best_profile = DEFAULT_GOAL_PROFILE
    best_score = 0
    for profile in GOAL_PROFILES:
        score = sum(1 for keyword in profile["keywords"] if keyword in text)
        if score > best_score:
            best_profile = profile
            best_score = score

    return best_profile


def profile_by_key(key):
    if key == DEFAULT_GOAL_PROFILE["key"]:
        return DEFAULT_GOAL_PROFILE
    for profile in GOAL_PROFILES:
        if profile["key"] == key:
            return profile
    return None


def goal_profile_options():
    profiles = [DEFAULT_GOAL_PROFILE] + GOAL_PROFILES
    return "\n".join(
        [
            (
                f"- key: {profile['key']} | label: {profile['label']} | "
                f"priority_categories: {format_categories(profile['priority_categories'])} | "
                f"description: {profile['description']}"
            )
            for profile in profiles
        ]
    )


def extract_json_object(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "candidates_token_count", None)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": getattr(usage, "total_tokens", getattr(usage, "total_token_count", None)),
    }


def cosine_similarity(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def embed_texts(texts, trace=None, agent="Embedding Router", step="embedding", mode="passage"):
    """Embed text locally; no API key, vectors, or raw model weights enter traces."""
    encoder = get_local_embedding_model()
    if encoder is None:
        return None
    prefix = "query: " if mode == "query" else "passage: "
    started_at = time.perf_counter()
    try:
        encoded = encoder.encode(
            [f"{prefix}{text}" for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
    except Exception as error:
        if trace:
            trace.record(
                "local_embedding_failed",
                step=step,
                agent=agent,
                input_data={"model": LOCAL_EMBEDDING_MODEL, "mode": mode, "text_count": len(texts)},
                error=error,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
        raise
    if trace:
        trace.record(
            "local_embedding",
            step=step,
            agent=agent,
            input_data={"model": LOCAL_EMBEDDING_MODEL, "mode": mode, "text_count": len(texts)},
            output={"vector_count": len(vectors), "dimensions": len(vectors[0]) if vectors else 0},
            duration_ms=(time.perf_counter() - started_at) * 1000,
        )
    return vectors


def call_llm_text(
    system_prompt,
    user_prompt,
    model,
    trace=None,
    agent="LLM",
    step="llm_call",
    api_key=None,
    json_mode=False,
):
    client = get_deepseek_client(api_key)
    if client is None:
        return None

    started_at = time.perf_counter()
    try:
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            **request,
        )
        output = response.choices[0].message.content
        if isinstance(output, list):
            output = "".join(
                item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                for item in output
            )
        if not output:
            raise RuntimeError("DeepSeek Chat Completions response did not contain text content.")
    except Exception as error:
        if trace:
            trace.record(
                "llm_call_failed",
                step=step,
                agent=agent,
                input_data={
                    "system_prompt": short_text(system_prompt),
                    "user_prompt": short_text(user_prompt),
                    "model": model,
                },
                error=error,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
        raise

    if trace:
        trace.record(
            "llm_call",
            step=step,
            agent=agent,
            input_data={
                "system_prompt": short_text(system_prompt),
                "user_prompt": short_text(user_prompt),
                "model": model,
            },
            output={"text": output},
            usage=extract_usage(response),
            duration_ms=(time.perf_counter() - started_at) * 1000,
        )
    return output


def call_llm_json(
    system_prompt,
    user_prompt,
    model,
    trace=None,
    agent="LLM",
    step="llm_json",
    api_key=None,
):
    output = call_llm_text(
        system_prompt,
        user_prompt,
        model,
        trace=trace,
        agent=agent,
        step=step,
        api_key=api_key,
        json_mode=True,
    )
    parsed = extract_json_object(output)
    if trace:
        trace.record(
            "structured_output_parsed" if parsed else "structured_output_invalid",
            step=step,
            agent=agent,
            output=parsed or {"raw_text": short_text(output)},
        )
    return parsed


def profile_embedding_text(profile):
    return (
        f"投资研究目标：{profile['label']}。{profile['description']}。"
        f"优先关注：{format_categories(profile['priority_categories'])}。"
    )


def infer_goal_profile_with_embedding(mission, trace=None):
    """Route an open-ended goal with a free local multilingual embedding model."""
    if not mission.strip():
        return None
    profiles = [DEFAULT_GOAL_PROFILE] + GOAL_PROFILES
    query_vectors = embed_texts(
        [mission],
        trace=trace,
        agent="PM Agent",
        step="goal_embedding_query",
        mode="query",
    )
    profile_vectors = embed_texts(
        [profile_embedding_text(profile) for profile in profiles],
        trace=trace,
        agent="PM Agent",
        step="goal_embedding_profiles",
        mode="passage",
    )
    if not query_vectors or not profile_vectors or len(profile_vectors) != len(profiles):
        return None
    similarities = [cosine_similarity(query_vectors[0], vector) for vector in profile_vectors]
    normalized_mission = mission.lower()
    explicit_matches = [
        sum(1 for keyword in profile.get("keywords", []) if keyword.lower() in normalized_mission)
        for profile in profiles
    ]
    # Keep embedding as the main signal, but let explicit high-signal intent words
    # break near-ties. This prevents generic words such as "长期" or "复盘" from
    # hiding a clear follow-up focus like "监管 / 反垄断 / 调查".
    adjusted_scores = [
        similarity + min(0.18, match_count * 0.06)
        for similarity, match_count in zip(similarities, explicit_matches)
    ]
    # Embeddings are the primary router, but short follow-ups need a
    # deterministic high-signal guardrail.  A phrase such as “监管风险” can
    # be closer to the generic risk profile in vector space; the concrete
    # regulatory topic must win that near-tie.
    high_signal_boosts = {
        "regulation_review": [
            "监管",
            "反垄断",
            "诉讼",
            "调查",
            "合规",
            "法律",
            "sec",
            "公告",
            "regulation",
            "antitrust",
            "lawsuit",
            "investigation",
            "compliance",
            "filing",
        ],
        "earnings_review": [
            "财报",
            "业绩",
            "营收",
            "利润",
            "指引",
            "earnings",
            "revenue",
            "guidance",
            "margin",
        ],
        "competition_review": [
            "竞争",
            "需求",
            "市场份额",
            "对手",
            "competition",
            "demand",
            "rival",
            "market share",
        ],
    }
    for index, candidate in enumerate(profiles):
        signals = high_signal_boosts.get(candidate["key"], [])
        signal_count = sum(1 for signal in signals if signal in normalized_mission)
        if signal_count:
            adjusted_scores[index] += min(0.3, signal_count * 0.15)
    best_index = max(range(len(profiles)), key=lambda index: adjusted_scores[index])
    profile = dict(profiles[best_index])
    alternatives = sorted(
        (
            {
                "profile_key": candidate["key"],
                "similarity": round(similarities[index], 3),
                "explicit_matches": explicit_matches[index],
                "adjusted_score": round(adjusted_scores[index], 3),
            }
            for index, candidate in enumerate(profiles)
        ),
        key=lambda item: item["adjusted_score"],
        reverse=True,
    )[:3]
    profile["analysis_source"] = "Local embedding semantic router"
    profile["llm_reason"] = (
        f"本地语义相似度 {similarities[best_index]:.3f}，"
        f"显式意图词 {explicit_matches[best_index]} 个，最接近“{profile['label']}”。"
    )
    if trace:
        trace.record(
            "semantic_match",
            step="goal_analysis",
            agent="PM Agent",
            input_data={"mission": mission, "embedding_model": LOCAL_EMBEDDING_MODEL},
            output={"selected_profile": profile["key"], "alternatives": alternatives},
        )
    return profile


def infer_goal_profile_with_llm(mission, model, trace=None, api_key=None):
    system_prompt = """
You are the goal analysis agent for StockPilot Agent.
Map the user's natural-language investing research goal to exactly one supported goal profile.
Return only compact JSON with keys: profile_key, reason.
Do not provide investment advice.
""".strip()
    user_prompt = f"""
User analysis goal:
{mission}

Supported goal profiles:
{goal_profile_options()}

Rules:
- Choose the closest profile_key from the supported list.
- reason must be Chinese, concise, and explain why this profile fits.
- Return JSON only.
""".strip()

    parsed = call_llm_json(
        system_prompt,
        user_prompt,
        model,
        trace=trace,
        agent="PM Agent",
        step="goal_analysis",
        api_key=api_key,
    )
    if not parsed:
        return None

    profile = profile_by_key(parsed.get("profile_key"))
    if not profile:
        return None

    enriched = dict(profile)
    enriched["analysis_source"] = "LLM"
    enriched["llm_reason"] = parsed.get("reason", "")
    return enriched


def analyze_goal(
    mission,
    use_llm,
    model,
    trace=None,
    api_key=None,
):
    if use_llm:
        # Resolve concrete topic changes before consulting a probabilistic
        # router.  This keeps short follow-ups deterministic even when the
        # local embedding model is unavailable or the provider returns a
        # generic profile.
        high_signal_profile = infer_goal_profile(mission)
        if high_signal_profile["key"] != DEFAULT_GOAL_PROFILE["key"]:
            high_signal_profile = dict(high_signal_profile)
            high_signal_profile["analysis_source"] = "High-signal intent guardrail"
            high_signal_profile["llm_reason"] = (
                "检测到明确的监管、财报或竞争主题词，优先切换到对应分析重点。"
            )
            return high_signal_profile, "High-signal intent router"
        try:
            profile = infer_goal_profile_with_embedding(mission, trace=trace)
            if profile:
                return profile, "Local embedding goal router"
        except Exception as error:
            if trace:
                trace.record(
                    "semantic_match_failed",
                    step="goal_analysis",
                    agent="PM Agent",
                    error=error,
                )
        try:
            profile = infer_goal_profile_with_llm(mission, model, trace=trace, api_key=api_key)
            if profile:
                return profile, "DeepSeek semantic goal router"
        except Exception as error:
            if trace:
                trace.record(
                    "semantic_match_failed",
                    step="goal_analysis",
                    agent="PM Agent",
                    error=error,
                )
            fallback = dict(infer_goal_profile(mission))
            fallback["analysis_source"] = "Rule fallback"
            fallback["llm_reason"] = "LLM 目标分析不可用，已使用规则分类。"
            fallback["llm_disabled_reason"] = llm_failure_notice(error)
            return fallback, "Rule fallback after LLM error"

    profile = dict(infer_goal_profile(mission))
    profile["analysis_source"] = "Rule fallback"
    if use_llm:
        profile["llm_reason"] = "未输入 DeepSeek API Key 或 openai SDK 不可用。"
        return profile, "Rule fallback because LLM is unavailable"
    profile["llm_reason"] = "LLM mode is off."
    return profile, "Rule-based goal parser"


def format_categories(categories):
    return "、".join(categories)


def build_next_steps(goal_profile):
    return "\n".join([f"- {step}" for step in goal_profile["next_steps"]])


def keyword_risk_findings(scored_news, priority_categories):
    """Deterministic fallback for offline runs and local embedding failures."""
    findings = []
    for _, row in scored_news.reset_index().iterrows():
        headline = row["headline"]
        lower_headline = headline.lower()
        for category, keywords in RISK_KEYWORDS.items():
            matched = [word for word in keywords if word in lower_headline]
            if matched:
                findings.append(
                    {
                        "category": category,
                        "headline": headline,
                        "matched_signal": ", ".join(matched[:3]),
                        "similarity": None,
                        "classification_source": "keyword fallback",
                        "sentiment_score": round(row["sentiment_score"], 3),
                        "focus_match": "Yes" if category in priority_categories else "No",
                    }
                )
    return findings


def semantic_risk_findings(scored_news, priority_categories, trace=None):
    """Map English headlines to the Chinese risk taxonomy with local embeddings."""
    headlines = scored_news.reset_index().to_dict("records")
    categories = list(RISK_CATEGORY_DESCRIPTIONS)
    category_vectors = embed_texts(
        list(RISK_CATEGORY_DESCRIPTIONS.values()),
        trace=trace,
        agent="Risk Agent",
        step="risk_embedding_categories",
        mode="passage",
    )
    headline_vectors = embed_texts(
        [row["headline"] for row in headlines],
        trace=trace,
        agent="Risk Agent",
        step="risk_embedding_headlines",
        mode="query",
    )
    if not category_vectors or not headline_vectors:
        return None
    findings = []
    for row, vector in zip(headlines, headline_vectors):
        similarities = [cosine_similarity(vector, category_vector) for category_vector in category_vectors]
        best_index = max(range(len(categories)), key=lambda index: similarities[index])
        best_score = similarities[best_index]
        if best_score < 0.35:
            continue
        category = categories[best_index]
        findings.append(
            {
                "category": category,
                "headline": row["headline"],
                "matched_signal": f"local semantic similarity {best_score:.3f}",
                "similarity": round(best_score, 3),
                "classification_source": "local embedding semantic classifier",
                "sentiment_score": round(row["sentiment_score"], 3),
                "focus_match": "Yes" if category in priority_categories else "No",
            }
        )
    return findings


def detect_risks(
    scored_news,
    snapshot,
    goal_profile,
    api_key=None,
    trace=None,
    use_semantic=False,
):
    priority_categories = set(goal_profile["priority_categories"])
    classifier = "local embedding semantic classifier"
    findings = None
    if use_semantic:
        try:
            findings = semantic_risk_findings(scored_news, priority_categories, trace=trace)
        except Exception as error:
            if trace:
                trace.record(
                    "semantic_match_failed",
                    step="risk_analysis",
                    agent="Risk Agent",
                    error=error,
                )
    if findings is None:
        classifier = "keyword fallback"
        findings = keyword_risk_findings(scored_news, priority_categories)

    change = parse_percent(snapshot.get("Change", ""))
    volatility_points = 0
    if change is not None and abs(change) >= 3:
        volatility_points = 18
        findings.append(
            {
                "category": "价格波动",
                "headline": f"FinViz shows intraday change of {change:.2f}%.",
                "matched_signal": "large price move",
                "similarity": None,
                "classification_source": "price-move rule",
                "sentiment_score": 0,
                "focus_match": "Yes" if "价格波动" in priority_categories else "No",
            }
        )

    negative_ratio = (
        float((scored_news["sentiment_label"] == "Negative").mean())
        if len(scored_news)
        else 0
    )
    risk_score = min(100, negative_ratio * 48 + len(findings) * 7 + volatility_points)

    if risk_score >= 60:
        level = "High"
    elif risk_score >= 30:
        level = "Medium"
    else:
        level = "Low"

    displayed_findings = sorted(
        findings,
        key=lambda item: (
            item["focus_match"] != "Yes",
            -(item["similarity"] or 0),
            item["sentiment_score"],
        ),
    )

    return {
        "level": level,
        "score": round(risk_score),
        "findings": displayed_findings[:8],
        "focus_matches": sum(
            1 for item in displayed_findings[:8] if item["focus_match"] == "Yes"
        ),
        "classifier": classifier,
    }


def choose_stance(summary, risk):
    avg_score = summary["avg_score"]
    level = risk["level"]

    if level == "High" or avg_score <= -0.18:
        return "谨慎观察：负面信号或风险密度偏高，先等待更多确认。"
    if avg_score >= 0.18 and level in {"Low", "Medium"}:
        return "积极观察：舆情偏正面，可加入重点跟踪清单。"
    return "中性观察：信号还不够一致，适合继续跟踪而不是立刻下结论。"


def top_headlines(scored_news, label, limit=3):
    rows = scored_news[scored_news["sentiment_label"] == label]
    if label == "Positive":
        rows = rows.sort_values("sentiment_score", ascending=False)
    elif label == "Negative":
        rows = rows.sort_values("sentiment_score", ascending=True)
    else:
        rows = rows.sort_values("sentiment_score")
    return rows.head(limit)["headline"].tolist()


def build_rule_memo(
    ticker,
    mission,
    goal_profile,
    summary,
    risk,
    snapshot,
    scored_news,
    source_note,
    is_follow_up=False,
):
    stance = choose_stance(summary, risk)
    positives = top_headlines(scored_news, "Positive")
    negatives = top_headlines(scored_news, "Negative")
    confidence = "High" if source_note.startswith("Live") and summary["total"] >= 12 else "Medium"
    if summary["total"] < 6:
        confidence = "Low"

    positive_lines = "\n".join([f"- {item}" for item in positives]) or "- 暂无明显正面标题。"
    negative_lines = "\n".join([f"- {item}" for item in negatives]) or "- 暂无明显负面标题。"
    next_step_lines = build_next_steps(goal_profile)
    snapshot_bits = [
        f"Price: {snapshot.get('Price', 'N/A')}",
        f"Change: {snapshot.get('Change', 'N/A')}",
        f"Market Cap: {snapshot.get('Market Cap', 'N/A')}",
        f"P/E: {snapshot.get('P/E', 'N/A')}",
    ]

    followup_block = (
        f"""
**本轮追问回答**
- 本轮问题：{mission}
- 回答基于上一轮已保留的新闻、情绪、风险和 SEC 证据。
- 本轮新增重点：请结合上面的证据区，直接回答本轮问题；若没有新增证据，明确说明“本轮未新增来源”。
"""
        if is_follow_up
        else ""
    )

    return f"""
### StockPilot Agent Memo: {ticker}

**用户目标**：{mission}

**识别到的分析重点**：{goal_profile['label']}  
{goal_profile['description']}
{followup_block}

**Agent 结论**：{stance}

**关键读数**
- 数据来源：{source_note}
- 情绪均值：{summary['avg_score']:.3f}
- Bullishness：{summary['bullishness']} / 100
- 风险等级：{risk['level']} ({risk['score']} / 100)
- 置信度：{confidence}
- 市场快照：{'; '.join(snapshot_bits)}

**正面证据**
{positive_lines}

**负面/风险证据**
{negative_lines}

**下一步建议**
{next_step_lines}

> 说明：本应用用于信息整理与自我复盘，不构成投资建议。
""".strip()


def compact_headlines(scored_news, limit=10):
    rows = (
        scored_news.reset_index()
        .sort_values("datetime", ascending=False)
        .head(limit)
    )
    return [
        {
            "datetime": row["datetime"].strftime("%Y-%m-%d %H:%M"),
            "headline": row["headline"],
            "sentiment_label": row["sentiment_label"],
            "sentiment_score": round(float(row["sentiment_score"]), 3),
        }
        for _, row in rows.iterrows()
    ]


def build_llm_memo(
    ticker,
    mission,
    goal_profile,
    summary,
    risk,
    snapshot,
    scored_news,
    source_note,
    model,
    sec_filings=None,
    trace=None,
    api_key=None,
    conversation_history=None,
    is_follow_up=False,
):
    system_prompt = """
You are Portfolio Copilot inside StockPilot Agent.
Write a concise Chinese holding-observation memo for a long-term individual investor.
Use only the supplied metrics, headlines, and risk findings.
Do not invent news, prices, financial facts, or future predictions.
Do not provide direct buy/sell/hold instructions.
Frame conclusions as observation and review guidance, not investment advice.
If this is a follow-up question, answer the latest question first, explicitly state
what changed from the previous memo, and do not simply repeat the previous memo.
If no new source was retrieved, say so clearly and tie the answer to retained evidence.
""".strip()

    payload = {
        "ticker": ticker,
        "user_goal": mission,
        "goal_profile": {
            "label": goal_profile["label"],
            "description": goal_profile["description"],
            "priority_categories": goal_profile["priority_categories"],
            "source": goal_profile.get("analysis_source", "unknown"),
            "reason": goal_profile.get("llm_reason", ""),
        },
        "source_note": source_note,
        "metrics": {
            "avg_sentiment": round(summary["avg_score"], 3),
            "bullishness": summary["bullishness"],
            "positive_headlines": summary["positive"],
            "neutral_headlines": summary["neutral"],
            "negative_headlines": summary["negative"],
            "risk_level": risk["level"],
            "risk_score": risk["score"],
        },
        "market_snapshot": snapshot,
        "risk_findings": risk["findings"],
        "recent_headlines": compact_headlines(scored_news),
        "sec_filings": sec_filings or {},
        "conversation_history": compact_conversation(conversation_history),
        "is_follow_up": is_follow_up,
    }
    user_prompt = f"""
Create the memo in Markdown with these sections:
1. StockPilot Agent Memo: {ticker}
2. User Goal
3. Goal Focus
4. Follow-up Answer (only when is_follow_up=true)
5. Observation
6. Key Readings
7. Evidence
8. Risks To Review
9. Next Steps
10. Disclaimer

Data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    return call_llm_text(
        system_prompt,
        user_prompt,
        model,
        trace=trace,
        agent="Portfolio Copilot",
        step="memo_generation",
        api_key=api_key,
    )


def build_memo(
    ticker,
    mission,
    goal_profile,
    summary,
    risk,
    snapshot,
    scored_news,
    source_note,
    use_llm,
    model,
    sec_filings=None,
    trace=None,
    api_key=None,
    conversation_history=None,
    is_follow_up=False,
):
    if use_llm:
        try:
            memo = build_llm_memo(
                ticker,
                mission,
                goal_profile,
                summary,
                risk,
                snapshot,
                scored_news,
                source_note,
                model,
                sec_filings=sec_filings,
                trace=trace,
                api_key=api_key,
                conversation_history=conversation_history,
                is_follow_up=is_follow_up,
            )
            if memo:
                return memo.strip(), "LLM memo generator"
        except Exception:
            pass

    memo = build_rule_memo(
        ticker,
        mission,
        goal_profile,
        summary,
        risk,
        snapshot,
        scored_news,
        source_note,
        is_follow_up=is_follow_up,
    )
    if use_llm:
        return memo, "Rule-based memo fallback"
    return memo, "Rule-based decision writer"


def plot_sentiment_timeline(scored_news, ticker):
    timeline = (
        scored_news["sentiment_score"]
        .resample("h")
        .mean()
        .dropna()
        .reset_index()
    )
    fig = px.bar(
        timeline,
        x="datetime",
        y="sentiment_score",
        title=f"{ticker} Hourly Sentiment",
        color="sentiment_score",
        color_continuous_scale=["#B42318", "#F6C177", "#027A48"],
        range_color=[-1, 1],
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_sentiment_mix(scored_news):
    counts = (
        scored_news["sentiment_label"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="headlines")
    )
    fig = px.pie(
        counts,
        names="sentiment",
        values="headlines",
        hole=0.55,
        color="sentiment",
        color_discrete_map={
            "Positive": "#027A48",
            "Neutral": "#667085",
            "Negative": "#B42318",
        },
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=30, b=20))
    return fig


AGENT_TOOLS = {
    "collect_news": {
        "agent": "News Scout Agent",
        "label": "FinViz news + market snapshot",
        "when": "需要近期市场叙事、标题证据或价格快照时。",
    },
    "fetch_sec_filings": {
        "agent": "Evidence Agent",
        "label": "SEC EDGAR filing metadata",
        "when": "财报、监管或管理层披露类问题需要一手来源，或 Critic 要求补证据时。",
    },
    "analyze_sentiment": {
        "agent": "Quant Agent",
        "label": "Sentiment scorer",
        "when": "已经获得新闻，且需要量化近期叙事倾向时。",
    },
    "assess_risk": {
        "agent": "Risk Agent",
        "label": "Risk signal scanner",
        "when": "已经获得带情绪的新闻，需要按用户目标归纳风险时。",
    },
    "draft_memo": {
        "agent": "Portfolio Copilot",
        "label": "Evidence-grounded memo writer",
        "when": "已有足够的新闻、情绪和风险摘要，需要回答用户问题时。",
    },
    "self_check": {
        "agent": "Critic Agent",
        "label": "Answer completeness check",
        "when": "memo 初稿完成后，检查是否回答问题、是否有证据和是否应补检索。",
    },
}


def compact_conversation(messages, max_messages=6, max_chars=900):
    """Keep the latest conversation turns available to the planner without unbounded prompts."""
    compacted = []
    for message in (messages or [])[-max_messages:]:
        role = message.get("role", "user")
        content = short_text(message.get("content", ""), max_chars)
        compacted.append({"role": role, "content": content})
    return compacted


def tool_statuses(state):
    """Expose the complete manifest while keeping data dependencies explicit."""
    critic = state.get("critic") or {}
    needs_unfetched_primary = critic.get("should_retrieve") and not state["sec_attempted"]
    return {
        "collect_news": {
            "available": state["news_df"] is None,
            "precondition": "没有可用新闻证据。",
        },
        "fetch_sec_filings": {
            "available": not state["sec_attempted"],
            "precondition": "尚未尝试 SEC 一手披露检索。",
        },
        "analyze_sentiment": {
            "available": state["news_df"] is not None and state["scored_news"] is None,
            "precondition": "已有新闻、尚未完成情绪量化。",
        },
        "assess_risk": {
            "available": state["scored_news"] is not None and state["risk"] is None,
            "precondition": "已有情绪新闻、尚未完成风险归纳。",
        },
        "draft_memo": {
            "available": state["risk"] is not None
            and (state["memo"] is None or state["revision_requested"])
            and not needs_unfetched_primary,
            "precondition": "已有风险结论，且没有待补的一手来源。",
        },
        "self_check": {
            "available": state["memo"] is not None
            and (
                state["pending_followup_review"]
                or state["reviewed_memo_version"] != state["memo_version"]
            ),
            "precondition": "已有 memo，且当前版本未审查或用户发起了追问。",
        },
        "finish": {
            "available": state["memo"] is not None
            and not state["pending_followup_review"]
            and state["reviewed_memo_version"] == state["memo_version"]
            and not needs_unfetched_primary,
            "precondition": "当前 memo 已审查，且没有可执行的补证据请求。",
        },
    }


def mission_needs_primary_source(mission):
    keywords = [
        "财报",
        "业绩",
        "营收",
        "利润",
        "指引",
        "监管",
        "诉讼",
        "反垄断",
        "调查",
        "公告",
        "earnings",
        "revenue",
        "guidance",
        "regulation",
        "lawsuit",
        "antitrust",
        "probe",
        "investigation",
        "filing",
    ]
    mission = mission.lower()
    return any(keyword in mission for keyword in keywords)


def planner_state(state):
    summary = state.get("summary") or {}
    risk = state.get("risk") or {}
    critic = state.get("critic") or {}
    return {
        "ticker": state["ticker"],
        "mission": state["mission"],
        "goal_focus": state["goal_profile"]["label"],
        "news_collected": state["news_df"] is not None,
        "headline_count": summary.get("total", 0),
        "sentiment_ready": state["scored_news"] is not None,
        "risk_ready": state["risk"] is not None,
        "risk_level": risk.get("level"),
        "sec_filing_attempted": state["sec_attempted"],
        "sec_filing_count": len((state.get("sec_filings") or {}).get("filings", [])),
        "memo_version": state["memo_version"],
        "memo_reviewed": state["reviewed_memo_version"] == state["memo_version"],
        "critic_status": critic.get("status"),
        "critic_requests_primary_source": critic.get("should_retrieve", False),
        "is_follow_up": state["is_follow_up"],
        "pending_followup_review": state["pending_followup_review"],
        "conversation_turns": len(state["messages"]),
        "retained_evidence": {
            "news": state["news_df"] is not None,
            "risk": state["risk"] is not None,
            "primary_source": bool((state.get("sec_filings") or {}).get("filings")),
        },
        "steps_remaining": MAX_AGENT_STEPS - state["tool_steps"],
    }


def valid_actions(state):
    statuses = tool_statuses(state)
    if state["pending_followup_review"]:
        # A follow-up must first inspect the old answer/evidence before it can re-retrieve.
        return ["self_check"]
    return [name for name, status in statuses.items() if status["available"]]


def rule_next_action(state):
    """Safe deterministic fallback when an LLM is unavailable or emits an invalid action."""
    if state["pending_followup_review"]:
        return "self_check", "用户正在追问，先检查历史回答和已有证据的缺口。"
    if state["news_df"] is None:
        return "collect_news", "缺少新闻证据，先建立可分析的事实基础。"
    if (state.get("critic") or {}).get("should_retrieve") and not state["sec_attempted"]:
        return "fetch_sec_filings", "Critic 要求补一手披露证据。"
    if state["scored_news"] is None:
        return "analyze_sentiment", "新闻已取得，需先量化叙事方向。"
    if state["risk"] is None:
        return "assess_risk", "情绪已量化，需映射到用户关注的风险维度。"
    if state["memo"] is None or state["revision_requested"]:
        return "draft_memo", "证据已就绪，需要生成或修订面向用户的问题回答。"
    if state["reviewed_memo_version"] != state["memo_version"]:
        return "self_check", "初稿还没有经过完整性和证据检查。"
    return "finish", "当前 memo 已完成检查，且没有可执行的补证据请求。"


def decide_next_action(state, use_llm, model, trace, api_key=None):
    fallback_action, fallback_reason = rule_next_action(state)
    available = valid_actions(state)
    if fallback_action not in available:
        fallback_action = available[0]
    if not use_llm or get_deepseek_client(api_key) is None:
        decision = {
            "action": fallback_action,
            "reason": fallback_reason,
            "decision_source": "rule fallback",
        }
        trace.record(
            "planner_decision",
            step=state["tool_steps"] + 1,
            agent="Supervisor Agent",
            input_data={"state": planner_state(state), "available_actions": available},
            decision=decision,
        )
        return decision

    statuses = tool_statuses(state)
    tool_text = "\n".join(
        (
            f"- {name} ({AGENT_TOOLS[name]['agent']}): {AGENT_TOOLS[name]['when']} "
            f"| precondition: {status['precondition']} | available_now: {status['available']}"
        )
        for name, status in statuses.items()
        if name in AGENT_TOOLS
    )
    tool_text += (
        f"\n- finish (Supervisor Agent): end the run only after review "
        f"| precondition: {statuses['finish']['precondition']} "
        f"| available_now: {statuses['finish']['available']}"
    )
    system_prompt = """
You are the Supervisor Agent for an evidence-grounded stock research workflow.
Choose exactly one next action from the available actions. Your job is to adapt the
path to the user's goal and the evidence state, not to follow a fixed checklist.
Never provide investment advice. A short reason is a decision justification, not
private chain-of-thought. Return JSON only: {"action": "...", "reason": "..."}.
""".strip()
    user_prompt = f"""
Current workflow state:
{json.dumps(planner_state(state), ensure_ascii=False, indent=2)}

Conversation history (latest turns):
{json.dumps(compact_conversation(state['messages']), ensure_ascii=False, indent=2)}

Complete tool manifest:
{tool_text}

Rules:
- You may reason over every listed tool. Choose exactly one action with `available_now: true`.
- On a fresh run, choose the evidence source that best fits the mission: news and SEC are both valid first actions when available.
- On a follow-up with a prior memo, choose `self_check` first; use the conversation and retained evidence to decide whether retrieval is actually missing.
- Do not restart evidence collection merely because the user asks a follow-up. Prefer the retained evidence unless Critic identifies a gap.
""".strip()
    try:
        parsed = call_llm_json(
            system_prompt,
            user_prompt,
            model,
            trace=trace,
            agent="Supervisor Agent",
            step=f"planner_{state['tool_steps'] + 1}",
            api_key=api_key,
        )
    except Exception as error:
        disable_llm_for_run(state, error, trace, "supervisor_planner")
        decision = {
            "action": fallback_action,
            "reason": fallback_reason,
            "decision_source": "LLM error fallback",
        }
        trace.record(
            "planner_decision",
            step=state["tool_steps"] + 1,
            agent="Supervisor Agent",
            input_data={"state": planner_state(state), "available_actions": available},
            decision=decision,
        )
        return decision
    action = (parsed or {}).get("action")
    reason = (parsed or {}).get("reason")
    if action not in available:
        trace.record(
            "planner_decision_rejected",
            step=state["tool_steps"] + 1,
            agent="Supervisor Agent",
            input_data={"eligible_actions": available, "tool_manifest": statuses},
            decision={"proposed_action": action, "reason": reason},
            error="LLM proposed an unavailable action; safe fallback applied.",
        )
        action, reason = fallback_action, fallback_reason
        source = "LLM invalid-output fallback"
    else:
        source = "LLM planner"
    decision = {"action": action, "reason": short_text(reason, 240), "decision_source": source}
    trace.record(
        "planner_decision",
        step=state["tool_steps"] + 1,
        agent="Supervisor Agent",
        input_data={
            "state": planner_state(state),
            "eligible_actions": available,
            "tool_manifest": statuses,
        },
        decision=decision,
    )
    return decision


def fallback_critic_check(state):
    has_primary_source = bool((state.get("sec_filings") or {}).get("filings"))
    needs_primary = mission_needs_primary_source(state["mission"]) and not has_primary_source
    has_live_evidence = state["source_note"].startswith("Live")
    evidence_score = 75 if has_live_evidence else 45
    if has_primary_source:
        evidence_score = min(95, evidence_score + 15)
    return {
        "status": "needs_more_evidence" if needs_primary else "pass",
        "answer_score": 72 if state["memo"] else 0,
        "evidence_score": evidence_score,
        "should_retrieve": needs_primary,
        "recommended_action": "fetch_sec_filings" if needs_primary else "finish",
        "missing_evidence": ["近期 SEC 披露或公告"] if needs_primary else [],
        "contradictions": [],
        "reason": (
            "用户问题涉及业绩/监管，但尚未成功取得一手披露来源。"
            if needs_primary
            else "规则检查未发现可自动判断的缺口；仍建议人工核对重要结论。"
        ),
        "predicted_user_followup": "ask_for_evidence" if needs_primary else "accept",
        "judge_source": "rule fallback",
    }


def self_check_memo(state, use_llm, model, trace, api_key=None):
    if not use_llm or get_deepseek_client(api_key) is None:
        return fallback_critic_check(state)

    system_prompt = """
You are the Critic Agent for a financial research assistant. Evaluate only whether
the memo answers the user's research question with the supplied evidence. Do not
give investment advice and do not invent facts. Return JSON only with: status
(pass|needs_more_evidence), answer_score (0-100), evidence_score (0-100),
should_retrieve (boolean), recommended_action (fetch_sec_filings|finish),
missing_evidence (array), contradictions (array), reason (concise Chinese).
predicted_user_followup (accept|ask_for_evidence|retry).
Request SEC filings only when the question or draft makes a material earnings,
regulatory, filing, or management-disclosure claim that headlines alone cannot support.
Use the conversation history to resolve references such as "that regulatory risk".
""".strip()
    evidence = {
        "mission": state["mission"],
        "source_note": state["source_note"],
        "headline_count": state["summary"]["total"],
        "risk": state["risk"],
        "sec_filings": state.get("sec_filings") or {},
        "memo": state["memo"],
        "conversation_history": compact_conversation(state["messages"]),
    }
    try:
        parsed = call_llm_json(
            system_prompt,
            json.dumps(evidence, ensure_ascii=False, indent=2),
            model,
            trace=trace,
            agent="Critic Agent",
            step=f"self_check_{state['memo_version']}",
            api_key=api_key,
        )
    except Exception as error:
        disable_llm_for_run(state, error, trace, "critic_self_check")
        fallback = fallback_critic_check(state)
        fallback["judge_source"] = "Rule fallback after LLM error"
        return fallback
    if not parsed or parsed.get("status") not in {"pass", "needs_more_evidence"}:
        fallback = fallback_critic_check(state)
        fallback["judge_source"] = "LLM invalid-output fallback"
        return fallback

    should_retrieve = bool(parsed.get("should_retrieve"))
    recommended = parsed.get("recommended_action")
    if recommended not in {"fetch_sec_filings", "finish"}:
        recommended = "fetch_sec_filings" if should_retrieve else "finish"
    followup = parsed.get("predicted_user_followup")
    if followup not in {"accept", "ask_for_evidence", "retry"}:
        followup = "ask_for_evidence" if should_retrieve else "accept"
    has_primary_source = bool((state.get("sec_filings") or {}).get("filings"))
    if mission_needs_primary_source(state["mission"]) and not has_primary_source:
        should_retrieve = True
        recommended = "fetch_sec_filings"
        followup = "ask_for_evidence"
        parsed["status"] = "needs_more_evidence"
        parsed["reason"] = (
            f"{parsed.get('reason', '')} 证据护栏：该问题需要成功取得一手披露来源。"
        ).strip()
    return {
        "status": parsed["status"],
        "answer_score": max(0, min(100, int(parsed.get("answer_score", 0)))),
        "evidence_score": max(0, min(100, int(parsed.get("evidence_score", 0)))),
        "should_retrieve": should_retrieve,
        "recommended_action": recommended,
        "missing_evidence": parsed.get("missing_evidence", [])[:5],
        "contradictions": parsed.get("contradictions", [])[:5],
        "reason": short_text(parsed.get("reason", ""), 500),
        "predicted_user_followup": followup,
        "judge_source": "LLM-as-judge",
    }


def execute_tool(
    action,
    state,
    use_llm,
    model,
    trace,
    api_key=None,
):
    step_number = state["tool_steps"] + 1
    agent = AGENT_TOOLS[action]["agent"]
    started_at = time.perf_counter()
    trace.record(
        "tool_started",
        step=step_number,
        agent=agent,
        input_data=planner_state(state),
        tool_call={"name": action},
    )
    try:
        if action == "collect_news":
            try:
                soup = fetch_finviz_soup(state["ticker"])
                state["snapshot"] = parse_snapshot(soup)
                state["news_df"] = parse_news(soup.find(id="news-table"), state["max_headlines"])
                state["source_note"] = "Live FinViz data"
                detail = f"抓取到 {len(state['news_df'])} 条最近新闻，并提取页面市场快照。"
                tool_label = "FinViz scraper"
            except Exception as error:
                if not state["allow_fallback_data"]:
                    raise
                state["source_note"] = f"Offline sample data; live fetch failed: {error}"
                state["news_df"] = create_fallback_news(state["ticker"], state["max_headlines"])
                detail = "实时网页抓取失败，已切换到离线样例数据；请不要把该结果作为当天行情判断。"
                tool_label = "Offline sample dataset"
            output = {"headlines": state["news_df"], "snapshot": state["snapshot"], "source_note": state["source_note"]}

        elif action == "fetch_sec_filings":
            state["sec_attempted"] = True
            try:
                state["sec_filings"] = fetch_sec_filings(state["ticker"])
                count = len(state["sec_filings"]["filings"])
                if state["is_follow_up"]:
                    state["revision_requested"] = True
                detail = f"取得 {count} 条近期 SEC 申报元数据，可作为一手来源入口。"
                tool_label = "SEC EDGAR filing metadata"
                output = state["sec_filings"]
            except Exception as error:
                state["sec_filings"] = {"filings": [], "error": str(error)}
                # No new evidence was obtained, so do not rewrite the same memo in a loop.
                state["revision_requested"] = state["is_follow_up"]
                detail = "SEC 披露补检索失败，已记录失败原因；继续基于已有证据生成观察结果。"
                tool_label = "SEC EDGAR (failed)"
                output = state["sec_filings"]

        elif action == "analyze_sentiment":
            state["scored_news"] = score_news(state["news_df"])
            state["summary"] = summarize_sentiment(state["scored_news"])
            detail = (
                f"完成 {state['summary']['total']} 条标题打分：Positive {state['summary']['positive']}，"
                f"Neutral {state['summary']['neutral']}，Negative {state['summary']['negative']}。"
            )
            tool_label = state["scored_news"].attrs.get("sentiment_tool", "Sentiment analyzer")
            output = state["summary"]

        elif action == "assess_risk":
            state["risk"] = detect_risks(
                state["scored_news"],
                state["snapshot"],
                state["goal_profile"],
                api_key=api_key,
                trace=trace,
                use_semantic=use_llm,
            )
            detail = (
                f"识别到 {len(state['risk']['findings'])} 个风险信号，其中 "
                f"{state['risk']['focus_matches']} 个与当前分析重点直接相关；综合风险等级为 {state['risk']['level']}。"
            )
            tool_label = state["risk"]["classifier"]
            output = state["risk"]

        elif action == "draft_memo":
            memo, tool_label = build_memo(
                state["ticker"],
                state["mission"],
                state["goal_profile"],
                state["summary"],
                state["risk"],
                state["snapshot"],
                state["scored_news"],
                state["source_note"],
                use_llm,
                model,
                sec_filings=state.get("sec_filings"),
                trace=trace,
                api_key=api_key,
                conversation_history=state["messages"],
                is_follow_up=state["is_follow_up"],
            )
            state["memo"] = memo
            state["memo_tool"] = tool_label
            state["memo_version"] += 1
            state["revision_requested"] = False
            detail = f"已生成第 {state['memo_version']} 版结构化 memo，等待 Critic 检查。"
            output = {"memo_version": state["memo_version"], "memo": memo, "generator": tool_label}

        elif action == "self_check":
            was_followup_review = state["pending_followup_review"]
            state["critic"] = self_check_memo(
                state,
                use_llm,
                model,
                trace,
                api_key=api_key,
            )
            state["reviewed_memo_version"] = state["memo_version"]
            state["pending_followup_review"] = False
            if was_followup_review:
                # Every follow-up must produce a new answer. If a missing SEC source
                # is requested, fetch it first; otherwise revise immediately from memory.
                state["revision_requested"] = not (
                    state["critic"].get("should_retrieve") and not state["sec_attempted"]
                )
            else:
                state["revision_requested"] = (
                    bool(state["critic"].get("should_retrieve")) and not state["sec_attempted"]
                )
            detail = (
                f"Critic 评估：{state['critic']['status']}，"
                f"答案分 {state['critic']['answer_score']}，证据分 {state['critic']['evidence_score']}。"
            )
            tool_label = state["critic"]["judge_source"]
            output = state["critic"]

        else:
            raise ValueError(f"Unknown agent tool: {action}")

        duration_ms = (time.perf_counter() - started_at) * 1000
        trace.record(
            "tool_completed",
            step=step_number,
            agent=agent,
            tool_call={"name": action, "label": tool_label},
            output=output,
            duration_ms=duration_ms,
        )
        state["tool_steps"] += 1
        return {
            "agent": agent,
            "tool": tool_label,
            "output": detail,
            "raw_output": output,
            "action": action,
        }
    except Exception as error:
        trace.record(
            "tool_failed",
            step=step_number,
            agent=agent,
            tool_call={"name": action},
            error=error,
            duration_ms=(time.perf_counter() - started_at) * 1000,
        )
        raise


def build_conversation_memory(state):
    """Keep structured evidence in Streamlit session state for follow-up questions."""
    fields = [
        "ticker",
        "mission",
        "goal_profile",
        "previous_goal_profile",
        "goal_changed",
        "source_note",
        "snapshot",
        "news_df",
        "scored_news",
        "summary",
        "risk",
        "sec_filings",
        "sec_attempted",
        "memo",
        "memo_tool",
        "memo_version",
        "reviewed_memo_version",
        "critic",
    ]
    return {field: state.get(field) for field in fields}


def run_workflow(
    ticker,
    mission,
    max_headlines,
    allow_fallback_data,
    use_llm=False,
    llm_model=DEFAULT_LLM_MODEL,
    trace_dir=None,
    api_key=None,
    prior_memory=None,
    messages=None,
):
    """Run an inspectable supervisor loop instead of a fixed, always-five-step pipeline."""
    prior_memory = prior_memory if (prior_memory or {}).get("ticker") == ticker else None
    is_follow_up = bool(prior_memory and prior_memory.get("memo"))
    messages = compact_conversation(messages)
    trace = TraceLogger(trace_dir=trace_dir)
    trace.record(
        "run_started",
        step=0,
        agent="Supervisor Agent",
        input_data={
            "ticker": ticker,
            "mission": mission,
            "max_headlines": max_headlines,
            "allow_fallback_data": allow_fallback_data,
            "llm_enabled": use_llm,
            "llm_model": llm_model,
            "is_follow_up": is_follow_up,
            "conversation_turns": len(messages),
        },
    )
    goal_started_at = time.perf_counter()
    goal_profile, goal_tool = analyze_goal(
        mission,
        use_llm,
        llm_model,
        trace=trace,
        api_key=api_key,
    )
    previous_goal_profile = (prior_memory or {}).get("goal_profile")
    goal_changed = bool(
        previous_goal_profile
        and previous_goal_profile.get("key") != goal_profile.get("key")
    )
    trace.record(
        "tool_completed",
        step=0,
        agent="PM Agent",
        tool_call={"name": "analyze_goal", "label": goal_tool},
        output=goal_profile,
        duration_ms=(time.perf_counter() - goal_started_at) * 1000,
    )
    steps = [
        {
            "agent": "PM Agent",
            "tool": goal_tool,
            "output": (
                f"识别到分析重点：{goal_profile['label']}。"
                f"优先关注：{format_categories(goal_profile['priority_categories'])}。"
                + (
                    f"本轮目标已从“{previous_goal_profile['label']}”切换到"
                    f"“{goal_profile['label']}”。"
                    if goal_changed
                    else (
                        "本轮沿用上一轮分析重点。"
                        if previous_goal_profile
                        else ""
                    )
                )
                + (
                    f"原因：{goal_profile.get('llm_reason', '')}"
                    if goal_profile.get("llm_reason")
                    else ""
                )
            ),
            "action": "analyze_goal",
            "raw_output": goal_profile,
        }
    ]
    initial_llm_disabled_reason = goal_profile.get("llm_disabled_reason")
    state = {
        "ticker": ticker,
        "mission": mission,
        "max_headlines": max_headlines,
        "allow_fallback_data": allow_fallback_data,
        "goal_profile": goal_profile,
        "previous_goal_profile": previous_goal_profile,
        "goal_changed": goal_changed,
        "source_note": (prior_memory or {}).get("source_note", "Evidence has not been collected yet."),
        "snapshot": (prior_memory or {}).get("snapshot", {}),
        "news_df": (prior_memory or {}).get("news_df"),
        "scored_news": (prior_memory or {}).get("scored_news"),
        "summary": (prior_memory or {}).get("summary"),
        "risk": (prior_memory or {}).get("risk"),
        "sec_filings": (prior_memory or {}).get("sec_filings"),
        "sec_attempted": (prior_memory or {}).get("sec_attempted", False),
        "memo": (prior_memory or {}).get("memo"),
        "memo_tool": (prior_memory or {}).get("memo_tool"),
        "memo_version": (prior_memory or {}).get("memo_version", 0),
        "reviewed_memo_version": (prior_memory or {}).get("reviewed_memo_version", 0),
        "revision_requested": False,
        "critic": (prior_memory or {}).get("critic"),
        "tool_steps": 0,
        "is_follow_up": is_follow_up,
        "pending_followup_review": is_follow_up,
        "messages": messages,
        "llm_disabled_reason": initial_llm_disabled_reason,
        "warnings": [initial_llm_disabled_reason] if initial_llm_disabled_reason else [],
    }
    if initial_llm_disabled_reason:
        trace.record(
            "llm_runtime_disabled",
            step=0,
            agent="PM Agent",
            input_data={"stage": "goal_analysis", "reason": initial_llm_disabled_reason},
        )

    terminal_reason = ""
    while state["tool_steps"] < MAX_AGENT_STEPS:
        llm_active = use_llm and not state["llm_disabled_reason"]
        decision = decide_next_action(state, llm_active, llm_model, trace, api_key=api_key)
        if decision["action"] == "finish":
            terminal_reason = decision["reason"]
            steps.append(
                {
                    "agent": "Supervisor Agent",
                    "tool": "Finish",
                    "output": f"结束本次分析：{terminal_reason}",
                    "action": "finish",
                    "decision_reason": decision["reason"],
                }
            )
            break
        step_result = execute_tool(
            decision["action"],
            state,
            llm_active,
            llm_model,
            trace,
            api_key=api_key,
        )
        step_result["decision_reason"] = decision["reason"]
        step_result["decision_source"] = decision.get("decision_source")
        steps.append(step_result)
    else:
        terminal_reason = f"达到 {MAX_AGENT_STEPS} 步上限，为防止循环而停止。"
        steps.append(
            {
                "agent": "Supervisor Agent",
                "tool": "Safety stop",
                "output": terminal_reason,
                "action": "safety_stop",
                "decision_reason": terminal_reason,
            }
        )

    if state["memo"] is None:
        trace.record(
            "run_failed",
            step=state["tool_steps"],
            agent="Supervisor Agent",
            error="Agent stopped before a memo was generated.",
        )
        raise RuntimeError("Agent stopped before a memo was generated.")

    trace.record(
        "run_finished",
        step=state["tool_steps"],
        agent="Supervisor Agent",
        output={
            "terminal_reason": terminal_reason,
            "memo_version": state["memo_version"],
            "critic": state["critic"],
            "source_note": state["source_note"],
            "llm_disabled_reason": state["llm_disabled_reason"],
        },
    )
    return {
        "ticker": ticker,
        "mission": mission,
        "goal_profile": goal_profile,
        "previous_goal_profile": previous_goal_profile,
        "goal_changed": goal_changed,
        "is_follow_up": is_follow_up,
        "source_note": state["source_note"],
        "snapshot": state["snapshot"],
        "scored_news": state["scored_news"],
        "summary": state["summary"],
        "risk": state["risk"],
        "sec_filings": state["sec_filings"],
        "critic": state["critic"],
        "memo": state["memo"],
        "memo_tool": state["memo_tool"],
        "goal_tool": goal_tool,
        "steps": steps,
        "trace_path": str(trace.path),
        "trace_run_id": trace.run_id,
        "terminal_reason": terminal_reason,
        "warnings": state["warnings"],
        "memory": build_conversation_memory(state),
    }


def render_agent_steps(steps):
    st.caption("以下展示每个 Agent 的决策理由、工具标签和完整结构化输出，便于录屏与复盘。")
    for index, step in enumerate(steps, start=1):
        with st.expander(f"{index}. {step['agent']} · {step['tool']}", expanded=True):
            st.markdown(step["output"])
            if step.get("decision_reason"):
                st.markdown("**Supervisor 决策理由**")
                st.info(step["decision_reason"])
            if step.get("raw_output") is not None:
                st.markdown("**完整工具输出**")
                raw_output = step["raw_output"]
                if isinstance(raw_output, dict) and "headlines" in raw_output:
                    headlines = raw_output["headlines"]
                    if isinstance(headlines, pd.DataFrame):
                        st.dataframe(headlines, use_container_width=True, hide_index=True)
                    remainder = {key: value for key, value in raw_output.items() if key != "headlines"}
                    if remainder:
                        st.json(remainder)
                elif isinstance(raw_output, pd.DataFrame):
                    st.dataframe(raw_output, use_container_width=True, hide_index=True)
                else:
                    st.json(raw_output)


def render_metrics(result):
    summary = result["summary"]
    risk = result["risk"]
    snapshot = result["snapshot"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bullishness", f"{summary['bullishness']} / 100")
    col2.metric("Avg Sentiment", f"{summary['avg_score']:.3f}")
    col3.metric("Risk Level", risk["level"])
    col3.caption(f"Risk score: {risk['score']} / 100")
    col4.metric("Price", snapshot.get("Price", "N/A"), snapshot.get("Change", ""))


def render_dashboard(result):
    goal_profile = result["goal_profile"]
    scored_news = result["scored_news"]
    ticker = result["ticker"]

    conclusion_tab, evidence_tab, workflow_tab, quality_tab = st.tabs(
        ["研究结论", "风险与证据", "Agent 过程", "质量与追踪"]
    )

    with conclusion_tab:
        st.caption(
            f"目标：{goal_profile['label']} · 重点：{format_categories(goal_profile['priority_categories'])}"
        )
        if result.get("goal_changed"):
            previous = result.get("previous_goal_profile") or {}
            st.info(
                f"本轮追问已切换分析目标："
                f"“{previous.get('label', '上一轮目标')}” → “{goal_profile['label']}”。"
            )
        elif result.get("is_follow_up"):
            st.caption("本轮追问沿用上一轮目标，但会优先回答最新问题并复用已有证据。")
        render_metrics(result)
        st.divider()
        # Render the memo only once; historical chat messages use compact receipts.
        st.markdown(result["memo"])
        with st.expander("查看情绪图表", expanded=False):
            chart_left, chart_right = st.columns([2, 1])
            chart_left.plotly_chart(plot_sentiment_timeline(scored_news, ticker), use_container_width=True)
            chart_right.plotly_chart(plot_sentiment_mix(scored_news), use_container_width=True)

    with evidence_tab:
        risk_col, snapshot_col = st.columns([1.25, 1])
        with risk_col:
            st.markdown("### 风险信号")
            st.caption(f"展示前 {len(result['risk']['findings'])} 条与目标最相关的信号。")
            if result["risk"]["findings"]:
                st.dataframe(
                    pd.DataFrame(result["risk"]["findings"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("当前新闻标题中没有识别到高频风险关键词。")

        with snapshot_col:
            st.markdown("### 市场快照")
            snapshot_df = pd.DataFrame(
                [{"metric": key, "value": value} for key, value in result["snapshot"].items()]
            )
            if len(snapshot_df):
                st.dataframe(snapshot_df, use_container_width=True, hide_index=True)
            else:
                st.info("市场快照暂不可用；当前结果主要基于新闻标题。")

        sec_filings = (result.get("sec_filings") or {}).get("filings", [])
        if sec_filings:
            st.markdown("### 一手来源（SEC EDGAR）")
            st.dataframe(pd.DataFrame(sec_filings), use_container_width=True, hide_index=True)

        st.markdown("### 新闻证据")
        table = scored_news.reset_index()
        table["datetime"] = table["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            table[
                [
                    "datetime",
                    "headline",
                    "sentiment_label",
                    "sentiment_score",
                    "neg",
                    "neu",
                    "pos",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with workflow_tab:
        st.subheader("Agent 执行路径")
        render_agent_steps(result["steps"])

    with quality_tab:
        st.subheader("Critic 检查")
        critic = result.get("critic") or {}
        if critic:
            st.caption(
                f"{critic.get('status', 'unknown')} · 答案分 {critic.get('answer_score', 'N/A')} · "
                f"证据分 {critic.get('evidence_score', 'N/A')} · "
                f"预期追问：{critic.get('predicted_user_followup', 'N/A')}"
            )
            st.write(critic.get("reason", ""))
            if critic.get("missing_evidence"):
                st.warning("待补证据：" + "；".join(critic["missing_evidence"]))

        trace_path = Path(result["trace_path"])
        if trace_path.exists():
            st.divider()
            st.markdown("### 可审计 Trace")
            st.caption(
                f"Run: {result['trace_run_id']} · JSONL 包含工具调用、决策、耗时和 token 用量。"
            )
            st.download_button(
                "下载 JSONL trace",
                data=trace_path.read_bytes(),
                file_name=trace_path.name,
                mime="application/x-ndjson",
            )


def main():
    st.set_page_config(page_title="StockPilot Agent", layout="wide")

    st.markdown(
        """
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 1.5rem;}
  .agent-step {
    border: 1px solid #EAECF0;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    background: #FFFFFF;
  }
  .agent-title {font-weight: 700; color: #101828;}
  .agent-tool {font-size: 0.85rem; color: #667085; margin: 3px 0 8px 0;}
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("StockPilot Agent")
    st.markdown("面向不能时刻盯盘的长线投资者：带上下文记忆地检索、分析、复盘和追问。")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = DEFAULT_LLM_MODEL

    with st.sidebar:
        st.header("Analysis Settings")
        ticker = st.text_input("Stock ticker", "").upper().strip()
        st.caption("在下方对话框输入首次研究目标；后续可继续追问，例如“详细说说那个监管风险”。")
        max_headlines = st.slider("Headlines to analyze", 6, 30, 16)
        use_llm = st.checkbox("Use LLM for goal analysis and memo", value=True)
        api_key = st.text_input(
            "DeepSeek API key",
            type="password",
            disabled=not use_llm,
            help="仅用于当前浏览器会话，不会写入 trace、文件或环境变量。",
        )
        llm_model = st.text_input(
            "DeepSeek model",
            disabled=not use_llm,
            key="llm_model",
            help="默认 deepseek-v4-flash；也可填写 deepseek-v4-pro。",
        )
        st.caption(
            "DeepSeek 用于研究推理；免费本地 multilingual-e5-small 用于语义 embedding，不需要第二个 API Key。"
        )
        allow_fallback_data = st.checkbox("Use offline sample data if live fetch fails", value=False)
        if st.button("New research conversation"):
            st.session_state.messages = []
            st.session_state.agent_memory = None
            st.session_state.last_result = None
            st.rerun()

    memory_ticker = (st.session_state.agent_memory or {}).get("ticker")
    if memory_ticker and ticker and memory_ticker != ticker:
        st.session_state.messages = []
        st.session_state.agent_memory = None
        st.session_state.last_result = None
        st.info("已因 ticker 变更创建新的研究会话。")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.caption(message.get("display", "上一轮研究已完成，可继续追问。"))
                with st.expander("查看上轮研究 memo", expanded=False):
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    mission = st.chat_input("输入研究目标或继续追问…")
    if mission:
        if not ticker:
            st.warning("Please enter a stock ticker.")
            return
        if use_llm and not api_key.strip():
            st.warning("请输入 DeepSeek API key，或关闭 LLM 模式后使用规则基线运行。")
            return

        st.session_state.messages.append({"role": "user", "content": mission})
        with st.chat_message("user"):
            st.markdown(mission)
        with st.chat_message("assistant"):
            with st.spinner("StockPilot Agent 正在分析上下文与证据..."):
                result = run_workflow(
                    ticker,
                    mission,
                    max_headlines,
                    allow_fallback_data,
                    use_llm,
                    llm_model,
                    api_key=api_key.strip(),
                    prior_memory=st.session_state.agent_memory,
                    messages=st.session_state.messages,
                )
            for warning in result.get("warnings", []):
                st.warning(warning)
            render_dashboard(result)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["memo"],
                "display": (
                    f"已完成 {ticker} 研究 · 数据来源：{result['source_note']} · "
                    "可继续追问以复用本轮证据。"
                ),
            }
        )
        st.session_state.agent_memory = result["memory"]
        st.session_state.last_result = result


if __name__ == "__main__":
    main()
