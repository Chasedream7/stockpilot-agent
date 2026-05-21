import datetime as dt
import json
import os
import re
import tempfile
import time
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


FINVIZ_URL = "https://finviz.com/quote.ashx?t="
DEFAULT_LLM_MODEL = "gpt-4o-mini"

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


def get_config_value(name, default=None):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_openai_client():
    api_key = get_config_value("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return None
    return OpenAI(api_key=api_key)


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


def parse_snapshot(soup):
    table = soup.find("table", class_="snapshot-table2")
    if not table:
        return {}

    cells = [cell.get_text(" ", strip=True) for cell in table.find_all("td")]
    return {cells[i]: cells[i + 1] for i in range(0, len(cells) - 1, 2)}


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


def call_llm_text(system_prompt, user_prompt, model):
    client = get_openai_client()
    if client is None:
        return None

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text


def infer_goal_profile_with_llm(mission, model):
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

    output = call_llm_text(system_prompt, user_prompt, model)
    parsed = extract_json_object(output)
    if not parsed:
        return None

    profile = profile_by_key(parsed.get("profile_key"))
    if not profile:
        return None

    enriched = dict(profile)
    enriched["analysis_source"] = "LLM"
    enriched["llm_reason"] = parsed.get("reason", "")
    return enriched


def analyze_goal(mission, use_llm, model):
    if use_llm:
        try:
            profile = infer_goal_profile_with_llm(mission, model)
            if profile:
                return profile, "LLM goal analysis"
        except Exception as error:
            fallback = dict(infer_goal_profile(mission))
            fallback["analysis_source"] = "Rule fallback"
            fallback["llm_reason"] = f"LLM goal analysis failed: {error}"
            return fallback, "Rule fallback after LLM error"

    profile = dict(infer_goal_profile(mission))
    profile["analysis_source"] = "Rule fallback"
    if use_llm:
        profile["llm_reason"] = "OPENAI_API_KEY is not configured or openai package is unavailable."
        return profile, "Rule fallback because LLM is unavailable"
    profile["llm_reason"] = "LLM mode is off."
    return profile, "Rule-based goal parser"


def format_categories(categories):
    return "、".join(categories)


def build_next_steps(goal_profile):
    return "\n".join([f"- {step}" for step in goal_profile["next_steps"]])


def detect_risks(scored_news, snapshot, goal_profile):
    priority_categories = set(goal_profile["priority_categories"])
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
                        "sentiment_score": round(row["sentiment_score"], 3),
                        "focus_match": "Yes" if category in priority_categories else "No",
                    }
                )

    change = parse_percent(snapshot.get("Change", ""))
    volatility_points = 0
    if change is not None and abs(change) >= 3:
        volatility_points = 18
        findings.append(
            {
                "category": "价格波动",
                "headline": f"FinViz shows intraday change of {change:.2f}%.",
                "matched_signal": "large price move",
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

    findings = sorted(
        findings,
        key=lambda item: (item["focus_match"] != "Yes", item["sentiment_score"]),
    )

    return {
        "level": level,
        "score": round(risk_score),
        "findings": findings[:8],
        "focus_matches": sum(1 for item in findings if item["focus_match"] == "Yes"),
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


def build_rule_memo(ticker, mission, goal_profile, summary, risk, snapshot, scored_news, source_note):
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

    return f"""
### StockPilot Agent Memo: {ticker}

**用户目标**：{mission}

**识别到的分析重点**：{goal_profile['label']}  
{goal_profile['description']}

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


def build_llm_memo(ticker, mission, goal_profile, summary, risk, snapshot, scored_news, source_note, model):
    system_prompt = """
You are Portfolio Copilot inside StockPilot Agent.
Write a concise Chinese holding-observation memo for a long-term individual investor.
Use only the supplied metrics, headlines, and risk findings.
Do not invent news, prices, financial facts, or future predictions.
Do not provide direct buy/sell/hold instructions.
Frame conclusions as observation and review guidance, not investment advice.
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
    }
    user_prompt = f"""
Create the memo in Markdown with these sections:
1. StockPilot Agent Memo: {ticker}
2. User Goal
3. Goal Focus
4. Observation
5. Key Readings
6. Evidence
7. Risks To Review
8. Next Steps
9. Disclaimer

Data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    return call_llm_text(system_prompt, user_prompt, model)


def build_memo(ticker, mission, goal_profile, summary, risk, snapshot, scored_news, source_note, use_llm, model):
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
            )
            if memo:
                return memo.strip(), "LLM memo generator"
        except Exception:
            pass

    memo = build_rule_memo(ticker, mission, goal_profile, summary, risk, snapshot, scored_news, source_note)
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


def run_workflow(ticker, mission, max_headlines, allow_fallback_data, use_llm=False, llm_model=DEFAULT_LLM_MODEL):
    goal_profile, goal_tool = analyze_goal(mission, use_llm, llm_model)
    steps = [
        {
            "agent": "Briefing Agent",
            "tool": goal_tool,
            "output": (
                f"识别到分析重点：{goal_profile['label']}。"
                f"优先关注：{format_categories(goal_profile['priority_categories'])}。"
                f"{'原因：' + goal_profile.get('llm_reason', '') if goal_profile.get('llm_reason') else ''}"
            ),
        }
    ]

    source_note = "Live FinViz data"
    snapshot = {}
    try:
        soup = fetch_finviz_soup(ticker)
        snapshot = parse_snapshot(soup)
        news_df = parse_news(soup.find(id="news-table"), max_headlines)
        steps.append(
            {
                "agent": "News Scout Agent",
                "tool": "FinViz scraper",
                "output": f"抓取到 {len(news_df)} 条最近新闻，并提取页面市场快照。",
            }
        )
    except Exception as error:
        if not allow_fallback_data:
            raise
        source_note = f"Offline sample data; live fetch failed: {error}"
        news_df = create_fallback_news(ticker, max_headlines)
        steps.append(
            {
                "agent": "News Scout Agent",
                "tool": "Offline sample dataset",
                "output": "实时网页抓取失败，已切换到离线样例数据；请不要把该结果作为当天行情判断。",
            }
        )

    scored_news = score_news(news_df)
    summary = summarize_sentiment(scored_news)
    steps.append(
        {
            "agent": "Sentiment Agent",
            "tool": scored_news.attrs.get("sentiment_tool", "Sentiment analyzer"),
            "output": (
                f"完成 {summary['total']} 条标题打分：Positive {summary['positive']}，"
                f"Neutral {summary['neutral']}，Negative {summary['negative']}。"
            ),
        }
    )

    risk = detect_risks(scored_news, snapshot, goal_profile)
    steps.append(
        {
            "agent": "Risk Agent",
            "tool": "Keyword risk scanner",
            "output": (
                f"识别到 {len(risk['findings'])} 个风险信号，其中 "
                f"{risk['focus_matches']} 个与当前分析重点直接相关；综合风险等级为 {risk['level']}。"
            ),
        }
    )

    memo, memo_tool = build_memo(
        ticker,
        mission,
        goal_profile,
        summary,
        risk,
        snapshot,
        scored_news,
        source_note,
        use_llm,
        llm_model,
    )
    steps.append(
        {
            "agent": "Portfolio Copilot",
            "tool": memo_tool,
            "output": "已生成可用于投资复盘、关注列表更新或个人记录的结构化 memo。",
        }
    )

    return {
        "ticker": ticker,
        "mission": mission,
        "goal_profile": goal_profile,
        "source_note": source_note,
        "snapshot": snapshot,
        "scored_news": scored_news,
        "summary": summary,
        "risk": risk,
        "memo": memo,
        "memo_tool": memo_tool,
        "goal_tool": goal_tool,
        "steps": steps,
    }


def render_agent_steps(steps):
    st.subheader("Agent Workflow")
    progress = st.progress(0)
    log_area = st.empty()
    rendered = []

    for index, step in enumerate(steps, start=1):
        rendered.append(
            f"""
<div class="agent-step">
  <div class="agent-title">{index}. {step['agent']}</div>
  <div class="agent-tool">Tool: {step['tool']}</div>
  <div>{step['output']}</div>
</div>
"""
        )
        log_area.markdown("\n".join(rendered), unsafe_allow_html=True)
        progress.progress(index / len(steps))
        time.sleep(0.2)


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
    st.subheader("Decision Dashboard")
    goal_profile = result["goal_profile"]
    st.caption(
        f"Goal focus: {goal_profile['label']} | "
        f"Priority areas: {format_categories(goal_profile['priority_categories'])}"
    )
    render_metrics(result)

    scored_news = result["scored_news"]
    ticker = result["ticker"]

    chart_left, chart_right = st.columns([2, 1])
    chart_left.plotly_chart(plot_sentiment_timeline(scored_news, ticker), use_container_width=True)
    chart_right.plotly_chart(plot_sentiment_mix(scored_news), use_container_width=True)

    memo_col, risk_col = st.columns([1.35, 1])
    with memo_col:
        st.markdown(result["memo"])

    with risk_col:
        st.markdown("### Risk Signals")
        if result["risk"]["findings"]:
            st.dataframe(pd.DataFrame(result["risk"]["findings"]))
        else:
            st.success("当前新闻标题中没有识别到高频风险关键词。")

        st.markdown("### Market Snapshot")
        snapshot_df = pd.DataFrame(
            [{"metric": key, "value": value} for key, value in result["snapshot"].items()]
        )
        if len(snapshot_df):
            st.dataframe(snapshot_df)
        else:
            st.info("市场快照暂不可用；当前结果主要基于新闻标题。")

    st.markdown("### Evidence Table")
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
        ]
    )


def render_empty_state():
    st.info("Enter a stock ticker and analysis goal, then click Run Agent Workflow.")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**1. 自动拆任务**\n\n从用户目标拆成新闻、情绪、风险、报告四步。")
    col2.markdown("**2. 带证据输出**\n\n每个判断都能回到新闻标题和风险信号。")
    col3.markdown("**3. 可交付 memo**\n\n最终结果不是分数，而是可用于汇报的决策摘要。")


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
    st.markdown("面向不能时刻盯盘的长线投资者：自动抓取新闻、分析情绪、识别风险、生成持仓观察 memo。")

    with st.sidebar:
        st.header("Analysis Settings")
        ticker = st.text_input("Stock ticker", "").upper().strip()
        mission = st.text_area(
            "Analysis goal",
            "判断这只股票今天是否出现需要复盘的风险信号，并输出适合投资日志记录的观察摘要。",
            height=110,
        )
        st.caption("Examples: holding risk / positive catalysts / earnings signals / regulation and lawsuits")
        max_headlines = st.slider("Headlines to analyze", 6, 30, 16)
        use_llm = st.checkbox("Use LLM for goal analysis and memo", value=True)
        llm_model = st.text_input("LLM model", get_config_value("OPENAI_MODEL", DEFAULT_LLM_MODEL))
        st.caption("Set OPENAI_API_KEY in your environment or Streamlit secrets. Without it, the app uses rule-based fallback.")
        allow_fallback_data = st.checkbox("Use offline sample data if live fetch fails", value=False)
        run_button = st.button("Run Agent Workflow")

    if run_button:
        if not ticker:
            st.warning("Please enter a stock ticker.")
            return

        with st.spinner("StockPilot Agent 正在运行..."):
            result = run_workflow(
                ticker,
                mission,
                max_headlines,
                allow_fallback_data,
                use_llm,
                llm_model,
            )

        st.caption(result["source_note"])
        render_agent_steps(result["steps"])
        render_dashboard(result)
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
