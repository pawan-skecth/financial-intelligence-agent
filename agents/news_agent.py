import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=30,
)


# ============================================================
# STEP 1: RETRIEVE RECENT STOCK NEWS
# ============================================================

def get_news(stock_symbol, max_items=5):
    """Fetch recent news headlines from Google News RSS."""

    search_term = (
        stock_symbol.strip()
        .upper()
        .replace(".NS", "")
        .replace(".BO", "")
    )

    if not search_term:
        return []

    query = urllib.parse.urlencode({
        "q": f"{search_term} when:30d",
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en",
    })

    url = "https://news.google.com/rss/search?" + query

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        response = urllib.request.urlopen(request, timeout=20)
        root = ET.fromstring(response.read())

        news_items = []
        seen_titles = set()

        for item in root.findall(".//item"):
            title = item.findtext("title")

            if not title or title in seen_titles:
                continue

            seen_titles.add(title)

            publisher_element = item.find("source")
            publisher = (
                publisher_element.text
                if publisher_element is not None
                else "Unknown"
            )

            news_items.append({
                "title": title,
                "publisher": publisher,
                "published_date": item.findtext("pubDate"),
            })

            if len(news_items) >= max_items:
                break

        return news_items

    except Exception as e:
        print("NEWS RETRIEVAL ERROR:", e)
        return []


# ============================================================
# STEP 2: PARSE STRUCTURED BLOCK FROM LLM OUTPUT
# ============================================================

def _extract_structured(text: str, headlines_count: int) -> dict:
    """Extract ===STRUCTURED=== ... ===END=== block from LLM output."""

    default = {
        "sentiment_label": "Neutral",
        "sentiment_confidence": "Low",
        "headlines_analyzed": headlines_count,
        "market_impact_direction": "Unclear",
        "key_catalyst": "",
    }

    if not text:
        return default

    match = re.search(
        r"===STRUCTURED===(.*?)===END===",
        text,
        re.DOTALL,
    )

    if not match:
        return default

    block = match.group(1).strip()
    result = dict(default)

    for line in block.splitlines():
        if ":" not in line:
            continue

        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if key in result:
            result[key] = value

    # Normalize + validate
    if result["sentiment_label"] not in ("Positive", "Negative", "Neutral"):
        result["sentiment_label"] = "Neutral"

    if result["sentiment_confidence"] not in ("High", "Medium", "Low"):
        result["sentiment_confidence"] = "Low"

    if result["market_impact_direction"] not in (
        "Positive", "Negative", "Mixed", "Unclear"
    ):
        result["market_impact_direction"] = "Unclear"

    try:
        result["headlines_analyzed"] = int(result["headlines_analyzed"])
    except (ValueError, TypeError):
        result["headlines_analyzed"] = headlines_count

    return result


# ============================================================
# STEP 3: NEWS + SENTIMENT ANALYSIS (MERGED AGENT)
# ============================================================

def analyze_news(stock_symbol):
    """
    Combined news + sentiment agent.

    Returns a structured dict:
    {
        "news_items":    list of headlines,
        "analysis_text": full markdown analysis (display / strategy agent),
        "sentiment":     structured dict (UI badge / evaluation),
        "error":         None or error string,
    }
    """

    headlines = get_news(stock_symbol)

    # ---- Handle empty news ----
    if not headlines:
        return {
            "news_items": [],
            "analysis_text": (
                "News Analysis: Unavailable\n"
                "Sentiment Analysis: Neutral / Insufficient Information\n"
                "No recent news available or news retrieval failed. "
                "No sentiment conclusion can be reliably established."
            ),
            "sentiment": {
                "sentiment_label": "Neutral",
                "sentiment_confidence": "Low",
                "headlines_analyzed": 0,
                "market_impact_direction": "Unclear",
                "key_catalyst": "",
            },
            "error": None,
        }

    prompt = f"""
You are a professional financial news and sentiment analyst.

Analyze the following retrieved news headlines for {stock_symbol}.

ACTUAL NEWS HEADLINES:

{headlines}


YOUR TASK:

Perform both news analysis and sentiment analysis.

1. NEWS SUMMARY

- Summarize the important developments from the headlines.
- Identify the major news events related to the company.

2. INDIVIDUAL NEWS SENTIMENT

- Analyse the sentiment of each headline.
- Classify each headline as Positive, Negative, or Neutral.
- Briefly explain the reason for each classification.

3. OVERALL NEWS SENTIMENT

- Determine the overall sentiment across all retrieved headlines.
- Classify it as Positive, Negative, or Neutral.
- Consider the importance and relevance of the news.
- Do not classify sentiment based only on the number of positive or negative headlines.
- Explain the main factors influencing the overall sentiment.

4. POTENTIAL MARKET IMPACT

- Explain how the reported developments could affect the company.
- Identify possible positive and negative implications.
- Distinguish reported facts from possible market implications.

5. INVESTOR INSIGHT

- Explain what investors should pay attention to.
- Identify important developments that may require further investigation.


IMPORTANT RULES:

- Analyse only the information provided in the headlines.
- Never invent financial figures, facts, or company developments.
- Do not assume missing financial data or cash-flow trends.
- Never predict a specific stock price movement or percentage.
- Do not call revenue or profit growth an earnings beat unless analyst expectations are provided.
- Do not treat an article title as proof of information not included in the retrieved data.
- Clearly distinguish reported facts from possible implications.
- If information is insufficient, clearly mention it.
- Do not assume every positive headline has a positive financial impact.
- Do not assume every negative headline has a negative long-term impact.
- Do not provide BUY, HOLD, or SELL recommendations.
- If the available headlines are insufficient to establish sentiment, clearly state that limitation.


RESPONSE FORMAT:

Write your full analysis in sections 1-5 above.

Then, at the VERY END of your response, add a machine-readable block
in EXACTLY this format. Nothing before "===STRUCTURED===" on its line,
nothing after "===END===" on its line:

===STRUCTURED===
sentiment_label: Positive | Negative | Neutral
sentiment_confidence: High | Medium | Low
headlines_analyzed: <number of headlines you actually analyzed>
market_impact_direction: Positive | Negative | Mixed | Unclear
key_catalyst: <one short phrase, max 15 words>
===END===

Keep the analysis concise, structured, and useful for the final stock strategy agent.
"""

    try:
        response = llm.invoke(prompt)
        raw_text = response.content or ""

        # Extract structured block
        sentiment = _extract_structured(raw_text, len(headlines))

        # Clean text for display (remove the structured tail)
        clean_text = re.split(
            r"===STRUCTURED===",
            raw_text,
            maxsplit=1,
        )[0].strip()

        return {
            "news_items": headlines,
            "analysis_text": clean_text,
            "sentiment": sentiment,
            "error": None,
        }

    except Exception as e:
        return {
            "news_items": headlines,
            "analysis_text": None,
            "sentiment": {
                "sentiment_label": "Neutral",
                "sentiment_confidence": "Low",
                "headlines_analyzed": len(headlines),
                "market_impact_direction": "Unclear",
                "key_catalyst": "",
            },
            "error": f"News and Sentiment Analysis Failed: {str(e)}",
        }