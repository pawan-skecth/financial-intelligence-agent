"""
Evaluation harness for the multi-agent stock analysis system.

Runs the pipeline on a list of stocks, records metrics per stock,
and exports results to CSV + prints a summary report.
"""

import csv
import re
import time
from datetime import datetime

from workflow.graph import app


# ============================================================
# STOCKS TO EVALUATE (10 stocks — mix of sectors)
# ============================================================
STOCKS = [
    "RELIANCE.NS",      # Conglomerate
    "TCS.NS",           # IT
    "HDFCBANK.NS",      # Banking
    "INFY.NS",          # IT
    "ICICIBANK.NS",     # Banking
    "HINDUNILVR.NS",    # FMCG
    "BHARTIARTL.NS",    # Telecom
    "ITC.NS",           # FMCG
    "LT.NS",            # Infrastructure
    "SBIN.NS",          # PSU Bank
]

# Required output sections for format compliance check
REQUIRED_SECTIONS = [
    "DECISION:",
    "Current Stock Price:",
    "Fundamental Analysis:",
    "Technical Analysis:",
    "Risk Analysis:",
    "News & Sentiment Analysis:",
    "Reason:",
    "Confidence Score:",
    "SENTIMENT_BADGE:",
]


# ============================================================
# HELPERS
# ============================================================
def extract_decision(text):
    m = re.search(r"DECISION\s*:?\s*\**\s*(BUY|HOLD|SELL)\b", text.upper())
    return m.group(1) if m else "UNKNOWN"


def extract_confidence(text):
    m = re.search(r"CONFIDENCE\s*SCORE\s*:?\s*\**\s*(\d{1,3})", text.upper())
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def extract_sentiment(text):
    m = re.search(
        r"SENTIMENT_BADGE\s*:?\s*\**\s*(POSITIVE|NEGATIVE|NEUTRAL)",
        text.upper(),
    )
    return m.group(1).capitalize() if m else "UNKNOWN"


def check_format_compliance(text):
    """Return (passed_count, total, missing_list)."""
    missing = [s for s in REQUIRED_SECTIONS if s.upper() not in text.upper()]
    passed = len(REQUIRED_SECTIONS) - len(missing)
    return passed, len(REQUIRED_SECTIONS), missing


# ============================================================
# RUN EVALUATION
# ============================================================
def run_evaluation():
    print("=" * 70)
    print(f"EVALUATION STARTED — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stocks: {len(STOCKS)}")
    print("=" * 70)

    results = []
    decisions_count = {"BUY": 0, "HOLD": 0, "SELL": 0, "UNKNOWN": 0}
    sentiments_count = {"Positive": 0, "Negative": 0, "Neutral": 0, "Unknown": 0}

    for i, symbol in enumerate(STOCKS, start=1):
        print(f"\n[{i}/{len(STOCKS)}] Analyzing {symbol} ...")
        start = time.time()

        try:
            state = app.invoke({"stock_symbol": symbol})
            elapsed = round(time.time() - start, 2)

            strategy = state.get("strategy", "") or ""
            decision = extract_decision(strategy)
            confidence = extract_confidence(strategy)
            sentiment = extract_sentiment(strategy)
            passed, total, missing = check_format_compliance(strategy)

            # Check for any agent-level errors
            errors = []
            for key in ["stock_data", "fundamental_data", "technical_data",
                        "risk_data", "news_data"]:
                agent_data = state.get(key)
                if isinstance(agent_data, dict) and agent_data.get("error"):
                    errors.append(f"{key}: {agent_data['error'][:60]}")

            status = "OK" if not errors else f"PARTIAL ({len(errors)} errors)"

            results.append({
                "symbol": symbol,
                "decision": decision,
                "confidence": confidence if confidence is not None else "",
                "sentiment": sentiment,
                "format_passed": passed,
                "format_total": total,
                "format_missing": ";".join(missing),
                "latency_sec": elapsed,
                "status": status,
                "error_summary": " | ".join(errors),
            })

            decisions_count[decision] = decisions_count.get(decision, 0) + 1
            sentiments_count[sentiment] = sentiments_count.get(sentiment, 0) + 1

            print(f"   → {decision} ({confidence}%) — "
                  f"Sentiment: {sentiment} — {elapsed}s — Format: {passed}/{total}")

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            print(f"   → ERROR: {e}")
            results.append({
                "symbol": symbol,
                "decision": "ERROR",
                "confidence": "",
                "sentiment": "ERROR",
                "format_passed": 0,
                "format_total": len(REQUIRED_SECTIONS),
                "format_missing": "ALL",
                "latency_sec": elapsed,
                "status": "FAILED",
                "error_summary": str(e)[:120],
            })

    # ============================================================
    # EXPORT CSV
    # ============================================================
    csv_file = "evaluation_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ============================================================
    # SUMMARY REPORT
    # ============================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total = len(results)
    ok_count = sum(1 for r in results if r["status"] == "OK")
    partial_count = sum(1 for r in results if r["status"].startswith("PARTIAL"))
    failed_count = sum(1 for r in results if r["status"] == "FAILED")

    # Format compliance
    total_sections = sum(r["format_total"] for r in results)
    passed_sections = sum(r["format_passed"] for r in results)
    format_pct = round(passed_sections / total_sections * 100, 1) if total_sections else 0

    # Latency
    latencies = [r["latency_sec"] for r in results if r["latency_sec"]]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0
    median_latency = round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0

    # Confidence stats (only numeric)
    confs = [r["confidence"] for r in results if isinstance(r["confidence"], int)]
    avg_conf = round(sum(confs) / len(confs), 1) if confs else 0
    min_conf = min(confs) if confs else 0
    max_conf = max(confs) if confs else 0

    print(f"\n📊 RUN STATS")
    print(f"   Total stocks         : {total}")
    print(f"   Completed OK         : {ok_count}")
    print(f"   Partial (some error) : {partial_count}")
    print(f"   Failed               : {failed_count}")

    print(f"\n📄 FORMAT COMPLIANCE")
    print(f"   Sections passed      : {passed_sections}/{total_sections} ({format_pct}%)")

    print(f"\n🎯 DECISION DISTRIBUTION")
    for dec in ["BUY", "HOLD", "SELL", "UNKNOWN"]:
        cnt = decisions_count.get(dec, 0)
        pct = round(cnt / total * 100, 1) if total else 0
        print(f"   {dec:8s} : {cnt:2d} ({pct:5.1f}%)")

    print(f"\n😊 SENTIMENT DISTRIBUTION")
    for sent in ["Positive", "Neutral", "Negative", "Unknown"]:
        cnt = sentiments_count.get(sent, 0)
        pct = round(cnt / total * 100, 1) if total else 0
        print(f"   {sent:8s} : {cnt:2d} ({pct:5.1f}%)")

    print(f"\n📈 CONFIDENCE SCORES")
    print(f"   Average              : {avg_conf}")
    print(f"   Min / Max            : {min_conf} / {max_conf}")
    print(f"   Range (max - min)    : {max_conf - min_conf}")

    print(f"\n⏱️  LATENCY")
    print(f"   Average              : {avg_latency}s")
    print(f"   Median               : {median_latency}s")

    print(f"\n💾 Detailed results saved to: {csv_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()