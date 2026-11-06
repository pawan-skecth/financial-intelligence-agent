# 🧠 Multi-Agent Stock Analysis System | End-to-End AI Deployment using LangGraph, Streamlit, Groq & Supabase

An end-to-end AI application that analyzes a stock symbol using **five specialized agents** and produces a structured financial analysis with a **BUY / HOLD / SELL** decision, a confidence score, and a sentiment verdict. The project demonstrates a production-style AI deployment lifecycle using LangGraph for orchestration, Streamlit for the UI, Groq for LLM inference, Supabase for persistent memory, and a custom evaluation harness for system health validation.

---

## 🚀 Project Overview

This project was developed to demonstrate how a multi-agent LLM system can be deployed as a real-world financial analysis application using modern AI engineering practices.

The application consists of:

- **5 specialized agents** for fundamental, technical, risk, news+sentiment, and strategy analysis
- **LangGraph** for deterministic multi-agent orchestration
- **Streamlit frontend** for an interactive user interface
- **Groq API** for fast LLM inference (`openai/gpt-oss-20b`)
- **Supabase** for persistent decision memory (with Row Level Security)
- **Evaluation harness** for system health validation
- A **strict 4-condition rubric** governing the final BUY/HOLD/SELL decision

The project follows a production-style AI engineering workflow with separation of computation and interpretation, structured outputs, graceful degradation, and evaluation.

---

## ✨ Features

- 🧠 **5 specialized agents** — Fundamental, Technical, Risk, News+Sentiment, Strategy
- 📊 **Deterministic metrics** — P/E, EPS, RSI, MACD, SMA, volatility, beta, drawdown, P/B computed in Python (LLM only interprets, never computes)
- 🎯 **Strict decision rubric** — 4 conditions (C1–C4) with a 3-of-4 rule for BUY and a risk gatekeeper
- 😊 **Structured sentiment** — News agent returns JSON with `sentiment_label`, `confidence`, `key_catalyst`
- 🔒 **Fixed P/B inversion bug** — Price-to-book pre-computed in Python and passed explicitly to the LLM
- 🧠 **Persistent memory** — Supabase stores every decision (symbol, decision, confidence, timestamp)
- 🛡️ **Security hardened** — Row Level Security enabled with restricted anon-role policies
- 🎨 **Interactive Streamlit UI** — Colored badges for BUY/HOLD/SELL, sentiment, and confidence
- 📈 **Candlestick charts** — Live price chart with SMA-20 overlay via Plotly
- 📊 **Evaluation harness** — Measures format compliance, decision distribution, confidence range, and latency
- ⚡ **Rate-limit aware** — Throttled LLM calls to stay within Groq free-tier limits
- 🔄 **Graceful degradation** — Empty symbol, missing data, and API errors return structured errors, not crashes

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Agent Orchestration | LangGraph |
| LLM Provider | Groq (`openai/gpt-oss-20b`) |
| LLM Framework | LangChain |
| Financial Data | yfinance |
| Frontend | Streamlit, Plotly |
| Database | Supabase (PostgreSQL) |
| Secret Management | python-dotenv |
| Version Control | Git & GitHub |
| Charting | Plotly, mplfinance |
| Data Processing | Pandas, NumPy |

---

## 🏗️ Project Architecture

```
                    User
                      │
                      ▼
            Streamlit Frontend
                      │
                      ▼
              LangGraph StateGraph
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   Stock Data    Fundamental    Technical
     Node           Agent         Agent
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                 Risk Agent
                      │
                      ▼
              News + Sentiment Agent
                      │
                      ▼
                Strategy Agent
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    Final         Supabase        Streamlit
   Decision        Memory           UI
```

The application follows a **state-machine architecture** where:

- **LangGraph** orchestrates a 6-node pipeline: `stock_data → fundamental → technical → risk → news → strategy`
- **Each agent** independently fetches data, computes deterministic metrics, and produces structured output
- **The Strategy agent** receives all upstream outputs, applies a strict C1–C4 rubric, and emits a final decision
- **Supabase** persists every decision for future context
- **Streamlit** renders the analysis with visual badges

---

## 📂 Project Structure

```
stock-ai-agent/
│
├── app.py                        # Streamlit Frontend
├── main.py                       # CLI Entry Point
├── evaluation.py                 # Evaluation Harness
├── requirements.txt              # Python Dependencies
├── README.md                     # Project Documentation
├── .gitignore                    # Ignored Files
├── .env                          # Environment Variables (NOT committed)
│
├── agents/
│   ├── fundamental_agent.py      # P/E, EPS, P/B, valuation
│   ├── technical_agent.py        # SMA, RSI, MACD, S/R
│   ├── risk_agent.py             # Volatility, drawdown, beta
│   ├── news_agent.py             # News + sentiment (merged)
│   └── strategy_agent.py         # Final decision + rubric
│
├── workflow/
│   ├── graph.py                  # LangGraph StateGraph
│   ├── nodes.py                  # 6 node wrappers
│   └── state.py                  # StockState TypedDict
│
├── tools/
│   └── stock_data.py             # yfinance data fetcher
│
└── utils/
    └── supabase_memory.py        # Supabase client + RLS
```

---

## 🔄 The Decision Rubric

The Strategy agent applies four conditions. A **BUY** requires **3 of 4 YES** with a risk gatekeeper. A **SELL** requires **2 of 3 NO**. Everything else is **HOLD**.

| Condition | YES if… | NO if… |
|---|---|---|
| **C1 — Fundamental** | EPS positive AND valuation is "Undervalued" or "Fairly Valued" | Valuation is "Overvalued" |
| **C2 — Technical** | Price > SMA-20 AND price > SMA-50 (or RSI > 55 with positive MACD) | Price < both SMAs (or MACD negative AND RSI < 45) |
| **C3 — Sentiment** | Sentiment = Positive | Sentiment = Negative |
| **C4 — Risk (gatekeeper)** | Volatility < 30% AND drawdown > −35% | Volatility ≥ 35% OR drawdown ≤ −40% |

**Decision logic:**
- **BUY** → 3 of 4 YES, and C4 is not NO
- **SELL** → 2 of C1, C2, C3 are NO, and C4 is not YES
- **HOLD** → everything else

**Confidence score** is computed deterministically:
```
Start at 50
+10 for each YES
+5  for each NEUTRAL
−10 for each NO
−10 if sentiment confidence is Low
−5  if any metric is unavailable
Cap at 95, floor at 30
```

---

## 🗄️ Supabase Memory with Row Level Security

The project uses Supabase to persist every decision. Security is enforced via **Row Level Security (RLS)**:

- **RLS enabled** on the `stock_memory` table
- **SELECT policy** — anon role can read (`using (true)`)
- **INSERT policy** — anon role can insert (`with check (true)`)
- **No UPDATE or DELETE policies** — data cannot be modified or deleted via the exposed key

This means even though the Supabase anon key is technically visible, the database is **safe by design** — it can only be read and appended to, never tampered with.

**Memory schema:**
```sql
CREATE TABLE stock_memory (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    stock_symbol TEXT NOT NULL,
    decision TEXT NOT NULL,          -- BUY / HOLD / SELL
    confidence INTEGER,               -- 0-100
    timestamp TEXT NOT NULL
);
```

---

## 📊 Evaluation Harness

The project includes `evaluation.py` — a system-health evaluation harness that runs the full pipeline on a configurable list of stocks and measures:

| Metric | What it validates |
|---|---|
| **Format compliance** | Did the output contain all 9 required sections? |
| **Decision distribution** | Is the rubric producing variety (BUY/HOLD/SELL), not defaulting? |
| **Confidence range** | Are confidence scores spread out, or flatlined? |
| **Sentiment distribution** | Is the news agent producing varied verdicts? |
| **Latency** | Is the pipeline usable in real time? |
| **Error rate** | How many stocks failed due to API errors? |

**Sample output:**
```
📊 RUN STATS
   Total stocks         : 10
   Completed OK         : 10
   Failed               : 0

📄 FORMAT COMPLIANCE
   Sections passed      : 90/90 (100%)

🎯 DECISION DISTRIBUTION
   BUY      :  4 (40.0%)
   HOLD     :  5 (50.0%)
   SELL     :  1 (10.0%)

📈 CONFIDENCE SCORES
   Average              : 67.5
   Min / Max            : 45 / 85

⏱️  LATENCY
   Median               : 58.0s
```

**Note on scope:** This evaluation validates *system health* (does the pipeline run reliably and differentiate stocks?). It does not validate *prediction accuracy* (do BUY decisions lead to gains?). A 30-day forward backtest would be required for that, which is on the roadmap.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/pawan-skecth/stock-ai-agent.git
cd stock-ai-agent
```

### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv stockenv
# Windows:
stockenv\Scripts\activate
# Linux/Mac:
source stockenv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here
```

- Get a **free Groq API key** at [console.groq.com](https://console.groq.com)
- Create a **free Supabase project** at [supabase.com](https://supabase.com) and create the `stock_memory` table with RLS enabled

### 5️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

**Open:** `http://localhost:8501`

### 6️⃣ (Optional) Run the Evaluation Harness
```bash
python evaluation.py
```

**Note:** The Groq free tier has a **200,000 token/day limit**. Each full analysis consumes ~35,000 tokens. Run at most 5 stocks per day, or upgrade to Groq's Dev Tier for scale testing.

---

## 🎯 Sample Usage

**Input:** `RELIANCE.NS`

**Output:**
```
DECISION: BUY

Current Stock Price: 1313.4 INR

Fundamental Analysis:
P/E 21.64, EPS ₹60.68, Market Cap ₹17.8 trn, Book Value ₹697.92,
P/B 1.88× (premium to book), Dividend Yield 0.46%. Valuation: Fairly Valued.

Technical Analysis:
Price above SMA-20 (₹1209) and SMA-50 (₹1235). RSI 67.3, MACD +18.1.
Support at ₹1306.56, resistance at ₹1319.00.

Risk Analysis:
Volatility 24.3%, max drawdown −26.9%, beta 0.99. Risk: Acceptable.

News & Sentiment Analysis:
Sentiment Positive (Medium confidence). Key catalyst: 41% profit jump
on refining and Jio strength. Counter: 2% dip after Jio Platforms listing.

Reason: All four criteria are met, confirming a strong buy case with
moderate risk and positive market sentiment.

Confidence Score: 85

SENTIMENT_BADGE: Positive / Medium
```

---

## 🔐 Security Considerations

- ✅ **Credentials in `.env`** — never committed to Git
- ✅ **`.env` in `.gitignore`** — protected from accidental push
- ✅ **Supabase RLS enabled** — anon key can only read/insert, never delete/modify
- ✅ **Structured error handling** — no stack traces exposed to users
- ✅ **Disclaimer appended** to every output — regulatory awareness (SEBI/SEC style)
- ⚠️ **Groq API key** — stored in `.env`, not hardcoded

---

## ⚠️ Important Disclaimer

This project is for **educational purposes only** and is **not financial advice**. Stock market analysis carries significant risk. Always consult a SEBI-registered (India) or SEC-registered (US) investment advisor before making investment decisions. The BUY/HOLD/SELL signals produced by this system are algorithmic interpretations of publicly available data and should not be acted upon as sole investment guidance.

---

## 🔮 Roadmap / Future Improvements

- **Backtest validation** — 30-day forward backtest to measure directional accuracy of BUY/SELL signals
- **LLM provider abstraction** — wrap the client to swap Groq → Gemini → Claude with one line
- **Structured output** — use Pydantic + `with_structured_output()` to eliminate regex parsing
- **Richer memory** — store reasoning summaries alongside decisions for trend-aware context
- **Parallel agent execution** — LangGraph `add_conditional_edges` to run independent agents in parallel
- **Docker deployment** — containerize the app for AWS EC2 / Render / Fly.io
- **Multi-market support** — extend beyond NSE to NYSE, NASDAQ, LSE

---

## 👨‍💻 Author

**Pawan** — [@pawan-skecth](https://github.com/pawan-skecth)

---

**⭐ If you found this project interesting, consider giving it a star!**
