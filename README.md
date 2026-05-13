# 📋 Procurement Intelligence Agent

> AI-powered contract analysis that reads vendor agreements, flags risks, and drafts approval memos — so your procurement team can focus on decisions, not documents.

---

## The Problem

Procurement teams manually review dozens of vendor contracts every quarter. Each review takes hours, risk clauses get missed, and approval memos are written from scratch every time. A single overlooked auto-renewal or liability gap can cost the business significantly.

---

## Key Features

- **Conversational Q&A with citations** — ask any question about a contract and get a direct answer with page-level source references
- **AI risk scanner with severity flags** — automatically identifies high/medium/low risk clauses across 10 procurement risk categories (unlimited liability, auto-renewal traps, IP ownership, data lock-in, and more)
- **Approval memo generation with PDF download** — generates structured internal memos ready for sign-off, exportable as PDF in one click
- **Multi-contract comparison** — side-by-side risk scans or Q&A across 2–4 contracts simultaneously, with a risk matrix showing coverage gaps at a glance
- **Live vendor research via web search** — searches the web for lawsuits, data breaches, financial distress, and regulatory penalties, then surfaces a structured risk card with sentiment rating

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM + structured output | Gemini 2.5 Flash (google-genai SDK) |
| Web search grounding | Gemini Google Search tool |
| Vector store | ChromaDB (local persistent) |
| PDF parsing | pypdf |
| UI | Streamlit |
| PDF export | reportlab |
| Embeddings | Gemini Embedding 001 |

---

## Run Locally

**1. Clone and set up the environment**
```bash
git clone https://github.com/PankajP0203/procurement-intelligence-agent.git
cd procurement-intelligence-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Add your API key**
```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

**3. Index the sample contracts**
```bash
python ingestion.py
```

**4. Launch the app**
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Demo Contracts Included

Five synthetic vendor contracts are bundled for testing:

| Contract | Type | Key Risks |
|---|---|---|
| CloudSync SaaS | SaaS | Auto-renewal trap, IP ownership risk |
| NexStaff IT Staffing | Staffing | Unlimited liability |
| SwiftLog Logistics | Logistics | Missing SLA, data lock-in |
| BrandCraft Marketing | Marketing | Low risk (control case) |
| DataCore ERP | ERP | Forced upgrade, AMC escalation |

---

*Built for the **TechEx Intelligent Enterprise Solutions Hackathon 2026** on [lablab.ai](https://lablab.ai)*
