import os
import streamlit as st

_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if _api_key:
    os.environ["GOOGLE_API_KEY"] = _api_key

from agent import run_agent
from ingestion import ingest_single_file
from utils.vendor_lookup import lookup_vendor_risk, VendorRiskReport

# ── Contract registry ──────────────────────────────────────────────────────────

CONTRACTS = {
    "All Contracts": "",
    "CloudSync SaaS": "contract_01_cloudsync_saas.pdf",
    "NexStaff IT Staffing": "contract_02_nexstaff_it_staffing.pdf",
    "SwiftLog Logistics": "contract_03_swiftlog_logistics.pdf",
    "BrandCraft Marketing": "contract_04_brandcraft_marketing.pdf",
    "DataCore ERP": "contract_05_datacore_erp.pdf",
}

VENDOR_NAMES = {
    "contract_01_cloudsync_saas.pdf":       "CloudSync",
    "contract_02_nexstaff_it_staffing.pdf": "NexStaff",
    "contract_03_swiftlog_logistics.pdf":   "SwiftLog",
    "contract_04_brandcraft_marketing.pdf": "BrandCraft",
    "contract_05_datacore_erp.pdf":         "DataCore",
}

SEVERITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Procurement Intelligence Agent",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .risk-card {
        border-left: 4px solid #e0e0e0;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 4px;
        background: #fafafa;
    }
    .risk-card.high   { border-left-color: #d32f2f; background: #fff5f5; }
    .risk-card.medium { border-left-color: #f57c00; background: #fffbf0; }
    .risk-card.low    { border-left-color: #388e3c; background: #f5fff6; }
    .risk-type { font-weight: 600; font-size: 0.95rem; }
    .risk-meta { font-size: 0.78rem; color: #666; margin-top: 2px; }
    .source-pill {
        display: inline-block;
        background: #e8eaf6;
        color: #3949ab;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px 3px 2px 0;
    }
    div[data-testid="stHorizontalBlock"] button { font-size: 0.85rem; }
    .vendor-card {
        border-left: 4px solid #9e9e9e;
        padding: 10px 12px;
        border-radius: 4px;
        background: #f8f8f8;
        margin-top: 8px;
        font-size: 0.82rem;
    }
    .vendor-card.positive   { border-left-color: #388e3c; background: #f5fff6; }
    .vendor-card.neutral    { border-left-color: #757575; background: #f8f8f8; }
    .vendor-card.concerning { border-left-color: #d32f2f; background: #fff5f5; }
    .sentiment-badge {
        border-radius: 10px;
        padding: 1px 8px;
        font-size: 0.72rem;
        font-weight: 500;
        color: white;
        margin-left: 6px;
    }
    .sentiment-badge.positive   { background: #388e3c; }
    .sentiment-badge.neutral    { background: #757575; }
    .sentiment-badge.concerning { background: #d32f2f; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_contracts" not in st.session_state:
    st.session_state.uploaded_contracts = {}
    try:
        import chromadb
        from pathlib import Path
        _client = chromadb.PersistentClient(path=str(Path("data/embeddings")))
        _col = _client.get_or_create_collection(name="contracts")
        if _col.count() > 0:
            _static_filenames = set(CONTRACTS.values())
            _all_meta = _col.get(include=["metadatas"])["metadatas"]
            for _fname in {m.get("source", "") for m in _all_meta} - _static_filenames:
                if _fname:
                    _label = _fname.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
                    st.session_state.uploaded_contracts[_label] = _fname
    except Exception:
        pass

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

if "vendor_risk_result" not in st.session_state:
    st.session_state.vendor_risk_result = None
if "vendor_risk_for" not in st.session_state:
    st.session_state.vendor_risk_for = ""

if "compare_results" not in st.session_state:
    st.session_state.compare_results = {}
if "compare_mode_last" not in st.session_state:
    st.session_state.compare_mode_last = "risk_scan"
if "compare_query_last" not in st.session_state:
    st.session_state.compare_query_last = ""

# ── Vendor risk card (rendered in sidebar) ────────────────────────────────────

def _render_vendor_risk_card(report: VendorRiskReport):
    snt = report.overall_sentiment
    st.markdown(f"""
<div class="vendor-card {snt}">
  <div style="font-weight:600;font-size:0.88rem;">
    {report.vendor_name}
    <span class="sentiment-badge {snt}">{snt.title()}</span>
  </div>
  <div style="margin-top:6px;color:#333;">{report.search_summary}</div>
</div>
""", unsafe_allow_html=True)
    if report.risk_indicators:
        st.markdown(
            "<div style='font-size:0.78rem;font-weight:600;margin-top:8px;'>Risk signals:</div>",
            unsafe_allow_html=True,
        )
        for indicator in report.risk_indicators:
            st.markdown(
                f"<div style='font-size:0.76rem;color:#555;padding-left:6px;margin-top:2px;'>· {indicator}</div>",
                unsafe_allow_html=True,
            )


# ── Sidebar (shared across both tabs) ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Procurement Agent")
    st.markdown("---")

    all_contracts = {**CONTRACTS, **st.session_state.uploaded_contracts}
    selected_label = st.selectbox(
        "Contract (chat scope)",
        list(all_contracts.keys()),
        help="Restricts Chat tab queries to this contract. Has no effect on the Compare tab.",
    )
    source_filter = all_contracts[selected_label]

    if source_filter:
        st.caption(f"Scope: `{source_filter}`")
    else:
        total = len(all_contracts) - 1
        st.caption(f"Scope: all {total} contracts")

    st.markdown("---")

    # ── Research Vendor ────────────────────────────────────────────────────────
    if source_filter:
        vendor_name = VENDOR_NAMES.get(
            source_filter,
            source_filter.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title(),
        )

        # Clear stale result when the user switches to a different contract
        if st.session_state.vendor_risk_for != source_filter:
            st.session_state.vendor_risk_result = None

        research_clicked = st.button(
            f"🔎 Research {vendor_name}",
            use_container_width=True,
            key="research_vendor_btn",
        )
        if research_clicked:
            with st.spinner(f"Searching for {vendor_name} news…"):
                try:
                    st.session_state.vendor_risk_result = lookup_vendor_risk(vendor_name)
                    st.session_state.vendor_risk_for = source_filter
                except Exception as e:
                    st.error(f"Vendor research failed: {e}")

        if st.session_state.vendor_risk_result is not None:
            _render_vendor_risk_card(st.session_state.vendor_risk_result)
    else:
        st.button("🔎 Research Vendor", use_container_width=True, disabled=True,
                  help="Select a specific contract to research its vendor.")

    st.markdown("---")
    st.markdown("**Upload contract**")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if uploaded_file.name in CONTRACTS.values():
            st.caption(f"✓ Already indexed: `{uploaded_file.name}`")
        elif uploaded_file.name not in st.session_state.ingested_files:
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                count = ingest_single_file(uploaded_file.getvalue(), uploaded_file.name)
            st.session_state.ingested_files.add(uploaded_file.name)
            label = uploaded_file.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
            if count > 0:
                st.session_state.uploaded_contracts[label] = uploaded_file.name
                st.success(f"✅ Indexed — {count} chunks added.")
            else:
                st.session_state.uploaded_contracts[label] = uploaded_file.name
                st.caption(f"✓ Already indexed: `{uploaded_file.name}`")
            st.rerun()
        else:
            st.caption(f"✓ Already indexed: `{uploaded_file.name}`")

    st.markdown("---")
    st.markdown("**Risk taxonomy**")
    for tag in [
        "unlimited_liability", "auto_renewal_trap", "ip_ownership_risk",
        "inadequate_liability_cap", "missing_sla", "data_lock_in",
        "payment_penalty", "forced_upgrade_risk", "unilateral_change_right",
        "short_claims_window",
    ]:
        st.caption(f"· {tag.replace('_', ' ')}")

    st.markdown("---")
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Shared render helpers ──────────────────────────────────────────────────────

def render_sources(sources: list[str]):
    if not sources:
        return
    pills = "".join(f'<span class="source-pill">{s}</span>' for s in sources)
    st.markdown(f"<div style='margin-top:8px'>{pills}</div>", unsafe_allow_html=True)


def render_risk_flags(flags: list[dict], needs_review: bool, sources: list[str]):
    if needs_review:
        st.warning("⚠️ **Human review required** — 2 or more high-severity risks detected.", icon=None)

    if not flags:
        st.info("No risk flags found in the retrieved contract sections.")
        render_sources(sources)
        return

    sorted_flags = sorted(flags, key=lambda f: SEVERITY_ORDER.get(f.get("severity", "low"), 2))

    counts = {"high": 0, "medium": 0, "low": 0}
    for f in sorted_flags:
        counts[f.get("severity", "low")] += 1

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High", counts["high"])
    c2.metric("🟡 Medium", counts["medium"])
    c3.metric("🟢 Low", counts["low"])
    st.markdown("")

    for flag in sorted_flags:
        sev = flag.get("severity", "low")
        icon = SEVERITY_ICON.get(sev, "⚪")
        rtype = flag.get("risk_type", "unknown").replace("_", " ").title()
        summary = flag.get("clause_summary", "")
        rec = flag.get("recommendation", "")
        src = flag.get("source", "")
        page = flag.get("page", "")
        st.markdown(f"""
<div class="risk-card {sev}">
  <div class="risk-type">{icon} {rtype}</div>
  <div style="margin-top:5px">{summary}</div>
  <div class="risk-meta">📄 {src} · Page {page} &nbsp;|&nbsp; <em>Recommendation:</em> {rec}</div>
</div>
""", unsafe_allow_html=True)

    render_sources(sources)


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))
        elif msg["type"] == "risks":
            render_risk_flags(msg["content"], msg.get("needs_review", False), msg.get("sources", []))
        elif msg["type"] == "memo":
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))
        elif msg["type"] == "error":
            st.error(msg["content"])


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_chat, tab_compare = st.tabs(["💬 Chat", "🔀 Compare Contracts"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.markdown(f"### {selected_label}")

    for msg in st.session_state.messages:
        render_message(msg)

    # Memo download button
    _memo_msgs = [m for m in st.session_state.messages if m.get("type") == "memo"]
    if _memo_msgs:
        from utils.pdf_export import generate_memo_pdf
        _pdf_bytes = generate_memo_pdf(_memo_msgs[-1]["content"])
        st.download_button(
            label="⬇️ Download Memo as PDF",
            data=_pdf_bytes,
            file_name="approval_memo.pdf",
            mime="application/pdf",
        )

    st.markdown("---")
    query = st.text_input(
        "Query",
        placeholder="e.g. What is the auto-renewal notice period?",
        label_visibility="collapsed",
        key="query_input",
    )

    col1, col2, col3 = st.columns(3)
    ask_clicked  = col1.button("💬 Ask Question",  use_container_width=True)
    risk_clicked = col2.button("🔍 Run Risk Scan", use_container_width=True)
    memo_clicked = col3.button("📝 Draft Memo",    use_container_width=True)

    DEFAULT_QUERIES = {
        "risk_scan": "Scan all clauses for procurement risks and red flags.",
        "draft_memo": "Draft an approval memo summarising the key terms and risks.",
    }

    def run_and_append(effective_query: str, override_intent: str):
        st.session_state.messages.append({
            "role": "user", "type": "text",
            "content": effective_query, "sources": [],
        })
        with st.spinner("Thinking…"):
            result = run_agent(
                query=effective_query,
                source_filter=source_filter,
                override_intent=override_intent,
            )
        intent = result.get("intent", "qa")
        sources = result.get("sources", [])

        if intent == "qa":
            answer = result.get("answer", "").strip() or "No relevant information found in the selected contract."
            st.session_state.messages.append({
                "role": "assistant", "type": "text",
                "content": answer, "sources": sources,
            })
        elif intent == "risk_scan":
            st.session_state.messages.append({
                "role": "assistant", "type": "risks",
                "content": result.get("risk_flags", []),
                "needs_review": result.get("needs_human_review", False),
                "sources": sources,
            })
        elif intent == "draft_memo":
            memo = result.get("memo", "").strip() or "Could not generate a memo from the available contract sections."
            st.session_state.messages.append({
                "role": "assistant", "type": "memo",
                "content": memo, "sources": sources,
            })
        st.rerun()

    if ask_clicked:
        if not query.strip():
            st.warning("Please type a question before clicking Ask Question.")
        else:
            run_and_append(query.strip(), override_intent="qa")

    if risk_clicked:
        run_and_append(query.strip() or DEFAULT_QUERIES["risk_scan"], override_intent="risk_scan")

    if memo_clicked:
        run_and_append(query.strip() or DEFAULT_QUERIES["draft_memo"], override_intent="draft_memo")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE CONTRACTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("### Compare Contracts")

    # Named contracts only (exclude the "All Contracts" catch-all)
    selectable = {k: v for k, v in all_contracts.items() if v}

    selected_for_compare = st.multiselect(
        "Select 2–4 contracts to compare",
        list(selectable.keys()),
        default=list(selectable.keys())[:2],
        max_selections=4,
    )

    compare_mode = st.radio(
        "Mode",
        ["🔍 Risk Scan", "💬 Ask a Question"],
        horizontal=True,
    )

    compare_query = ""
    if "Ask" in compare_mode:
        compare_query = st.text_input(
            "Question to ask across all selected contracts",
            placeholder="e.g. What is the liability cap?",
            key="compare_query_input",
        )

    col_run, col_clear = st.columns([2, 1])
    run_compare = col_run.button(
        "▶ Run Comparison",
        disabled=len(selected_for_compare) < 2,
        type="primary",
        use_container_width=True,
    )
    clear_compare = col_clear.button(
        "🗑 Clear",
        use_container_width=True,
        key="clear_compare",
    )

    if len(selected_for_compare) < 2:
        st.caption("Select at least 2 contracts to enable comparison.")

    if clear_compare:
        st.session_state.compare_results = {}
        st.session_state.compare_mode_last = "risk_scan"
        st.session_state.compare_query_last = ""
        st.rerun()

    # ── Run comparison ─────────────────────────────────────────────────────────

    if run_compare:
        if "Ask" in compare_mode and not compare_query.strip():
            st.warning("Please enter a question to ask across contracts.")
        else:
            intent = "qa" if "Ask" in compare_mode else "risk_scan"
            query_text = (
                compare_query.strip()
                if intent == "qa"
                else "Scan all clauses for procurement risks and red flags."
            )

            progress_slot = st.empty()
            new_results = {}
            n = len(selected_for_compare)
            for i, label in enumerate(selected_for_compare):
                progress_slot.progress(
                    i / n,
                    text=f"Processing {label}… ({i + 1}/{n})",
                )
                new_results[label] = run_agent(
                    query=query_text,
                    source_filter=selectable[label],
                    override_intent=intent,
                )
            progress_slot.empty()

            st.session_state.compare_results = new_results
            st.session_state.compare_mode_last = intent
            st.session_state.compare_query_last = query_text if intent == "qa" else ""

    # ── Render results ─────────────────────────────────────────────────────────

    if st.session_state.compare_results:
        results = st.session_state.compare_results
        mode_last = st.session_state.compare_mode_last

        st.markdown("---")

        if mode_last == "qa" and st.session_state.compare_query_last:
            st.markdown(f"**Q: {st.session_state.compare_query_last}**")
            st.markdown("")

        cols = st.columns(len(results))

        for col, (label, result) in zip(cols, results.items()):
            with col:
                st.markdown(f"#### {label}")
                st.markdown("---")

                if mode_last == "risk_scan":
                    flags = result.get("risk_flags", [])
                    needs_review = result.get("needs_human_review", False)

                    if needs_review:
                        st.warning("⚠️ Human review required")

                    if not flags:
                        st.info("No risks found.")
                    else:
                        counts = {"high": 0, "medium": 0, "low": 0}
                        for f in flags:
                            counts[f.get("severity", "low")] += 1
                        m1, m2, m3 = st.columns(3)
                        m1.metric("🔴", counts["high"])
                        m2.metric("🟡", counts["medium"])
                        m3.metric("🟢", counts["low"])
                        st.markdown("")

                        for flag in sorted(flags, key=lambda f: SEVERITY_ORDER.get(f.get("severity", "low"), 2)):
                            sev = flag.get("severity", "low")
                            icon = SEVERITY_ICON.get(sev, "⚪")
                            rtype = flag.get("risk_type", "unknown").replace("_", " ").title()
                            summary = flag.get("clause_summary", "")
                            rec = flag.get("recommendation", "")
                            page = flag.get("page", "")
                            st.markdown(f"""
<div class="risk-card {sev}">
  <div class="risk-type">{icon} {rtype}</div>
  <div style="margin-top:4px;font-size:0.88rem">{summary}</div>
  <div class="risk-meta">Page {page} &nbsp;|&nbsp; <em>{rec}</em></div>
</div>
""", unsafe_allow_html=True)

                else:  # qa mode
                    answer = result.get("answer", "No answer found.").strip()
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    render_sources(sources)

        # ── Risk matrix (risk scan only) ───────────────────────────────────────

        if mode_last == "risk_scan":
            all_risk_types = sorted({
                f.get("risk_type", "unknown")
                for r in results.values()
                for f in r.get("risk_flags", [])
            })

            if all_risk_types:
                st.markdown("---")
                st.markdown("#### Risk Matrix")
                st.caption("Which risk types appear across contracts, and at what severity.")

                import pandas as pd
                sev_display = {"high": "🔴 High", "medium": "🟡 Med", "low": "🟢 Low"}
                matrix_data = {}
                for label, result in results.items():
                    by_type = {f["risk_type"]: f["severity"] for f in result.get("risk_flags", [])}
                    matrix_data[label] = [
                        sev_display.get(by_type.get(rt, ""), "—")
                        for rt in all_risk_types
                    ]
                df = pd.DataFrame(
                    matrix_data,
                    index=[rt.replace("_", " ").title() for rt in all_risk_types],
                )
                st.dataframe(df, use_container_width=True)
