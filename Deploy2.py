"""
Spam Detector — Streamlit Application
======================================
Multinomial Naive Bayes · scikit-learn · Plotly · Streamlit
Supports English / Arabic · Dark / Light themes
Persistent history (session + CSV export)
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import os
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# TRANSLATIONS
# ──────────────────────────────────────────────────────────────────────────────
TEXTS = {
    "EN": {
        "page_title":       "Spam Detector",
        "hero_sub":         "Multinomial Naive Bayes · Instant email & SMS classification",
        "model_overview":   "Model Overview",
        "accuracy_lbl":     "Accuracy",
        "precision_lbl":    "Precision",
        "recall_lbl":       "Recall",
        "f1_lbl":           "F1 Score",
        "total_lbl":        "Dataset Size",
        "ham_lbl":          "Legitimate",
        "spam_lbl":         "Spam",
        "tab_classify":     "🔍  Classifier",
        "tab_history":      "📋  History",
        "tab_dashboard":    "📊  Dashboard",
        "classify_sec":     "Classify a Message",
        "placeholder":      "Paste or type your email / SMS message here…",
        "analyse_btn":      "Analyse Message",
        "sample_title":     "Quick samples",
        "sample_spam":      "💣 Spam Example",
        "sample_ham":       "✉️ Ham Example",
        "step_vec":         "Vectorising text…",
        "step_cls":         "Running classifier…",
        "step_done":        "Done!",
        "result_spam":      "🚨  SPAM",
        "result_ham":       "✅  NOT SPAM",
        "desc_spam":        "This message has been flagged as spam. Do not click any links or share personal information.",
        "desc_ham":         "This message appears to be legitimate. No spam indicators were detected.",
        "prob_spam":        "Spam probability",
        "prob_ham":         "Ham probability",
        "your_msg":         "Your message",
        "chars":            "characters",
        "warn_empty":       "Please enter a message before analysing.",
        "history_title":    "Analysis History",
        "no_history":       "No messages analysed yet. Head to the Classifier tab to get started.",
        "clear_btn":        "🗑️  Clear History",
        "export_btn":       "⬇️  Export CSV",
        "hist_cleared":     "History cleared.",
        "col_time":         "Time",
        "col_result":       "Result",
        "col_conf":         "Confidence",
        "col_msg":          "Message",
        "col_len":          "Length",
        "dash_title":       "Dashboard",
        "dash_total":       "Total Analysed",
        "dash_spam":        "Spam Detected",
        "dash_ham":         "Legitimate",
        "dash_rate":        "Spam Rate",
        "chart_dist":       "Result Distribution",
        "chart_trend":      "Predictions Over Time",
        "chart_conf_hist":  "Confidence Distribution",
        "chart_len":        "Message Length by Category",
        "chart_conf_bar":   "Confidence per Analysis",
        "chart_heatmap":    "Confusion Matrix (Training)",
        "no_data":          "No data yet — analyse some messages first.",
        "lang_lbl":         "Language",
        "theme_lbl":        "Theme",
        "theme_dark":       "🌙 Dark",
        "theme_light":      "☀️ Light",
        "footer":           "Multinomial Naive Bayes · scikit-learn · Streamlit",
    },
    "AR": {
        "page_title":       "كاشف البريد المزعج",
        "hero_sub":         "Naive Bayes متعدد الحدود · تصنيف فوري للبريد والرسائل",
        "model_overview":   "نظرة عامة على النموذج",
        "accuracy_lbl":     "الدقة",
        "precision_lbl":    "الضبط",
        "recall_lbl":       "الاسترجاع",
        "f1_lbl":           "F1",
        "total_lbl":        "حجم البيانات",
        "ham_lbl":          "مشروع",
        "spam_lbl":         "مزعج",
        "tab_classify":     "🔍  التصنيف",
        "tab_history":      "📋  السجل",
        "tab_dashboard":    "📊  لوحة التحكم",
        "classify_sec":     "تصنيف رسالة",
        "placeholder":      "الصق رسالتك أو اكتبها هنا…",
        "analyse_btn":      "تحليل الرسالة",
        "sample_title":     "أمثلة سريعة",
        "sample_spam":      "💣 مثال مزعج",
        "sample_ham":       "✉️ مثال مشروع",
        "step_vec":         "تحويل النص إلى متجهات…",
        "step_cls":         "تشغيل المصنّف…",
        "step_done":        "اكتمل!",
        "result_spam":      "🚨  بريد مزعج",
        "result_ham":       "✅  بريد مشروع",
        "desc_spam":        "تم تصنيف هذه الرسالة كبريد مزعج. لا تنقر على أي روابط أو تشارك بياناتك الشخصية.",
        "desc_ham":         "تبدو هذه الرسالة مشروعة. لم يتم رصد أي مؤشرات بريد مزعج.",
        "prob_spam":        "احتمال البريد المزعج",
        "prob_ham":         "احتمال المشروعية",
        "your_msg":         "رسالتك",
        "chars":            "حرف",
        "warn_empty":       "يرجى إدخال رسالة قبل التحليل.",
        "history_title":    "سجل التحليلات",
        "no_history":       "لم يتم تحليل أي رسائل بعد. انتقل إلى تبويب التصنيف للبدء.",
        "clear_btn":        "🗑️  مسح السجل",
        "export_btn":       "⬇️  تصدير CSV",
        "hist_cleared":     "تم مسح السجل.",
        "col_time":         "الوقت",
        "col_result":       "النتيجة",
        "col_conf":         "الثقة",
        "col_msg":          "الرسالة",
        "col_len":          "الطول",
        "dash_title":       "لوحة التحكم",
        "dash_total":       "إجمالي التحليلات",
        "dash_spam":        "مزعج",
        "dash_ham":         "مشروع",
        "dash_rate":        "نسبة البريد المزعج",
        "chart_dist":       "توزيع النتائج",
        "chart_trend":      "التوقعات عبر الزمن",
        "chart_conf_hist":  "توزيع درجات الثقة",
        "chart_len":        "طول الرسائل حسب التصنيف",
        "chart_conf_bar":   "درجة الثقة لكل تحليل",
        "chart_heatmap":    "مصفوفة الارتباك (التدريب)",
        "no_data":          "لا توجد بيانات بعد — قم بتحليل بعض الرسائل أولاً.",
        "lang_lbl":         "اللغة",
        "theme_lbl":        "المظهر",
        "theme_dark":       "🌙 داكن",
        "theme_light":      "☀️ فاتح",
        "footer":           "Naive Bayes متعدد الحدود · scikit-learn · Streamlit",
    },
}

SAMPLE_SPAM = (
    "WINNER!! As a valued customer you have been selected to receive a £900 prize. "
    "Call 09061701461 from a landline. Claim code: KL341. Valid 12 hours only."
)
SAMPLE_HAM = (
    "Hey, are you coming to the team meeting tomorrow at 10am? Let me know if you "
    "need the Zoom link again."
)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
for _k, _v in [("lang", "EN"), ("theme", "dark"), ("history", []), ("prefill", "")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

T    = TEXTS[st.session_state.lang]
dark = st.session_state.theme == "dark"
rtl  = st.session_state.lang  == "AR"
DIR  = "rtl" if rtl else "ltr"
TALIGN = "right" if rtl else "left"

# ──────────────────────────────────────────────────────────────────────────────
# THEME PALETTE
# ──────────────────────────────────────────────────────────────────────────────
if dark:
    BG          = "linear-gradient(135deg,#0f0c29,#302b63,#24243e)"
    CARD_BG     = "rgba(255,255,255,0.05)"
    CARD_BORDER = "rgba(255,255,255,0.12)"
    CARD_SHD    = ""
    TEXT_PRI    = "#ffffff"
    TEXT_MUT    = "rgba(255,255,255,0.55)"
    TEXT_LBL    = "rgba(255,255,255,0.42)"
    STAT_BG     = "rgba(255,255,255,0.05)"
    STAT_BORDER = "rgba(255,255,255,0.1)"
    STAT_VAL    = "#a78bfa"
    STAT_KEY    = "rgba(255,255,255,0.38)"
    PREV_BG     = "rgba(255,255,255,0.04)"
    PREV_TEXT   = "rgba(255,255,255,0.75)"
    INP_BG      = "rgba(255,255,255,0.06)"
    INP_BORDER  = "rgba(255,255,255,0.15)"
    RES_DESC    = "rgba(255,255,255,0.62)"
    HR          = "rgba(255,255,255,0.08)"
    FOOT_CLR    = "rgba(255,255,255,0.28)"
    HIST_ODD    = "rgba(255,255,255,0.025)"
    HIST_HOVER  = "rgba(255,255,255,0.055)"
    HIST_BDR    = "rgba(255,255,255,0.07)"
    BSP_BG      = "rgba(239,68,68,0.18)";  BSP_CLR = "#f87171"
    BHA_BG      = "rgba(52,211,153,0.18)"; BHA_CLR = "#34d399"
    MET_BG      = "rgba(255,255,255,0.05)"
    MET_BDR     = "rgba(255,255,255,0.1)"
    PLT_FONT    = "#e0e0e0"
    PLT_GRID    = "rgba(255,255,255,0.08)"
    PLT_PAPER   = "rgba(0,0,0,0)"
    GAUGE_BG    = "rgba(255,255,255,0.07)"
else:
    BG          = "linear-gradient(135deg,#f0f4ff,#ebebff,#f5f0ff)"
    CARD_BG     = "rgba(255,255,255,0.92)"
    CARD_BORDER = "rgba(99,102,241,0.13)"
    CARD_SHD    = "box-shadow:0 4px 20px rgba(99,102,241,0.08);"
    TEXT_PRI    = "#1e1b4b"
    TEXT_MUT    = "#6b7280"
    TEXT_LBL    = "#9ca3af"
    STAT_BG     = "rgba(99,102,241,0.06)"
    STAT_BORDER = "rgba(99,102,241,0.16)"
    STAT_VAL    = "#6366f1"
    STAT_KEY    = "#9ca3af"
    PREV_BG     = "rgba(99,102,241,0.05)"
    PREV_TEXT   = "#374151"
    INP_BG      = "rgba(99,102,241,0.04)"
    INP_BORDER  = "rgba(99,102,241,0.22)"
    RES_DESC    = "#4b5563"
    HR          = "rgba(99,102,241,0.14)"
    FOOT_CLR    = "#9ca3af"
    HIST_ODD    = "rgba(99,102,241,0.025)"
    HIST_HOVER  = "rgba(99,102,241,0.07)"
    HIST_BDR    = "rgba(99,102,241,0.1)"
    BSP_BG      = "rgba(239,68,68,0.1)";   BSP_CLR = "#dc2626"
    BHA_BG      = "rgba(16,185,129,0.1)";  BHA_CLR = "#059669"
    MET_BG      = "rgba(255,255,255,0.95)"
    MET_BDR     = "rgba(99,102,241,0.14)"
    PLT_FONT    = "#1e1b4b"
    PLT_GRID    = "rgba(99,102,241,0.1)"
    PLT_PAPER   = "rgba(0,0,0,0)"
    GAUGE_BG    = "rgba(99,102,241,0.05)"

PREV_BDL = f"border-{'right' if rtl else 'left'}:3px solid #8b5cf6;"
PREV_RAD = f"border-radius:{'10px 0 0 10px' if rtl else '0 10px 10px 0'};"

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Cairo:wght@400;500;600;700;800&display=swap');
*{{box-sizing:border-box;}}

[data-testid="stAppViewContainer"]{{
    background:{BG};min-height:100vh;
    direction:{DIR};
    font-family:{'Cairo' if rtl else 'Inter'},sans-serif;
}}
[data-testid="stHeader"]{{background:transparent;}}
[data-testid="stDecoration"]{{display:none;}}

/* ── Tabs ─── */
.stTabs [data-baseweb="tab-list"]{{
    gap:0.35rem;background:{CARD_BG};border:1px solid {CARD_BORDER};
    border-radius:14px;padding:0.32rem;backdrop-filter:blur(10px);
}}
.stTabs [data-baseweb="tab"]{{
    border-radius:10px;color:{TEXT_MUT};font-weight:600;
    font-size:0.87rem;padding:0.42rem 1rem;
}}
.stTabs [aria-selected="true"]{{
    background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
    color:white!important;box-shadow:0 4px 12px rgba(99,102,241,0.35);
}}
.stTabs [data-baseweb="tab-panel"]{{padding-top:1.2rem;}}

/* ── Card ─── */
.card{{
    background:{CARD_BG};border:1px solid {CARD_BORDER};
    border-radius:20px;padding:1.6rem 1.8rem;
    backdrop-filter:blur(12px);margin-bottom:1rem;{CARD_SHD}
}}

/* ── Hero ─── */
.hero-title{{
    font-size:2.4rem;font-weight:800;
    background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;text-align:center;line-height:1.2;margin-bottom:0.2rem;
}}
.hero-sub{{color:{TEXT_MUT};text-align:center;font-size:0.88rem;margin-bottom:1.1rem;}}

/* ── Section label ─── */
.section-label{{
    color:{TEXT_LBL};font-size:0.68rem;font-weight:600;
    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.42rem;
}}

/* ── Accuracy badge ─── */
.accuracy-badge{{
    display:inline-block;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:white;padding:0.36rem 1.1rem;border-radius:50px;
    font-weight:700;font-size:1.05rem;box-shadow:0 4px 15px rgba(99,102,241,0.4);
}}

/* ── Stat row ─── */
.stat-row{{display:flex;gap:0.65rem;justify-content:center;flex-wrap:wrap;margin-top:0.35rem;}}
.stat-box{{
    background:{STAT_BG};border:1px solid {STAT_BORDER};border-radius:14px;
    padding:0.62rem 1rem;text-align:center;min-width:88px;
}}
.stat-value{{font-size:1.18rem;font-weight:700;color:{STAT_VAL};}}
.stat-key{{font-size:0.63rem;color:{STAT_KEY};text-transform:uppercase;letter-spacing:0.07em;}}

/* ── Metric card ─── */
.metric-card{{
    background:{MET_BG};border:1px solid {MET_BDR};
    border-radius:16px;padding:1rem 1.2rem;text-align:center;
    {CARD_SHD}margin-bottom:0.7rem;
}}
.metric-value{{font-size:1.85rem;font-weight:800;line-height:1.1;margin:0.22rem 0;}}
.metric-label{{font-size:0.68rem;color:{TEXT_LBL};text-transform:uppercase;letter-spacing:0.08em;}}

/* ── Result boxes ─── */
.result-spam{{
    background:linear-gradient(135deg,rgba(239,68,68,0.14),rgba(220,38,38,0.07));
    border:1px solid rgba(239,68,68,0.38);border-radius:16px;
    padding:1.4rem 1.8rem;text-align:center;margin:0.65rem 0;
}}
.result-ham{{
    background:linear-gradient(135deg,rgba(52,211,153,0.14),rgba(16,185,129,0.07));
    border:1px solid rgba(52,211,153,0.38);border-radius:16px;
    padding:1.4rem 1.8rem;text-align:center;margin:0.65rem 0;
}}
.result-label{{font-size:1.75rem;font-weight:800;color:{TEXT_PRI};margin-bottom:0.32rem;}}
.result-desc{{color:{RES_DESC};font-size:0.9rem;}}

/* ── Message preview ─── */
.message-preview{{
    background:{PREV_BG};{PREV_BDL}{PREV_RAD}
    padding:0.82rem 1.1rem;color:{PREV_TEXT};font-size:0.87rem;
    font-style:italic;margin-top:0.65rem;word-break:break-word;direction:{DIR};
}}

/* ── Char counter ─── */
.char-counter{{
    color:{TEXT_LBL};font-size:0.71rem;text-align:{'left' if rtl else 'right'};
    margin-top:-0.2rem;margin-bottom:0.5rem;
}}

/* ── History table ─── */
.hist-table{{width:100%;border-collapse:collapse;font-size:0.85rem;direction:{DIR};}}
.hist-table th{{
    color:{TEXT_LBL};font-size:0.66rem;text-transform:uppercase;letter-spacing:0.08em;
    padding:0.52rem 0.9rem;border-bottom:1px solid {HIST_BDR};
    font-weight:600;text-align:{TALIGN};
}}
.hist-table td{{
    padding:0.68rem 0.9rem;color:{TEXT_PRI};border-bottom:1px solid {HIST_BDR};
    text-align:{TALIGN};vertical-align:middle;
}}
.hist-table tr:nth-child(odd) td{{background:{HIST_ODD};}}
.hist-table tr:hover td{{background:{HIST_HOVER};transition:background 0.15s;}}

.badge-spam{{display:inline-block;background:{BSP_BG};color:{BSP_CLR};
    padding:0.17rem 0.62rem;border-radius:50px;font-size:0.74rem;font-weight:700;white-space:nowrap;}}
.badge-ham{{display:inline-block;background:{BHA_BG};color:{BHA_CLR};
    padding:0.17rem 0.62rem;border-radius:50px;font-size:0.74rem;font-weight:700;white-space:nowrap;}}

/* ── Textarea ─── */
textarea{{
    background:{INP_BG}!important;border:1px solid {INP_BORDER}!important;
    border-radius:12px!important;color:{TEXT_PRI}!important;
    font-size:0.92rem!important;direction:{DIR}!important;
}}
textarea:focus{{border-color:#8b5cf6!important;box-shadow:0 0 0 2px rgba(139,92,246,0.22)!important;}}

/* ── Buttons ─── */
.stButton>button{{
    background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
    color:white!important;border:none!important;border-radius:12px!important;
    padding:0.56rem 1.8rem!important;font-weight:700!important;
    font-size:0.9rem!important;transition:all 0.2s!important;
    box-shadow:0 4px 14px rgba(99,102,241,0.35)!important;width:100%!important;
}}
.stButton>button:hover{{transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(99,102,241,0.5)!important;}}

/* ── Download button ─── */
[data-testid="stDownloadButton"]>button{{
    background:rgba(99,102,241,0.12)!important;
    color:#8b5cf6!important;border:1px solid rgba(139,92,246,0.3)!important;
    border-radius:12px!important;padding:0.42rem 1.2rem!important;
    font-weight:600!important;font-size:0.84rem!important;width:auto!important;
    box-shadow:none!important;transition:all 0.2s!important;
}}
[data-testid="stDownloadButton"]>button:hover{{
    background:rgba(99,102,241,0.22)!important;
    transform:translateY(-1px)!important;
}}

/* ── Radio widgets ─── */
[data-testid="stWidgetLabel"] p{{
    color:{TEXT_LBL}!important;font-size:0.66rem!important;
    font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.09em!important;
}}
.stRadio [role="radiogroup"]{{flex-direction:row!important;gap:0.2rem;flex-wrap:nowrap;}}

/* ── Progress bar ─── */
[data-testid="stProgressBar"]>div>div{{background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;}}

/* ── Alerts ─── */
[data-testid="stAlert"]{{border-radius:12px!important;}}

/* ── Footer ─── */
.page-footer{{
    text-align:center;color:{FOOT_CLR};font-size:0.73rem;
    margin-top:2rem;padding-top:0.9rem;border-top:1px solid {HR};
}}

/* ── Divider ─── */
.divider{{height:1px;background:{HR};margin:1rem 0;}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached across reruns)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load (or retrain) the Naive Bayes classifier and return
    (model, vectorizer, accuracy, precision, recall, f1, dataset_stats).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "spam.csv")
    pkl_path = os.path.join(base_dir, "naive_bayes_model.pkl")

    df = pd.read_csv(csv_path)
    df["label"] = df["Category"].map({"spam": 1, "ham": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        df["Message"], df["label"], train_size=0.8, random_state=42, stratify=df["label"]
    )

    vec = CountVectorizer(stop_words="english", min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    # Try to load the saved model; retrain if vocabulary doesn't match
    mdl = None
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                mdl = pickle.load(f)
            if mdl.feature_count_.shape[1] != len(vec.vocabulary_):
                mdl = None          # shape mismatch → retrain
        except Exception:
            mdl = None

    if mdl is None:
        mdl = MultinomialNB(alpha=0.1)
        mdl.fit(Xtr, y_train)

    y_pred = mdl.predict(Xte)
    acc  = accuracy_score(y_test, y_pred)
    rep  = classification_report(y_test, y_pred, output_dict=True)
    prec = rep["1"]["precision"]
    rec  = rep["1"]["recall"]
    f1   = rep["1"]["f1-score"]
    cm   = confusion_matrix(y_test, y_pred)

    ds = {
        "total": len(df),
        "ham":   int((df["label"] == 0).sum()),
        "spam":  int((df["label"] == 1).sum()),
    }

    # Avg message lengths per category (for dashboard)
    df["msg_len"] = df["Message"].str.len()
    len_stats = {
        "spam_mean": df[df["label"] == 1]["msg_len"].mean(),
        "ham_mean":  df[df["label"] == 0]["msg_len"].mean(),
        "spam_lens": df[df["label"] == 1]["msg_len"].tolist(),
        "ham_lens":  df[df["label"] == 0]["msg_len"].tolist(),
    }

    return mdl, vec, acc, prec, rec, f1, ds, cm, len_stats


with st.spinner("Loading model…"):
    model, vectorizer, model_acc, model_prec, model_rec, model_f1, ds, cm, len_stats = load_model()


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: consistent Plotly layout
# ──────────────────────────────────────────────────────────────────────────────
def _base_layout(**kwargs):
    return dict(
        paper_bgcolor=PLT_PAPER,
        plot_bgcolor=PLT_PAPER,
        font=dict(color=PLT_FONT, family="Inter, Cairo, sans-serif"),
        margin=dict(t=32, b=10, l=10, r=10),
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL HEADER  (title + language + theme toggles)
# ──────────────────────────────────────────────────────────────────────────────
c_title, _, c_lang, c_theme = st.columns([5, 2, 1, 1])

with c_title:
    st.markdown(f'<div class="hero-title">🛡️ {T["page_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{T["hero_sub"]}</div>', unsafe_allow_html=True)

with c_lang:
    lang_choice = st.radio(
        T["lang_lbl"], ["EN", "AR"],
        index=0 if st.session_state.lang == "EN" else 1,
        horizontal=True, key="r_lang",
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

with c_theme:
    dark_opt  = T["theme_dark"]
    light_opt = T["theme_light"]
    theme_choice = st.radio(
        T["theme_lbl"], [dark_opt, light_opt],
        index=0 if st.session_state.theme == "dark" else 1,
        horizontal=True, key="r_theme",
    )
    new_theme = "dark" if theme_choice == dark_opt else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_cls, tab_hist, tab_dash = st.tabs(
    [T["tab_classify"], T["tab_history"], T["tab_dashboard"]]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with tab_cls:
    _, cmain, _ = st.columns([1, 3, 1])
    with cmain:

        # ── Model overview card ───────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{T["model_overview"]}</div>', unsafe_allow_html=True)

        ca, cb = st.columns([1, 2])
        with ca:
            st.markdown(
                f'<div style="display:flex;flex-direction:column;align-items:center;'
                f'justify-content:center;height:100%;padding-top:0.35rem;">'
                f'<div class="section-label" style="margin-bottom:0.45rem;">{T["accuracy_lbl"]}</div>'
                f'<div class="accuracy-badge">{model_acc * 100:.2f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with cb:
            st.markdown(
                f'<div class="stat-row">'
                f'<div class="stat-box">'
                f'  <div class="stat-value">{ds["total"]:,}</div>'
                f'  <div class="stat-key">{T["total_lbl"]}</div>'
                f'</div>'
                f'<div class="stat-box">'
                f'  <div class="stat-value" style="color:#34d399;">{ds["ham"]:,}</div>'
                f'  <div class="stat-key">{T["ham_lbl"]}</div>'
                f'</div>'
                f'<div class="stat-box">'
                f'  <div class="stat-value" style="color:#f87171;">{ds["spam"]:,}</div>'
                f'  <div class="stat-key">{T["spam_lbl"]}</div>'
                f'</div>'
                f'<div class="stat-box">'
                f'  <div class="stat-value" style="color:#60a5fa;">{model_prec * 100:.1f}%</div>'
                f'  <div class="stat-key">{T["precision_lbl"]}</div>'
                f'</div>'
                f'<div class="stat-box">'
                f'  <div class="stat-value" style="color:#f59e0b;">{model_rec * 100:.1f}%</div>'
                f'  <div class="stat-key">{T["recall_lbl"]}</div>'
                f'</div>'
                f'<div class="stat-box">'
                f'  <div class="stat-value" style="color:#a78bfa;">{model_f1 * 100:.1f}%</div>'
                f'  <div class="stat-key">{T["f1_lbl"]}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Sample buttons ────────────────────────────────────────────────────
        st.markdown(f'<div class="section-label" style="margin-bottom:0.35rem;">{T["sample_title"]}</div>', unsafe_allow_html=True)
        sb1, sb2 = st.columns(2)
        with sb1:
            if st.button(T["sample_spam"], key="btn_sample_spam"):
                st.session_state.prefill = SAMPLE_SPAM
                st.rerun()
        with sb2:
            if st.button(T["sample_ham"], key="btn_sample_ham"):
                st.session_state.prefill = SAMPLE_HAM
                st.rerun()

        # ── Input card ────────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{T["classify_sec"]}</div>', unsafe_allow_html=True)

        user_input = st.text_area(
            "Message",
            value=st.session_state.prefill,
            placeholder=T["placeholder"],
            height=160,
            label_visibility="collapsed",
            key="msg_input",
        )

        char_count = len(user_input)
        st.markdown(
            f'<div class="char-counter">{char_count} {T["chars"]}</div>',
            unsafe_allow_html=True,
        )

        predict_btn = st.button(T["analyse_btn"], use_container_width=True, key="btn_analyse")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Prediction logic ──────────────────────────────────────────────────
        if predict_btn:
            # Clear prefill after use
            st.session_state.prefill = ""

            if not user_input.strip():
                st.warning(T["warn_empty"], icon="⚠️")
            else:
                bar = st.progress(0, text=T["step_vec"])
                for p in range(0, 55, 11):
                    time.sleep(0.04)
                    bar.progress(p, text=T["step_vec"])

                msg_vec = vectorizer.transform([user_input])

                for p in range(55, 95, 10):
                    time.sleep(0.04)
                    bar.progress(p, text=T["step_cls"])

                pred  = model.predict(msg_vec)[0]
                proba = model.predict_proba(msg_vec)[0]
                bar.progress(100, text=T["step_done"])
                time.sleep(0.15)
                bar.empty()

                is_spam = bool(pred == 1)
                sp_pct  = float(proba[1] * 100)
                ha_pct  = float(proba[0] * 100)

                # Append to session history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message":   user_input,
                    "preview":   user_input[:80] + ("…" if len(user_input) > 80 else ""),
                    "verdict":   "spam" if is_spam else "ham",
                    "sp_pct":    round(sp_pct, 2),
                    "ha_pct":    round(ha_pct, 2),
                    "msg_len":   len(user_input),
                })

                # Result box
                res_cls = "result-spam" if is_spam else "result-ham"
                lbl  = T["result_spam"] if is_spam else T["result_ham"]
                desc = T["desc_spam"]   if is_spam else T["desc_ham"]
                st.markdown(
                    f'<div class="{res_cls}">'
                    f'<div class="result-label">{lbl}</div>'
                    f'<div class="result-desc">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Probability bars
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(f'<div class="section-label">{T["prob_spam"]}</div>', unsafe_allow_html=True)
                    st.progress(min(int(sp_pct), 100), text=f"{sp_pct:.1f}%")
                with r2:
                    st.markdown(f'<div class="section-label">{T["prob_ham"]}</div>', unsafe_allow_html=True)
                    st.progress(min(int(ha_pct), 100), text=f"{ha_pct:.1f}%")

                # Message preview
                preview = user_input[:300] + ("…" if len(user_input) > 300 else "")
                st.markdown(
                    f'<div class="section-label" style="margin-top:0.9rem;">{T["your_msg"]}</div>'
                    f'<div class="message-preview">"{preview}"</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    history = st.session_state.history

    ht1, ht2, ht3 = st.columns([5, 1, 1])
    with ht1:
        st.markdown(
            f'<div style="font-size:1.18rem;font-weight:800;color:{TEXT_PRI};margin-bottom:0.55rem;">'
            f'{T["history_title"]}</div>',
            unsafe_allow_html=True,
        )
    with ht2:
        if history:
            # Build CSV in memory for download
            df_export = pd.DataFrame([
                {
                    T["col_time"]:   e["timestamp"],
                    T["col_result"]: T["result_spam"] if e["verdict"] == "spam" else T["result_ham"],
                    T["col_conf"]:   f"{e['sp_pct'] if e['verdict']=='spam' else e['ha_pct']:.1f}%",
                    T["col_len"]:    e.get("msg_len", len(e["message"])),
                    T["col_msg"]:    e["message"],
                }
                for e in history
            ])
            csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=T["export_btn"],
                data=csv_bytes,
                file_name="spam_history.csv",
                mime="text/csv",
                key="dl_csv",
            )
    with ht3:
        if history and st.button(T["clear_btn"], key="btn_clear"):
            st.session_state.history = []
            st.success(T["hist_cleared"])
            st.rerun()

    if not history:
        st.markdown(
            f'<div class="card" style="text-align:center;color:{TEXT_MUT};padding:2.5rem 1rem;">'
            f'{T["no_history"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        rows = "".join(
            f'<tr>'
            f'<td style="white-space:nowrap;font-size:0.74rem;color:{TEXT_MUT};">{e["timestamp"]}</td>'
            f'<td><span class="badge-{"spam" if e["verdict"]=="spam" else "ham"}">'
            f'{T["result_spam"] if e["verdict"]=="spam" else T["result_ham"]}</span></td>'
            f'<td style="font-size:0.8rem;">{e["sp_pct"] if e["verdict"]=="spam" else e["ha_pct"]:.1f}%</td>'
            f'<td style="font-size:0.8rem;color:{TEXT_MUT};">{e.get("msg_len", len(e["message"]))}</td>'
            f'<td style="max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{e["preview"]}</td>'
            f'</tr>'
            for e in reversed(history)
        )
        st.markdown(
            f'<div class="card" style="overflow-x:auto;">'
            f'<table class="hist-table"><thead><tr>'
            f'<th>{T["col_time"]}</th>'
            f'<th>{T["col_result"]}</th>'
            f'<th>{T["col_conf"]}</th>'
            f'<th>{T["col_len"]}</th>'
            f'<th>{T["col_msg"]}</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    history = st.session_state.history

    st.markdown(
        f'<div style="font-size:1.18rem;font-weight:800;color:{TEXT_PRI};margin-bottom:0.85rem;">'
        f'{T["dash_title"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Static dataset section (always visible) ──────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">{T["model_overview"]}</div>', unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)

    with dc1:
        # Donut: dataset class distribution
        st.markdown(f'<div class="section-label">{T["chart_dist"]}</div>', unsafe_allow_html=True)
        donut_ds = go.Figure(go.Pie(
            labels=[T["spam_lbl"], T["ham_lbl"]],
            values=[ds["spam"], ds["ham"]],
            hole=0.6,
            marker=dict(colors=["#ef4444", "#34d399"], line=dict(color=PLT_PAPER, width=2)),
            textinfo="label+percent",
            textfont=dict(color=PLT_FONT, size=12),
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        ))
        donut_ds.update_layout(**_base_layout(showlegend=False, height=260))
        st.plotly_chart(donut_ds, use_container_width=True, config={"displayModeBar": False})

    with dc2:
        # Confusion matrix heatmap (training data)
        st.markdown(f'<div class="section-label">{T["chart_heatmap"]}</div>', unsafe_allow_html=True)
        labels_cm = [T["ham_lbl"], T["spam_lbl"]]
        hm = go.Figure(go.Heatmap(
            z=cm,
            x=labels_cm,
            y=labels_cm,
            colorscale=[[0, GAUGE_BG], [1, "#6366f1"]],
            showscale=False,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        hm.update_layout(
            **_base_layout(height=260),
            xaxis=dict(title=dict(text="Predicted", font=dict(color=PLT_FONT)), color=PLT_FONT),
            yaxis=dict(title=dict(text="True", font=dict(color=PLT_FONT)), color=PLT_FONT),
        )
        st.plotly_chart(hm, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Message length comparison (always visible) ────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">{T["chart_len"]}</div>', unsafe_allow_html=True)

    n_bins = 40
    spam_lens_trunc = [min(l, 500) for l in len_stats["spam_lens"]]
    ham_lens_trunc  = [min(l, 500) for l in len_stats["ham_lens"]]

    len_fig = go.Figure()
    len_fig.add_trace(go.Histogram(
        x=spam_lens_trunc,
        name=T["spam_lbl"],
        marker_color="#ef4444",
        opacity=0.7,
        nbinsx=n_bins,
        hovertemplate=T["spam_lbl"] + " · length %{x}: %{y}<extra></extra>",
    ))
    len_fig.add_trace(go.Histogram(
        x=ham_lens_trunc,
        name=T["ham_lbl"],
        marker_color="#34d399",
        opacity=0.7,
        nbinsx=n_bins,
        hovertemplate=T["ham_lbl"] + " · length %{x}: %{y}<extra></extra>",
    ))
    len_fig.update_layout(
        **_base_layout(height=240, barmode="overlay"),
        xaxis=dict(showgrid=False, color=PLT_FONT, title=""),
        yaxis=dict(showgrid=True, gridcolor=PLT_GRID, color=PLT_FONT, title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(color=PLT_FONT)),
    )
    st.plotly_chart(len_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Live session analytics (requires history) ─────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not history:
        st.markdown(
            f'<div class="card" style="text-align:center;color:{TEXT_MUT};padding:2.5rem 1rem;">'
            f'{T["no_data"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        total    = len(history)
        sp_count = sum(1 for h in history if h["verdict"] == "spam")
        ha_count = total - sp_count
        sp_rate  = sp_count / total * 100

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        for col, val, lbl, clr in [
            (k1, str(total),        T["dash_total"], "#a78bfa"),
            (k2, str(sp_count),     T["dash_spam"],  "#f87171"),
            (k3, str(ha_count),     T["dash_ham"],   "#34d399"),
            (k4, f"{sp_rate:.1f}%", T["dash_rate"],  "#60a5fa"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{lbl}</div>'
                    f'<div class="metric-value" style="color:{clr};">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        df_h = pd.DataFrame(history)
        df_h["ts"] = pd.to_datetime(df_h["timestamp"])

        # ── Row 1: donut + confidence bar ────────────────────────────────────
        ch_pie, ch_bar = st.columns([1, 2])

        with ch_pie:
            st.markdown(f'<div class="section-label">{T["chart_dist"]}</div>', unsafe_allow_html=True)
            donut = go.Figure(go.Pie(
                labels=[T["spam_lbl"], T["ham_lbl"]],
                values=[sp_count, ha_count],
                hole=0.58,
                marker=dict(colors=["#ef4444", "#34d399"], line=dict(color=PLT_PAPER, width=2)),
                textinfo="label+percent",
                textfont=dict(color=PLT_FONT, size=12),
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            donut.update_layout(**_base_layout(showlegend=False, height=268))
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

        with ch_bar:
            st.markdown(f'<div class="section-label">{T["chart_conf_bar"]}</div>', unsafe_allow_html=True)
            df_h["conf"] = df_h.apply(
                lambda r: r["sp_pct"] if r["verdict"] == "spam" else r["ha_pct"], axis=1
            )
            df_h["label_str"] = df_h["verdict"].map(
                {"spam": T["spam_lbl"], "ham": T["ham_lbl"]}
            )
            df_h["n"] = range(1, len(df_h) + 1)

            bar_fig = px.bar(
                df_h, x="n", y="conf", color="label_str",
                color_discrete_map={T["spam_lbl"]: "#ef4444", T["ham_lbl"]: "#34d399"},
                labels={"n": "#", "conf": T["col_conf"], "label_str": T["col_result"]},
                custom_data=["timestamp", "label_str"],
            )
            bar_fig.update_traces(
                hovertemplate="%{customdata[1]}<br>%{customdata[0]}<br>"
                              + T["col_conf"] + ": %{y:.1f}%<extra></extra>",
                marker_line_width=0,
            )
            bar_fig.update_layout(
                **_base_layout(height=268),
                xaxis=dict(showgrid=False, color=PLT_FONT, title=""),
                yaxis=dict(showgrid=True, gridcolor=PLT_GRID, color=PLT_FONT, range=[0, 107], title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(color=PLT_FONT)),
                bargap=0.18,
            )
            st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Row 2: timeline + confidence histogram ───────────────────────────
        ch_trend, ch_hist = st.columns(2)

        with ch_trend:
            st.markdown(f'<div class="section-label">{T["chart_trend"]}</div>', unsafe_allow_html=True)
            # Cumulative counts over time
            df_sorted = df_h.sort_values("ts")
            df_sorted["cum_spam"] = (df_sorted["verdict"] == "spam").cumsum()
            df_sorted["cum_ham"]  = (df_sorted["verdict"] == "ham").cumsum()

            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(
                x=df_sorted["ts"], y=df_sorted["cum_spam"],
                name=T["spam_lbl"], line=dict(color="#ef4444", width=2.5),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
                hovertemplate=T["spam_lbl"] + ": %{y}<extra></extra>",
            ))
            trend_fig.add_trace(go.Scatter(
                x=df_sorted["ts"], y=df_sorted["cum_ham"],
                name=T["ham_lbl"], line=dict(color="#34d399", width=2.5),
                fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
                hovertemplate=T["ham_lbl"] + ": %{y}<extra></extra>",
            ))
            trend_fig.update_layout(
                **_base_layout(height=240),
                xaxis=dict(showgrid=False, color=PLT_FONT, title=""),
                yaxis=dict(showgrid=True, gridcolor=PLT_GRID, color=PLT_FONT, title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(color=PLT_FONT)),
            )
            st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

        with ch_hist:
            st.markdown(f'<div class="section-label">{T["chart_conf_hist"]}</div>', unsafe_allow_html=True)
            conf_hist = go.Figure()
            conf_hist.add_trace(go.Histogram(
                x=df_h[df_h["verdict"] == "spam"]["conf"],
                name=T["spam_lbl"], marker_color="#ef4444",
                opacity=0.75, nbinsx=10,
                hovertemplate=T["spam_lbl"] + " %{x:.0f}%: %{y}<extra></extra>",
            ))
            conf_hist.add_trace(go.Histogram(
                x=df_h[df_h["verdict"] == "ham"]["conf"],
                name=T["ham_lbl"], marker_color="#34d399",
                opacity=0.75, nbinsx=10,
                hovertemplate=T["ham_lbl"] + " %{x:.0f}%: %{y}<extra></extra>",
            ))
            conf_hist.update_layout(
                **_base_layout(height=240, barmode="overlay"),
                xaxis=dict(showgrid=False, color=PLT_FONT, title="", range=[0, 105]),
                yaxis=dict(showgrid=True, gridcolor=PLT_GRID, color=PLT_FONT, title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(color=PLT_FONT)),
            )
            st.plotly_chart(conf_hist, use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="page-footer">{T["footer"]}</div>', unsafe_allow_html=True)
