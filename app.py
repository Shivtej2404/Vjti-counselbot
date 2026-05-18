import numpy as np
import pandas as pd
import tensorflow
import keras
import tensorflow.keras.models
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
from PIL import Image
import streamlit as st
from bokeh.plotting import figure
import math
from bokeh.palettes import Greens, Category10
from bokeh.transform import cumsum
from bokeh.models import LabelSet, ColumnDataSource
from bokeh.embed import file_html
from bokeh.resources import CDN

import string
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VJTI CounselBot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.main-title {
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0;
}

.chat-container {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.3) !important;
}

.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stSelectbox > div > div > div {
    color: white !important;
}

h1, h2, h3, h4 {
    color: #e2e8f0 !important;
}

.stSuccess {
    background: rgba(52,211,153,0.15) !important;
    border: 1px solid rgba(52,211,153,0.4) !important;
    border-radius: 10px !important;
}

.section-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}

.question-header {
    color: #a78bfa !important;
    font-weight: 600;
}

.stMarkdown p {
    color: #cbd5e1;
}

div[data-testid="stHeader"] {
    color: #a78bfa !important;
}

.results-title {
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_resource(show_spinner=False)
def load_resources():
    mdl = load_model(str(BASE / 'botmodel.h5'))
    tk = joblib.load(str(BASE / 'tokenizer_t.pkl'))
    wds = joblib.load(str(BASE / 'words.pkl'))
    df_bot = pd.read_csv(str(BASE / 'bot.csv'))
    df_intents = pd.read_csv(str(BASE / 'intents.csv'))
    df_q10 = pd.read_csv(str(BASE / 'questions_g10.csv'))
    df_q12 = pd.read_csv(str(BASE / 'questions_g12.csv'))
    df_qug = pd.read_csv(str(BASE / 'questions_ug.csv'))
    return mdl, tk, wds, df_bot, df_intents, df_q10, df_q12, df_qug

model, tok_global, words_global, df2, df_intents, df_q10, df_q12, df_qug = load_resources()

# ─── Session State ────────────────────────────────────────────────────────────
if 'test_started' not in st.session_state:
    st.session_state.test_started = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # list of {"role":"user"|"bot", "text":"..."}
if 'clear_chat' not in st.session_state:
    st.session_state.clear_chat = False

# ─── NLP Helpers (for ML fallback) ──────────────────────────────────────────
lem = WordNetLemmatizer()

def _tokenize(x):
    tokens = x.split()
    rep = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [rep.sub('', i) for i in tokens]
    tokens = [i for i in tokens if i.isalpha()]
    tokens = [lem.lemmatize(i.lower()) for i in tokens]
    return [i.lower() for i in tokens if len(i) > 1]

def _ml_answer(text):
    """ML model fallback — used only when keyword engine finds no match."""
    joined = ' '.join(_tokenize(text))
    enc = tok_global.texts_to_sequences([joined])
    padded = pad_sequences(enc, maxlen=16, padding='post')
    pred = int(np.argmax(model.predict(padded, verbose=0)))
    
    # Filter out stop words to see if we have actual meaningful words
    toks = _tokenize(text)
    stop_words = set(stopwords.words('english'))
    generic_verbs = {'tell', 'know', 'give', 'make', 'want', 'need', 'like'}
    content_words = [w for w in toks if w not in stop_words and w not in generic_verbs]
    
    # if no known specific content words in input, do not use ML model
    if not content_words or not any(w in words_global for w in content_words):
        return None
        
    try:
        grp = df2.groupby('labels').get_group(pred)
        return list(grp.bot)[np.random.randint(0, grp.shape[0])]
    except Exception:
        return None

# ─── Keyword-Intent Engine (CSV-driven) ───────────────────────────────────────────────
import random

def _build_intent_map(df):
    """Build list of (keywords_list, response) from intents.csv."""
    result = []
    for _, row in df.iterrows():
        kws = [k.strip() for k in str(row['keywords']).split('|') if k.strip()]
        resp = str(row['response']).replace('\\n', '\n')
        result.append((kws, resp))
    return result

INTENT_MAP = _build_intent_map(df_intents)


def keyword_match(text):
    """Return response if any keyword phrase found in lowercased input."""
    tl = text.lower()
    for keywords, response in INTENT_MAP:
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', tl):
                return response
    return None


def botans(text):
    """Primary: CSV keyword engine. Fallback: ML model."""
    ans = keyword_match(text)
    if ans:
        return ans
    ml_ans = _ml_answer(text)
    if ml_ans:
        return ml_ans
    return ("I'm not sure I understood that. Could you rephrase? 🤔\n\n"
            "You can ask me about:\n"
            "• Careers after 10th / 12th / graduation\n"
            "• Specific fields: Engineering, Medicine, Law, Commerce, Arts, Design\n"
            "• Entrance exams: JEE, NEET, CLAT, CAT, UPSC\n"
            "• Salaries, scholarships, internships, skills\n\n"
            "Or type **'start my test'** for a personalised career personality test! 🎓")



# ─── Pie Chart Helper ────────────────────────────────────────────────────────
def make_pie(title, labels_list, data):
    graph = figure(title=title, height=500, width=500)
    radians = [math.radians((p / 100) * 360) for p in data]
    start_angle = [math.radians(0)]
    prev = start_angle[0]
    for r in radians[:-1]:
        start_angle.append(r + prev)
        prev = r + prev
    end_angle = start_angle[1:] + [math.radians(360)]
    n = len(labels_list)
    color_palette = Category10[max(3, min(n, 10))]
    colors = color_palette[:n]
    graph.xgrid.visible = False
    graph.ygrid.visible = False
    graph.xaxis.visible = False
    graph.yaxis.visible = False
    for i in range(n):
        graph.wedge(0, 0, 0.8,
                    start_angle=start_angle[i],
                    end_angle=end_angle[i],
                    color=colors[i],
                    legend_label=labels_list[i] + " - " + str(round(data[i])) + "%")
    graph.add_layout(graph.legend[0], 'right')
    return graph

def make_line(title, labels_list, y_list):
    x = ['2000', '2005', '2010', '2015', '2020']
    colors = ["#a78bfa", "#60a5fa", "#34d399", "#f59e0b", "#ef4444"]
    graph2 = figure(title=title, x_range=x)
    for i, (lbl, yy) in enumerate(zip(labels_list, y_list)):
        graph2.line(x, yy, line_color=colors[i], legend_label=lbl, line_width=2)
    graph2.add_layout(graph2.legend[0], 'right')
    return graph2

# ─── Score Calculator ────────────────────────────────────────────────────────
def compute_results(input_list, top_n=3):
    """
    Each answer directly maps to one field (1-indexed).
    Score = raw answer value (1=StronglyDisagree .. 5=StronglyAgree).
    Top-N fields are those with highest scores.
    Percentages are normalised across the top-N only, so they reflect
    how strongly the student leaned toward each recommended field.
    """
    # Build (field_index_1based, score) pairs
    scored = [(i + 1, v) for i, v in enumerate(input_list)]
    # Sort descending by score
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    top = scored_sorted[:top_n]
    indices = [t[0] for t in top]
    raw_scores = [t[1] for t in top]
    total = sum(raw_scores) if sum(raw_scores) > 0 else 1
    percentages = [round(s * 100 / total, 1) for s in raw_scores]
    return indices, percentages

def Convert(string):
    parts = string.split(",")
    return list(map(float, [p.strip() for p in parts if p.strip()]))

# ─── Question Selectbox Helper ────────────────────────────────────────────────
OPTIONS = ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]
QVALS = {"Select an Option": 0, "Strongly Agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2, "Strongly Disagree": 1}

def ask_question(num, text, key):
    st.markdown(f"<h4 class='question-header'>Question {num}</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-card'><p style='color:#e2e8f0;font-size:1.05rem'>{text}</p></div>", unsafe_allow_html=True)
    return st.selectbox("Your answer", OPTIONS, key=key, label_visibility="collapsed")

# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Header
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("<h1 class='main-title'>🎓 VJTI's CounselBot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8;font-size:1.1rem;margin-top:0'>Your AI-powered Career Counseling Assistant</p>", unsafe_allow_html=True)
    with col2:
        try:
            logo = Image.open(str(BASE / "img/vjti_logo.png"))
            st.image(logo, use_container_width=True)
        except Exception:
            pass

    try:
        banner = Image.open(str(BASE / "img/21.png"))
        st.image(banner, use_container_width=True)
    except Exception:
        pass

    st.markdown("""
    <div class='chat-container'>
    <p style='color:#cbd5e1;font-size:1.05rem'>
    👋 Hi! I'm <strong style='color:#a78bfa'>VJTI's CounselBot</strong> — your personal career counseling assistant.
    Ask me anything about careers, streams, entrance exams, salaries, or scholarships.
    Type <strong style='color:#60a5fa'>"start my test"</strong> for a personalised personality assessment!
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Clear Chat Button ────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", key="clear_btn"):
        st.session_state.chat_history = []
        st.rerun()

    # ── Chat History Display ─────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["text"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["text"])

    # ── Chat Input ───────────────────────────────────────────────────────────
    user_input = st.chat_input("💬 Ask me anything — careers, streams, JEE, NEET, salary, scholarships…")
    if user_input:
        txt = user_input.strip()
        tl = txt.lower()
        if "start my test" in tl:
            st.session_state.test_started = True
        # Append user message
        st.session_state.chat_history.append({"role": "user", "text": txt})
        # Get and append bot reply
        reply = botans(tl)
        st.session_state.chat_history.append({"role": "bot", "text": reply})
        st.rerun()


    # ── Personality Test ───────────────────────────────────────────────────
    if st.session_state.test_started:
        st.markdown("---")
        st.markdown("<h2 style='color:#a78bfa'>📝 Personality Test</h2>", unsafe_allow_html=True)

        kr = st.selectbox("Would you like to begin the test?", ["Select an Option", "Yes", "No"], key='begin')

        if kr == "No":
            st.session_state.test_started = False
            st.info("Test cancelled. Feel free to ask me questions anytime!")

        elif kr == "Yes":
            kr1 = st.selectbox("📚 Select your level of education",
                               ["Select an Option", "Grade 10", "Grade 12", "Undergraduate"], key='edu_level')

            # ═══════════════ GRADE 10 ════════════════════════════════════════
            if kr1 == "Grade 10":
                lis = []
                questions = df_q10['question'].tolist()
                subjects = {i+1: s for i, s in enumerate(df_q10['subject'].tolist())}

                answers = []
                all_answered = True
                for i, q in enumerate(questions):
                    resp = ask_question(i + 1, q, key=f'g10_q{i+1}')
                    if resp == "Select an Option":
                        all_answered = False
                        break
                    answers.append(QVALS[resp])

                if all_answered and len(answers) == 10:
                    st.success("✅ Test Completed!")
                    st.markdown("<h2 class='results-title'>📊 Your Results</h2>", unsafe_allow_html=True)
                    df_sub = pd.read_csv(str(BASE / "Subjects.csv"))
                    l, data = compute_results(answers)

                    out = [subjects[n] for n in l]
                    pie = make_pie("Recommended Subjects", out, data)
                    st.components.v1.html(file_html(pie, CDN), height=600)

                    st.markdown("<h3 style='color:#60a5fa'>📖 More on the Subjects</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {subjects[int(l[i])]}"):
                            st.write(df_sub['about'][int(l[i]) - 1])

                    st.markdown("<h3 style='color:#60a5fa'>🎓 Choice of Degrees</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {subjects[int(l[i])]}"):
                            st.write(df_sub['further career'][int(l[i]) - 1])

                    st.markdown("<h3 style='color:#60a5fa'>📈 Trends Over the Years</h3>", unsafe_allow_html=True)
                    y = [Convert(df_sub['trends'][int(l[i]) - 1]) for i in range(4)]
                    line = make_line("Subject Trends", out, y)
                    st.components.v1.html(file_html(line, CDN), height=500)

                    st.markdown("<h3 style='color:#60a5fa'>📞 Expert Contacts</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {subjects[int(l[i])]}"):
                            xl = df_sub['contacts'][int(l[i]) - 1].split(",")
                            for k in xl:
                                st.write(k.strip())

            # ═══════════════ GRADE 12 ════════════════════════════════════════
            elif kr1 == "Grade 12":
                lis = []
                questions = df_q12['question'].tolist()
                streams = {i+1: s for i, s in enumerate(df_q12['stream'].tolist())}

                answers = []
                all_answered = True
                for i, q in enumerate(questions):
                    resp = ask_question(i + 1, q, key=f'g12_q{i+1}')
                    if resp == "Select an Option":
                        all_answered = False
                        break
                    answers.append(QVALS[resp])

                if all_answered and len(answers) == 10:
                    st.success("✅ Test Completed!")
                    st.markdown("<h2 class='results-title'>📊 Your Results</h2>", unsafe_allow_html=True)
                    df_grad = pd.read_csv(str(BASE / "Graduate.csv"))
                    l, data = compute_results(answers)

                    out = [streams[n] for n in l]
                    pie = make_pie("Recommended Fields", out, data)
                    st.components.v1.html(file_html(pie, CDN), height=600)

                    st.markdown("<h3 style='color:#60a5fa'>📖 More on the Fields</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {streams[int(l[i])]}"):
                            st.write(df_grad['About'][int(l[i]) - 1])

                    st.markdown("<h3 style='color:#60a5fa'>💰 Average Annual Salary</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {streams[int(l[i])]}"):
                            st.write("Rs. " + str(df_grad['avgsal'][int(l[i]) - 1]))

                    st.markdown("<h3 style='color:#60a5fa'>📈 Trends Over the Years</h3>", unsafe_allow_html=True)
                    y = [Convert(df_grad['trends'][int(l[i]) - 1]) for i in range(4)]
                    line = make_line("Field Trends", out, y)
                    st.components.v1.html(file_html(line, CDN), height=500)

                    st.markdown("<h3 style='color:#60a5fa'>📞 Expert Contacts</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {streams[int(l[i])]}"):
                            xl = df_grad['contacts'][int(l[i]) - 1].split(",")
                            for k in xl:
                                st.write(k.strip())

            # ═══════════════ UNDERGRADUATE ═══════════════════════════════════
            elif kr1 == "Undergraduate":
                questions = df_qug['question'].tolist()
                professions = {i+1: s for i, s in enumerate(df_qug['profession'].tolist())}

                answers = []
                all_answered = True
                for i, q in enumerate(questions):
                    resp = ask_question(i + 1, q, key=f'ug_q{i+1}')
                    if resp == "Select an Option":
                        all_answered = False
                        break
                    answers.append(QVALS[resp])

                if all_answered and len(answers) == 10:
                    st.success("✅ Test Completed!")
                    st.markdown("<h2 class='results-title'>📊 Your Results</h2>", unsafe_allow_html=True)
                    df_occ = pd.read_csv(str(BASE / "Occupations.csv"), encoding='windows-1252')
                    l, data = compute_results(answers)

                    out = [professions[n] for n in l]
                    pie = make_pie("Recommended Professions", out, data)
                    st.components.v1.html(file_html(pie, CDN), height=600)

                    st.markdown("<h3 style='color:#60a5fa'>📖 More on the Professions</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {professions[int(l[i])]}"):
                            st.write(df_occ['Information'][int(l[i]) - 1])

                    st.markdown("<h3 style='color:#60a5fa'>💰 Monthly Income</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {professions[int(l[i])]}"):
                            st.write("Rs. " + str(df_occ['Income'][int(l[i]) - 1]))

                    st.markdown("<h3 style='color:#60a5fa'>📈 Trends Over the Years</h3>", unsafe_allow_html=True)
                    y = [Convert(df_occ['trends'][int(l[i]) - 1]) for i in range(4)]
                    line = make_line("Profession Trends", out, y)
                    st.components.v1.html(file_html(line, CDN), height=500)

                    st.markdown("<h3 style='color:#60a5fa'>📞 Expert Contacts</h3>", unsafe_allow_html=True)
                    for i in range(4):
                        with st.expander(f"🔹 {professions[int(l[i])]}"):
                            xl = df_occ['contacts'][int(l[i]) - 1].split(",")
                            for k in xl:
                                st.write(k.strip())

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#475569;font-size:0.85rem'>"
        "🎓 VJTI CounselBot — Empowering Students with Smart Career Guidance"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
