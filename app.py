import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ─── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─── CUSTOM CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #18181f;
    --border: #26263a;
    --accent: #6c63ff;
    --accent2: #ff6b6b;
    --accent3: #43e97b;
    --accent4: #ffd166;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --green: #43e97b;
    --red: #ff6b6b;
    --gold: #ffd166;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
#MainMenu, footer { visibility: hidden; }
/* Hero */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6c63ff 0%, #a78bfa 50%, #ffd166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.02em;
}
/* Metric cards */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
    height: 100%;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6c63ff, #ffd166);
}
.metric-card:hover { border-color: var(--accent); }
.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.25rem;
}
/* Section */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.25rem 0;
}
.section-divider {
    height: 1px;
    background: var(--border);
    margin: 0.75rem 0 1.25rem 0;
}
/* Input overrides */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(108,99,255,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(108,99,255,0.5) !important;
}
/* Result cards */
.result-win {
    background: linear-gradient(135deg, rgba(67,233,123,0.12), rgba(67,233,123,0.04));
    border: 1.5px solid rgba(67,233,123,0.4);
    border-radius: 20px;
    padding: 1.75rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-loss {
    background: linear-gradient(135deg, rgba(255,107,107,0.12), rgba(255,107,107,0.04));
    border: 1.5px solid rgba(255,107,107,0.4);
    border-radius: 20px;
    padding: 1.75rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    margin-top: 0.5rem;
}
.result-pct {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    margin-top: 0.25rem;
}
.result-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.25rem;
}
/* Team logo ring */
.team-logo-wrap {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    background: var(--surface2);
    border: 2px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 0.75rem auto;
    overflow: hidden;
    padding: 6px;
}
/* Stats row */
.stat-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.stat-chip-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stat-chip-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 2px;
}
/* Insight banner */
.tip-box {
    background: rgba(108,99,255,0.1);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #c4b5fd;
    margin: 1rem 0 0 0;
}
.tip-box-gold {
    background: rgba(255,209,102,0.08);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: var(--gold);
    margin: 1rem 0 0 0;
}
.tip-box-red {
    background: rgba(255,107,107,0.08);
    border-left: 3px solid var(--red);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: var(--red);
    margin: 1rem 0 0 0;
}
/* Sidebar brand */
.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6c63ff, #ffd166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* Progress bar override */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6c63ff, #43e97b) !important;
    border-radius: 999px !important;
}
/* ─── NEW FEATURE STYLES ─── */

/* Chase Difficulty Badge */
.difficulty-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.25rem;
    border-radius: 999px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.difficulty-easy    { background: rgba(67,233,123,0.15); border: 1.5px solid #43e97b; color: #43e97b; }
.difficulty-moderate{ background: rgba(255,209,102,0.12); border: 1.5px solid #ffd166; color: #ffd166; }
.difficulty-hard    { background: rgba(255,107,107,0.12); border: 1.5px solid #ff6b6b; color: #ff6b6b; }
.difficulty-extreme { background: rgba(180,0,0,0.18); border: 1.5px solid #cc0000; color: #ff4444; }

/* Momentum Gauge */
.momentum-wrap {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.momentum-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #ff6b6b, #ffd166, #43e97b);
}
.momentum-bar-track {
    background: var(--border);
    border-radius: 999px;
    height: 12px;
    width: 100%;
    position: relative;
    margin: 0.75rem 0 0.35rem 0;
    overflow: hidden;
}
.momentum-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}
.momentum-center-tick {
    position: absolute;
    left: 50%;
    top: -3px;
    width: 2px;
    height: 18px;
    background: #ffffff30;
    transform: translateX(-50%);
}

/* Milestone card */
.milestone-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    height: 100%;
    min-height: 110px;
    box-sizing: border-box;
}
.milestone-icon {
    font-size: 1.75rem;
    min-width: 40px;
    text-align: center;
    padding-top: 2px;
}
.milestone-text {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.6;
}
.milestone-text b {
    color: var(--text);
    font-size: 1rem;
}
/* Force equal height columns for milestone row */
[data-testid="stHorizontalBlock"]:has(.milestone-card) > div {
    display: flex;
    flex-direction: column;
}
[data-testid="stHorizontalBlock"]:has(.milestone-card) > div > div {
    flex: 1;
    height: 100%;
}

/* Scenario table */
.scenario-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.scenario-table th {
    background: var(--surface);
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.scenario-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.scenario-table tr:last-child td { border-bottom: none; }
.scenario-table tr:hover td { background: rgba(108,99,255,0.05); }
.pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-green { background: rgba(67,233,123,0.15); color: #43e97b; }
.pill-red   { background: rgba(255,107,107,0.15); color: #ff6b6b; }
.pill-gold  { background: rgba(255,209,102,0.12); color: #ffd166; }

/* History sparkline area */
.sparkline-wrap {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.sparkline-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6c63ff, #43e97b);
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ─────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open('pipe.pkl', 'rb'))

pipe = load_model()

# ─── DATA ───────────────────────────────────────────
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Ahmedabad', 'Pune'
]
team_logos = {
    "Mumbai Indians":               "https://upload.wikimedia.org/wikipedia/en/c/cd/Mumbai_Indians_Logo.svg",
    "Chennai Super Kings":          "https://upload.wikimedia.org/wikipedia/en/thumb/2/2b/Chennai_Super_Kings_Logo.svg/500px-Chennai_Super_Kings_Logo.svg.png",
    "Royal Challengers Bangalore":  "https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Royal_Challengers_Bengaluru_Logo.svg/330px-Royal_Challengers_Bengaluru_Logo.svg.png",
    "Kolkata Knight Riders":        "https://upload.wikimedia.org/wikipedia/en/4/4c/Kolkata_Knight_Riders_Logo.svg",
    "Sunrisers Hyderabad":          "https://upload.wikimedia.org/wikipedia/en/thumb/5/51/Sunrisers_Hyderabad_Logo.svg/500px-Sunrisers_Hyderabad_Logo.svg.png",
    "Delhi Capitals":               "https://upload.wikimedia.org/wikipedia/en/2/2f/Delhi_Capitals.svg",
    "Rajasthan Royals":             "https://documents.iplt20.com/ipl/RR/Logos/RR_Logo.png",
    "Kings XI Punjab":              "https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.svg",
}
team_colors = {
    "Mumbai Indians":               "#004BA0",
    "Chennai Super Kings":          "#FDB913",
    "Royal Challengers Bangalore":  "#EC1C24",
    "Kolkata Knight Riders":        "#3A225D",
    "Sunrisers Hyderabad":          "#FF822A",
    "Delhi Capitals":               "#0078BC",
    "Rajasthan Royals":             "#254AA5",
    "Kings XI Punjab":              "#ED1C24",
}

# ─── HELPER: predict probability ─────────────────────
def get_win_prob(batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target, crr, rrr):
    if balls_left <= 0 or runs_left <= 0:
        return None
    inp = pd.DataFrame({
        'batting_team':  [batting_team],
        'bowling_team':  [bowling_team],
        'city':          [city],
        'runs_left':     [runs_left],
        'balls_left':    [balls_left],
        'wickets_left':  [wickets_left],
        'total_runs_x':  [target],
        'crr':           [crr],
        'rr':            [rrr],
    })
    try:
        return pipe.predict_proba(inp)[0][1]
    except:
        return None

# ─── SIDEBAR ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem 0;'>
        <div class='sidebar-brand'>🏏 IPL Oracle</div>
        <div style='font-size: 0.78rem; color: #6b6b80; margin-top: 2px;'>Live Win Probability Engine</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.82rem; color: #9090a8; line-height: 1.9;'>
        <b style='color: #c4b5fd;'>How it works</b><br>
        Enter the live match state — target, score, overs, and wickets — and the ML model instantly computes win probability for both teams.<br><br>
        <b style='color: #c4b5fd;'>Key factors</b><br>
        • Runs left vs balls left<br>
        • Current & required run rate<br>
        • Wickets in hand<br>
        • Historical team performance<br>
        • Venue advantage
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.78rem; color: #6b6b80;'>
        <b style='color: #9090a8;'>Model</b><br>
        Logistic Regression on IPL match data<br><br>
        <b style='color: #9090a8;'>Teams covered</b><br>
        8 IPL franchises · 11 venues
    </div>
    """, unsafe_allow_html=True)

# ─── HERO ───────────────────────────────────────────
st.markdown('<div class="hero-title">IPL Win<br>Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time win probability · Second-innings chase analysis</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─── TEAM SELECTION ─────────────────────────────────
st.markdown('<div class="section-title">🏟️ Match Setup</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    batting_team = st.selectbox("🏏 Batting Team", sorted(teams), key="bat")
with col2:
    bowling_team_options = [t for t in sorted(teams) if t != batting_team]
    bowling_team = st.selectbox("🎯 Bowling Team", bowling_team_options, key="bowl")
with col3:
    city = st.selectbox("📍 Match Venue", sorted(cities))

st.markdown("<br>", unsafe_allow_html=True)

# ─── MATCH STATE ────────────────────────────────────
st.markdown('<div class="section-title">📊 Live Match State</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
col4, col5, col6, col7 = st.columns(4)
with col4:
    target = st.number_input("🎯 Target", min_value=1, max_value=300, value=165,
                              help="Runs the batting team needs to win")
with col5:
    score = st.number_input("🏃 Current Score", min_value=0, max_value=300, value=78,
                             help="Runs scored so far")
with col6:
    overs = st.number_input("⏱️ Overs Completed", min_value=0.1, max_value=19.5,
                             value=10.0, step=0.1,
                             help="Overs bowled so far (e.g. 10.3 = 10 overs 3 balls)")
with col7:
    wickets = st.number_input("💀 Wickets Lost", min_value=0, max_value=9, value=3,
                               help="Wickets fallen so far")

# ─── VALIDATION ─────────────────────────────────────
runs_left    = target - score
balls_left   = 120 - int(overs * 6)
wickets_left = 10 - wickets
crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

has_error = False
if score >= target:
    st.error("⚠️ Current score has already reached or surpassed the target. Match is over!")
    has_error = True
if balls_left <= 0:
    st.error("⚠️ No balls remaining. Please check overs entered.")
    has_error = True

# ─── PREDICT BUTTON ─────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Win Probability", disabled=has_error)

# ─── LIVE CONTEXT CHIPS ─────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
chip1, chip2, chip3, chip4, chip5 = st.columns(5)
with chip1:
    st.markdown(f"""
    <div class="stat-chip">
        <div class="stat-chip-label">Runs Left 🏏</div>
        <div class="stat-chip-value" style="color:{'#ff6b6b' if runs_left > 80 else '#ffd166' if runs_left > 40 else '#43e97b'}">{max(runs_left,0)}</div>
    </div>""", unsafe_allow_html=True)
with chip2:
    st.markdown(f"""
    <div class="stat-chip">
        <div class="stat-chip-label">Balls Left ⏱️</div>
        <div class="stat-chip-value" style="color:{'#ff6b6b' if balls_left < 30 else '#ffd166' if balls_left < 60 else '#43e97b'}">{max(balls_left,0)}</div>
    </div>""", unsafe_allow_html=True)
with chip3:
    st.markdown(f"""
    <div class="stat-chip">
        <div class="stat-chip-label">Wickets Left 💀</div>
        <div class="stat-chip-value" style="color:{'#ff6b6b' if wickets_left <= 3 else '#ffd166' if wickets_left <= 6 else '#43e97b'}">{wickets_left}</div>
    </div>""", unsafe_allow_html=True)
with chip4:
    st.markdown(f"""
    <div class="stat-chip">
        <div class="stat-chip-label">Current RR 📊</div>
        <div class="stat-chip-value">{round(crr, 2)}</div>
    </div>""", unsafe_allow_html=True)
with chip5:
    rrr_color = "#ff6b6b" if rrr > 12 else "#ffd166" if rrr > 9 else "#43e97b"
    st.markdown(f"""
    <div class="stat-chip">
        <div class="stat-chip-label">Required RR 📈</div>
        <div class="stat-chip-value" style="color:{rrr_color}">{round(rrr, 2)}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# ─── PREDICTION RESULTS ─────────────────────────────
# ═══════════════════════════════════════════════════
if predict_clicked and not has_error:
    input_df = pd.DataFrame({
        'batting_team':  [batting_team],
        'bowling_team':  [bowling_team],
        'city':          [city],
        'runs_left':     [runs_left],
        'balls_left':    [balls_left],
        'wickets_left':  [wickets_left],
        'total_runs_x':  [target],
        'crr':           [crr],
        'rr':            [rrr],
    })
    result    = pipe.predict_proba(input_df)
    win_prob  = result[0][1]
    loss_prob = result[0][0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Win Probability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_bat, col_vs, col_bowl = st.columns([1, 0.15, 1])
    with col_bat:
        win_pct    = round(win_prob * 100, 1)
        card_class = "result-win" if win_prob >= 0.5 else "result-loss"
        pct_color  = "#43e97b" if win_prob >= 0.5 else "#ff6b6b"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="team-logo-wrap">
                <img src="{team_logos[batting_team]}" style="width:52px; height:52px; object-fit:contain;" />
            </div>
            <div style='font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em;'>Batting</div>
            <div class="result-title" style="color:{pct_color};">{batting_team}</div>
            <div class="result-pct" style="color:{pct_color};">{win_pct}%</div>
            <div class="result-sub">Win Probability</div>
        </div>""", unsafe_allow_html=True)
        st.progress(float(win_prob))

    with col_vs:
        st.markdown("""
        <div style='height:100%; display:flex; align-items:center; justify-content:center;
                    font-family: Syne, sans-serif; font-weight:800; font-size:1rem;
                    color:#6b6b80; padding-top: 60px;'>VS</div>
        """, unsafe_allow_html=True)

    with col_bowl:
        loss_pct   = round(loss_prob * 100, 1)
        card_class2 = "result-win" if loss_prob >= 0.5 else "result-loss"
        pct_color2  = "#43e97b" if loss_prob >= 0.5 else "#ff6b6b"
        st.markdown(f"""
        <div class="{card_class2}">
            <div class="team-logo-wrap">
                <img src="{team_logos[bowling_team]}" style="width:52px; height:52px; object-fit:contain;" />
            </div>
            <div style='font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em;'>Bowling</div>
            <div class="result-title" style="color:{pct_color2};">{bowling_team}</div>
            <div class="result-pct" style="color:{pct_color2};">{loss_pct}%</div>
            <div class="result-sub">Win Probability</div>
        </div>""", unsafe_allow_html=True)
        st.progress(float(loss_prob))

    # ─── CONTEXTUAL INSIGHT ─────────────────────────
    rr_ratio = rrr / crr if crr > 0 else 99
    if win_prob >= 0.70:
        insight_class, insight_icon, insight_msg = "tip-box", "🔥", \
            f"<b>{batting_team}</b> are in command. They need {max(runs_left,0)} off {max(balls_left,0)} balls with {wickets_left} wickets in hand — a very achievable ask at this RRR of <b>{round(rrr,2)}</b>."
    elif win_prob >= 0.50:
        insight_class, insight_icon, insight_msg = "tip-box-gold", "⚖️", \
            f"The chase is <b>on a knife edge</b>. {batting_team} need {max(runs_left,0)} off {max(balls_left,0)} balls. " \
            f"RRR ({round(rrr,2)}) is {round(rr_ratio,1)}x the current run rate — momentum matters here."
    else:
        insight_class, insight_icon, insight_msg = "tip-box-red", "💀", \
            f"<b>{batting_team}</b> face a steep climb — {max(runs_left,0)} needed off {max(balls_left,0)} balls at RRR {round(rrr,2)}. " \
            f"Only {wickets_left} wickets remain. <b>{bowling_team}</b> are clear favourites."
    st.markdown(f'<div class="{insight_class}">{insight_icon} {insight_msg}</div>', unsafe_allow_html=True)

# ─── PHASE ANALYSIS ────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Match Phase Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    overs_done = int(overs)
    if overs_done < 6:
        phase      = "Powerplay"
        phase_note = "Field restrictions in effect — expect aggressive batting."
    elif overs_done < 15:
        phase      = "Middle Overs"
        phase_note = "Dot balls can shift momentum rapidly. Partnership key."
    else:
        phase      = "Death Overs"
        phase_note = "High-risk, high-reward phase. Big hits or wickets expected."

    pa, pb, pc = st.columns(3)
    with pa:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Match Phase</div>
            <div class="metric-value" style="font-size:1.3rem; color:#a78bfa;">{phase}</div>
            <div class="metric-sub">{phase_note}</div>
        </div>""", unsafe_allow_html=True)
    with pb:
        pressure = min(round(rrr - 1, 1), 10)
        pressure = max(0, pressure)
        bar_color = "#43e97b" if pressure < 5 else "#ffd166" if pressure < 7.5 else "#ff6b6b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pressure Index</div>
            <div class="metric-value" style="color:{bar_color};">{min(pressure, 10.0):.1f}<span style="font-size:1rem;">/10</span></div>
            <div class="metric-sub">Based on RRR vs overs left</div>
        </div>""", unsafe_allow_html=True)
    with pc:
        run_diff  = round(crr - rrr, 2)
        diff_color = "#43e97b" if run_diff >= 0 else "#ff6b6b"
        diff_label = "ahead of rate" if run_diff >= 0 else "behind rate"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rate Differential</div>
            <div class="metric-value" style="color:{diff_color};">{abs(run_diff)}</div>
            <div class="metric-sub">CRR is {diff_label}</div>
        </div>""", unsafe_allow_html=True)


    # ═══════════════════════════════════════════════
    # ─── NEW FEATURE 1: CHASE DIFFICULTY BADGE ──────
    # ═══════════════════════════════════════════════
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Chase Difficulty Rating</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Difficulty score: composite of RRR, wickets, balls left
    difficulty_score = (rrr / 10) * 0.5 + (1 - wickets_left / 10) * 0.3 + (1 - balls_left / 120) * 0.2
    difficulty_score = min(difficulty_score, 1.0)

    if difficulty_score < 0.45:
        diff_label, diff_class, diff_icon, diff_desc = "Easy Chase", "difficulty-easy", "🟢", \
        f"RRR of {round(rrr,2)} is very manageable with {wickets_left} wickets in hand."
    
    elif difficulty_score < 0.65:
        diff_label, diff_class, diff_icon, diff_desc = "Moderate Chase", "difficulty-moderate", "🟡", \
        f"A competitive chase. Maintaining {round(rrr,2)} RRR will be key."

    elif difficulty_score < 0.85:
        diff_label, diff_class, diff_icon, diff_desc = "Tough Chase", "difficulty-hard", "🔴", \
        f"A tough ask — RRR {round(rrr,2)} requires aggressive batting."

    else:
        diff_label, diff_class, diff_icon, diff_desc = "Near Impossible", "difficulty-extreme", "💀", \
        f"Extremely unlikely. {runs_left} off {balls_left} balls at RRR {round(rrr,2)}."

    df_col1, df_col2 = st.columns([0.4, 1])
    with df_col1:
        st.markdown(f"""
        <div style='padding: 1.5rem; background: var(--surface2); border: 1px solid var(--border); border-radius: 16px; text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>{diff_icon}</div>
            <div class='difficulty-badge {diff_class}'>{diff_label}</div>
            <div style='font-size: 0.72rem; color: var(--muted); margin-top: 0.5rem;'>Difficulty Score: {round(difficulty_score * 10, 1)}/10</div>
        </div>
        """, unsafe_allow_html=True)
    with df_col2:
        st.markdown(f"""
        <div style='padding: 1.5rem; background: var(--surface2); border: 1px solid var(--border); border-radius: 16px; height: 100%;'>
            <div class='metric-label'>Assessment</div>
            <div style='font-size: 0.95rem; color: var(--text); line-height: 1.7; margin-top: 0.5rem;'>{diff_desc}</div>
            <div style='margin-top: 1rem;'>
                <div class='metric-label' style='margin-bottom: 0.4rem;'>Difficulty Meter</div>
                <div style='background: var(--border); border-radius: 999px; height: 10px; overflow: hidden;'>
                    <div style='width: {round(difficulty_score*100)}%; height: 100%; border-radius: 999px;
                        background: {"#43e97b" if difficulty_score < 0.35 else "#ffd166" if difficulty_score < 0.55 else "#ff6b6b" if difficulty_score < 0.75 else "#cc0000"};
                        transition: width 0.6s ease;'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # ─── NEW FEATURE 2: MOMENTUM GAUGE ──────────────
    # ═══════════════════════════════════════════════
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Match Momentum</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Momentum: batting side's momentum based on CRR vs RRR
    # > 50 means batting team has momentum, < 50 means bowling team
    if rrr > 0:
        momentum_raw = crr / rrr  # >1 = batting team ahead, <1 = behind
    else:
        momentum_raw = 2.0
    momentum_pct = min(max(momentum_raw / 2, 0), 1)  # normalise 0-1 where 0.5 = neutral

    if momentum_pct > 0.65:
        mom_label = f"🏏 {batting_team} in Control"
        mom_color = "#43e97b"
        mom_desc  = f"Batting side is scoring at {round(crr,2)} RPO vs required {round(rrr,2)} — they are ahead of the game."
    elif momentum_pct > 0.5:
        mom_label = f"⚖️ Slight edge to {batting_team}"
        mom_color = "#a3e635"
        mom_desc  = f"Marginally ahead of the rate. A wicket or dot-ball phase could shift momentum."
    elif momentum_pct == 0.5:
        mom_label = "⚖️ Perfectly Balanced"
        mom_color = "#ffd166"
        mom_desc  = "CRR exactly matches RRR. The next over is critical."
    elif momentum_pct > 0.35:
        mom_label = f"🎯 Slight edge to {bowling_team}"
        mom_color = "#fb923c"
        mom_desc  = f"Bowling side is applying pressure. {batting_team} need to up the scoring rate."
    else:
        mom_label = f"🎯 {bowling_team} Dominant"
        mom_color = "#ff6b6b"
        mom_desc  = f"Bowling side firmly in control. Required rate is significantly above current rate."

    bar_width = round(momentum_pct * 100)
    st.markdown(f"""
    <div class="momentum-wrap">
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <div class='metric-label'>Momentum Indicator</div>
                <div style='font-family: Syne, sans-serif; font-size: 1.1rem; font-weight: 700; color: {mom_color}; margin-top: 2px;'>{mom_label}</div>
            </div>
            <div style='text-align: right; font-size: 0.8rem; color: var(--muted);'>
                CRR {round(crr,2)} vs RRR {round(rrr,2)}
            </div>
        </div>
        <div class='momentum-bar-track'>
            <div class='momentum-bar-fill' style='width: {bar_width}%;
                background: linear-gradient(90deg, #ff6b6b, #ffd166, #43e97b);
                clip-path: inset(0 {100 - bar_width}% 0 0 round 999px);
                width: 100%;'></div>
            <div class='momentum-center-tick'></div>
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted);'>
            <span>🎯 {bowling_team} dominant</span>
            <span>Balanced</span>
            <span>🏏 {batting_team} dominant</span>
        </div>
        <div style='font-size: 0.84rem; color: var(--muted); margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border);'>{mom_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # ─── NEW FEATURE 3: KEY MILESTONES ──────────────
    # ═══════════════════════════════════════════════
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚩 Key Milestones</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Next 50-run mark
    next_50 = ((score // 50) + 1) * 50
    runs_to_next_50 = next_50 - score

    # Balls to reach target at current CRR
    if crr > 0:
        balls_to_win_at_crr = int(runs_left / (crr / 6))
    else:
        balls_to_win_at_crr = balls_left

    # Halfway point of the target
    halfway = target // 2
    halfway_done = score >= halfway

    # Overs remaining
    overs_remaining = (balls_left) / 6

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""
        <div class="milestone-card">
            <div class="milestone-icon">🏅</div>
            <div>
                <div class="milestone-text">Next 50-run mark</div>
                <div class="milestone-text"><b>{next_50} runs</b> — {runs_to_next_50} more needed</div>
                <div class="milestone-text" style="font-size:0.78rem;">≈ {round(runs_to_next_50 / (crr/6),1) if crr>0 else "?"} balls at CRR</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        hw_status = "✅ Crossed!" if halfway_done else f"{halfway - score} runs away"
        hw_color  = "#43e97b" if halfway_done else "#ffd166"
        st.markdown(f"""
        <div class="milestone-card">
            <div class="milestone-icon">⚖️</div>
            <div>
                <div class="milestone-text">Halfway to target ({halfway})</div>
                <div class="milestone-text"><b style="color:{hw_color};">{hw_status}</b></div>
                <div class="milestone-text" style="font-size:0.78rem;">Target: {target} runs</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        # Win-at-CRR projection
        if crr >= rrr:
            proj_text = f"On track to win in ~{round(overs + balls_to_win_at_crr/6, 1)} overs"
            proj_color = "#43e97b"
        else:
            deficit_per_over = (rrr - crr)
            proj_text = f"Need +{round(deficit_per_over,1)} RPO more to stay on track"
            proj_color = "#ff6b6b"
        st.markdown(f"""
        <div class="milestone-card">
            <div class="milestone-icon">🎯</div>
            <div>
                <div class="milestone-text">Scoring Projection</div>
                <div class="milestone-text"><b style="color:{proj_color};">{proj_text}</b></div>
                <div class="milestone-text" style="font-size:0.78rem;">{round(overs_remaining,1)} overs remaining</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # ─── NEW FEATURE 4: SCENARIO SIMULATOR ──────────
    # ═══════════════════════════════════════════════
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: var(--muted); margin-bottom: 1rem;">How does win probability change under different next-over outcomes?</div>', unsafe_allow_html=True)

    scenarios = [
        {"label": "Wicket maiden (0 runs, 1 wkt)",  "runs": 0,  "wkts": 1, "balls": 6},
        {"label": "Dot ball over (0 runs, 0 wkt)",  "runs": 0,  "wkts": 0, "balls": 6},
        {"label": "Quiet over (4 runs, 0 wkt)",     "runs": 4,  "wkts": 0, "balls": 6},
        {"label": "Average over (7 runs, 0 wkt)",   "runs": 7,  "wkts": 0, "balls": 6},
        {"label": "Good over (10 runs, 0 wkt)",     "runs": 10, "wkts": 0, "balls": 6},
        {"label": "Boundary over (12 runs, 0 wkt)", "runs": 12, "wkts": 0, "balls": 6},
        {"label": "Six-hitting over (18 runs, 0 wkt)", "runs": 18, "wkts": 0, "balls": 6},
        {"label": "Two-wicket over (6 runs, 2 wkt)","runs": 6,  "wkts": 2, "balls": 6},
    ]

    rows_html = ""
    for sc in scenarios:
        new_runs_left   = max(runs_left - sc["runs"], 0)
        new_balls_left  = max(balls_left - sc["balls"], 0)
        new_wkts_left   = max(wickets_left - sc["wkts"], 0)
        new_score       = score + sc["runs"]
        new_overs       = overs + sc["balls"] / 6
        new_crr         = new_score / new_overs if new_overs > 0 else 0
        new_rrr         = (new_runs_left * 6) / new_balls_left if new_balls_left > 0 else 0

        new_prob = get_win_prob(
            batting_team, bowling_team, city,
            new_runs_left, new_balls_left, new_wkts_left,
            target, new_crr, new_rrr
        )

        if new_prob is None:
            prob_str  = "—"
            delta_str = "—"
            pill_class = "pill-gold"
        else:
            new_pct   = round(new_prob * 100, 1)
            delta     = round((new_prob - win_prob) * 100, 1)
            prob_str  = f"{new_pct}%"
            sign      = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta}%"
            pill_class = "pill-green" if delta >= 0 else "pill-red"

        rows_html += f"""
        <tr>
            <td>{sc['label']}</td>
            <td>{sc['runs']} runs, {sc['wkts']} wkt(s)</td>
            <td>{prob_str}</td>
            <td><span class='pill {pill_class}'>{delta_str}</span></td>
        </tr>"""

    table_iframe_html = f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=Syne:wght@700&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #18181f; font-family: 'DM Sans', sans-serif; color: #e8e8f0; overflow: hidden; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
thead th {{
    background: #111118; color: #6b6b80; text-transform: uppercase;
    letter-spacing: 0.08em; font-size: 0.72rem; padding: 0.75rem 1rem;
    text-align: left; border-bottom: 1px solid #26263a; font-weight: 500;
}}
tbody td {{ padding: 0.7rem 1rem; border-bottom: 1px solid #26263a; color: #e8e8f0; vertical-align: middle; }}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover td {{ background: rgba(108,99,255,0.07); }}
.pill {{ display: inline-block; padding: 0.22rem 0.75rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }}
.pill-green {{ background: rgba(67,233,123,0.15); color: #43e97b; }}
.pill-red   {{ background: rgba(255,107,107,0.15); color: #ff6b6b; }}
.pill-gold  {{ background: rgba(255,209,102,0.12); color: #ffd166; }}
</style>
</head>
<body>
<table>
  <thead>
    <tr>
      <th>Scenario (Next Over)</th>
      <th>Outcome</th>
      <th>New Win %</th>
      <th>Change</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""
    st.iframe(table_iframe_html, height=310)

    # ═══════════════════════════════════════════════
    # ─── NEW FEATURE 5: WIN PROBABILITY TREND ───────
    # ═══════════════════════════════════════════════
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Win Probability Over Remaining Overs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: var(--muted); margin-bottom: 1rem;">Projected win probability if the batting side scores at exactly the required run rate from each future over.</div>', unsafe_allow_html=True)

    overs_int = int(overs)
    trend_overs = list(range(overs_int + 1, 21))
    trend_probs = []
    trend_labels = []

    current_score_sim = score
    current_overs_sim = overs

    for ov in trend_overs:
        balls_done = ov * 6
        sim_balls_left = max(120 - balls_done, 0)
        # How many balls elapsed since now
        elapsed_balls = balls_done - int(overs * 6)
        # Score at RRR from current state
        runs_scored_at_rrr = (elapsed_balls / 6) * rrr
        sim_score = score + runs_scored_at_rrr
        sim_runs_left = max(target - sim_score, 0)
        sim_crr = sim_score / ov if ov > 0 else 0
        sim_rrr = (sim_runs_left * 6) / sim_balls_left if sim_balls_left > 0 else 0

        p = get_win_prob(batting_team, bowling_team, city,
                         sim_runs_left, sim_balls_left, wickets_left,
                         target, sim_crr, sim_rrr)
        if p is not None:
            trend_probs.append(round(p * 100, 1))
            trend_labels.append(f"Ov {ov}")

    if trend_probs:
        import json
        labels_json = json.dumps(trend_labels)
        probs_json  = json.dumps(trend_probs)
        current_prob_pct = round(win_prob * 100, 1)

        chart_html = f"""
        <div class="sparkline-wrap">
            <canvas id="trendChart" height="80"></canvas>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <script>
        (function() {{
            var ctx = document.getElementById('trendChart').getContext('2d');
            var labels = {labels_json};
            var data   = {probs_json};
            var grad   = ctx.createLinearGradient(0, 0, 0, 200);
            grad.addColorStop(0, 'rgba(108,99,255,0.35)');
            grad.addColorStop(1, 'rgba(108,99,255,0)');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: '{batting_team} Win %',
                        data: data,
                        borderColor: '#6c63ff',
                        backgroundColor: grad,
                        borderWidth: 2.5,
                        pointBackgroundColor: '#6c63ff',
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        tension: 0.4,
                        fill: true,
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ labels: {{ color: '#9090a8', font: {{ family: 'DM Sans' }} }} }},
                        tooltip: {{
                            backgroundColor: '#18181f',
                            borderColor: '#26263a',
                            borderWidth: 1,
                            titleColor: '#e8e8f0',
                            bodyColor: '#9090a8',
                            callbacks: {{
                                label: function(ctx) {{ return ' Win Prob: ' + ctx.parsed.y + '%'; }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{ ticks: {{ color: '#6b6b80' }}, grid: {{ color: '#26263a' }} }},
                        y: {{
                            min: 0, max: 100,
                            ticks: {{ color: '#6b6b80', callback: function(v) {{ return v + '%'; }} }},
                            grid: {{ color: '#26263a' }}
                        }}
                    }}
                }}
            }});
        }})();
        </script>
        """
        st.iframe(chart_html, height=300)