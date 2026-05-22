"""
NutriScreen Cameroun — Dépistage IA de la Sous nutrition Infantile
Application Streamlit complète — EDS 2018 — 60 features
"""
import json, base64, numpy as np, pandas as pd, joblib, folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

pdf = FPDF()
pdf.add_page()

# Favicon inline (croix médicale SVG encodée base64) 
_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#0f6b8e"/>
  <rect x="26" y="10" width="12" height="44" rx="4" fill="#fff"/>
  <rect x="10" y="26" width="44" height="12" rx="4" fill="#fff"/>
</svg>"""
_favicon_b64 = base64.b64encode(_FAVICON_SVG.encode()).decode()

st.set_page_config(
    page_title="NutriScreen CM — Dépistage IA Sous nutrition Infantile",
    page_icon=f"data:image/svg+xml;base64,{_favicon_b64}",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<meta name="description" content="Dépistage IA de la sous nutrition infantile au Cameroun — EDS 2018 — Stacking Classifier.">
<meta name="author" content="NGOUMTSOP TEUZEM Yeiayel">
""", unsafe_allow_html=True)

# Références géographiques
REGION_COORDS = {
    1:  {"nom":"Adamaoua",     "lat":7.33,   "lon":13.58,  "color":"#e74c3c"},
    2:  {"nom":"Centre",       "lat":4.57,   "lon":11.52,  "color":"#3498db"},
    3:  {"nom":"Douala",       "lat":4.0483, "lon":9.7043, "color":"#9b59b6"},
    4:  {"nom":"Est",          "lat":4.25,   "lon":13.50,  "color":"#e67e22"},
    5:  {"nom":"Extrême-Nord", "lat":10.60,  "lon":14.32,  "color":"#1abc9c"},
    6:  {"nom":"Littoral",     "lat":4.60,   "lon":10.00,  "color":"#f39c12"},
    7:  {"nom":"Nord",         "lat":9.30,   "lon":13.39,  "color":"#2ecc71"},
    8:  {"nom":"Nord-Ouest",   "lat":6.07,   "lon":10.15,  "color":"#e84393"},
    9:  {"nom":"Ouest",        "lat":5.48,   "lon":10.42,  "color":"#00cec9"},
    10: {"nom":"Sud",          "lat":2.84,   "lon":10.92,  "color":"#6c5ce7"},
    11: {"nom":"Sud-Ouest",    "lat":5.00,   "lon":9.20,   "color":"#fd9644"},
    12: {"nom":"Yaoundé",      "lat":3.8667, "lon":11.5167,"color":"#d63031"},
}
LABELS_SEXE      = {1:"Masculin", 2:"Féminin"}
LABELS_MILIEU    = {1:"Urbain", 2:"Rural"}
LABELS_EDUCATION = {0:"Aucun", 1:"Primaire", 2:"Secondaire", 3:"Supérieur"}
LABELS_RICHESSE  = {1:"Le plus pauvre", 2:"Pauvre", 3:"Moyen", 4:"Riche", 5:"Le plus riche"}
LABELS_RELIGION  = {1:"Catholique", 2:"Protestant", 3:"Autre Chrétien",
                    4:"Musulman", 5:"Animiste/Autre", 6:"Sans religion"}
SOURCE_EAU_LABELS = {
    11.0:"Eau courante (domicile)", 12.0:"Eau courante (voisin)",
    13.0:"Fontaine publique", 14.0:"Robinet ouvert",
    21.0:"Puits à pompe / forage", 31.0:"Puits protégé",
    32.0:"Puits non protégé", 41.0:"Source protégée",
    42.0:"Source non protégée", 43.0:"Eau de pluie",
    51.0:"Eau en bouteille", 61.0:"Camion-citerne",
    62.0:"Charrette avec petite cuve", 71.0:"Eau de surface (rivière/lac)",
    92.0:"Autre", 96.0:"Non applicable", 97.0:"Ne sait pas",
}
TOILETTES_LABELS = {
    11.0:"Chasse d'eau — tout-à-l'égout", 12.0:"Chasse d'eau — fosse septique",
    13.0:"Chasse d'eau — latrine", 14.0:"Chasse d'eau — inconnu",
    15.0:"Chasse d'eau — rivière/lac", 21.0:"Latrine améliorée ventilée",
    22.0:"Latrine à dalle", 23.0:"Latrine à fosse",
    31.0:"Toilettes à compost", 41.0:"Seau",
    43.0:"Suspendue / rivière", 97.0:"Aucune / plein air",
}

# Assets
st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>',
    unsafe_allow_html=True,
)

# CSS 
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&display=swap');

/* ── Reset & base ── */
html, body, .stApp { font-family:'DM Sans',sans-serif; background:#f0f4f8; color:#1a1f2e; }
.stApp .block-container { max-width:1280px; padding:0 2.5rem 4rem; }
* { box-sizing:border-box; }

/* ── Hero ── */
.hero {
  background:linear-gradient(135deg,#0d1b2a 0%,#1b3a5c 50%,#0f6b8e 100%);
  border-radius:0 0 24px 24px; padding:56px 64px 60px;
  color:#fff; margin-bottom:36px;
  box-shadow:0 8px 40px rgba(13,27,42,.4);
  text-align:center; position:relative; overflow:hidden;
}
.hero::before {
  content:''; position:absolute; inset:0;
  background:radial-gradient(ellipse at 60% 50%,rgba(15,107,142,.35) 0%,transparent 70%);
  pointer-events:none;
}
.hero-inner { position:relative; z-index:1; max-width:860px; margin:0 auto; }
.hero-eyebrow {
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22);
  border-radius:100px; padding:6px 18px; font-size:.78rem; font-weight:700;
  letter-spacing:1.2px; text-transform:uppercase; color:rgba(255,255,255,.92);
  margin-bottom:20px;
}
.hero h1 {
  font-size:2.7rem; font-weight:800; margin:0 0 16px;
  line-height:1.15; letter-spacing:-.6px;
}
.hero h1 span { color:#67e8f9; }
.hero-desc {
  font-size:1.05rem; font-weight:400; opacity:.84;
  max-width:720px; margin:0 auto 32px; line-height:1.7;
}
.hero-stats { display:flex; gap:16px; justify-content:center; flex-wrap:wrap; }
.hero-stat {
  background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.18);
  border-radius:14px; padding:14px 24px; min-width:148px; text-align:center;
  backdrop-filter:blur(4px);
}
.hero-stat-val { font-size:1.6rem; font-weight:800; color:#fff; line-height:1; }
.hero-stat-lbl { font-size:.74rem; color:rgba(255,255,255,.68); margin-top:4px; letter-spacing:.3px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background:#fff; border-radius:16px; padding:5px 6px; gap:4px;
  border:1px solid #dde4ef; box-shadow:0 2px 8px rgba(0,0,0,.06);
  margin-bottom:28px;
}
.stTabs [data-baseweb="tab"] {
  border-radius:11px !important; font-weight:600 !important;
  font-size:.9rem !important; padding:10px 26px !important;
  color:#5a6a7e !important; transition:all .15s !important;
  letter-spacing:.1px !important;
}
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg,#1b3a5c,#0f6b8e) !important;
  color:#fff !important; box-shadow:0 3px 12px rgba(15,107,142,.32) !important;
}

/* ── Section headers ── */
.sec-hdr {
  display:flex; align-items:center; gap:12px;
  font-size:1.2rem; font-weight:700; color:#1b3a5c; margin:0 0 8px;
}
.sec-hdr i { color:#0f6b8e; font-size:1rem; }
.sec-sub { font-size:.9rem; color:#64748b; margin:0 0 22px; line-height:1.65; max-width:800px; }

/* ── Form blocks ── */
.form-block {
  background:#fff; border-radius:16px; padding:26px 28px;
  border:1px solid #e2e8f2; margin-bottom:16px;
  box-shadow:0 2px 8px rgba(0,0,0,.05);
}
.form-block-title {
  font-size:.8rem; font-weight:800; letter-spacing:1.8px;
  text-transform:uppercase; color:#0f6b8e;
  margin:0 0 20px; padding-bottom:12px;
  border-bottom:2px solid #e6f0fb;
  display:flex; align-items:center; gap:9px;
}
.form-block-title i { font-size:.9rem; }

/* ── Streamlit widget overrides ── */
/* Labels */
.stSelectbox > label,
.stRadio > label,
.stNumberInput > label,
.stSlider > label,
.stTextInput > label {
  font-size:.9rem !important; font-weight:700 !important;
  color:#1b3a5c !important; letter-spacing:.1px !important;
  margin-bottom:6px !important;
}
/* Select inputs */
div[data-baseweb="select"] > div {
  border-radius:12px !important;
  border:1.5px solid #c8d6e8 !important;
  background:#f8fbff !important;
  min-height:46px !important;
  font-size:.95rem !important;
  font-weight:500 !important;
  color:#1a1f2e !important;
  transition:border-color .15s !important;
}
div[data-baseweb="select"] > div:hover {
  border-color:#0f6b8e !important;
}
div[data-baseweb="select"] > div:focus-within {
  border-color:#0f6b8e !important;
  box-shadow:0 0 0 3px rgba(15,107,142,.12) !important;
}
/* Number inputs */
.stNumberInput > div > div > input {
  border-radius:12px !important;
  border:1.5px solid #c8d6e8 !important;
  background:#f8fbff !important;
  height:46px !important;
  font-size:.95rem !important;
  font-weight:500 !important;
  color:#1a1f2e !important;
  padding:0 14px !important;
}
.stNumberInput > div > div > input:focus {
  border-color:#0f6b8e !important;
  box-shadow:0 0 0 3px rgba(15,107,142,.12) !important;
  outline:none !important;
}
.stNumberInput [data-testid="stNumberInputField"] { background:#f8fbff !important; }
/* Radio */
.stRadio [role="radiogroup"] { gap:12px !important; margin-top:4px !important; }
.stRadio [data-testid="stMarkdownContainer"] { font-size:.95rem !important; }
/* Slider */
.stSlider [data-testid="stTickBar"] { font-size:.78rem !important; }
/* Select_slider */
div[data-testid="stSlider"] > div > div {
  color:#1b3a5c !important;
}

/* ── IMC box ── */
.imc-box {
  background:linear-gradient(135deg,#f0f7ff,#e8f2fd);
  border:1.5px solid #c7ddf8; border-radius:12px;
  padding:14px 16px; margin-top:6px;
}
.imc-box-label { font-size:.72rem; font-weight:800; color:#64748b;
  letter-spacing:1px; text-transform:uppercase; }
.imc-box-val { font-size:1.4rem; font-weight:800; color:#1b3a5c; margin:4px 0 6px; }
.imc-box-tag { font-size:.78rem; font-weight:700; border-radius:7px;
  padding:3px 10px; display:inline-block; }
.imc-normal { background:#dcfce7; color:#166534; }
.imc-low    { background:#fef9c3; color:#854d0e; }
.imc-high   { background:#fee2e2; color:#991b1b; }

/* ── Submit button ── */
.stButton > button {
  background:linear-gradient(135deg,#1b3a5c,#0f6b8e) !important;
  color:#fff !important; border:none !important; border-radius:14px !important;
  padding:15px 36px !important; font-weight:800 !important;
  font-size:1rem !important; letter-spacing:.3px !important;
  cursor:pointer !important;
  box-shadow:0 5px 20px rgba(15,107,142,.35) !important;
  transition:transform .15s, box-shadow .15s !important;
  width:100% !important;
}
.stButton > button:hover {
  transform:translateY(-2px) !important;
  box-shadow:0 10px 28px rgba(15,107,142,.45) !important;
}
/* Download buttons */
.stDownloadButton > button {
  background:#fff !important; color:#0f6b8e !important;
  border:2px solid #0f6b8e !important; border-radius:11px !important;
  padding:10px 24px !important; font-weight:700 !important;
  font-size:.88rem !important; cursor:pointer !important;
  transition:all .15s !important;
}
.stDownloadButton > button:hover {
  background:#0f6b8e !important; color:#fff !important;
}

/* ── Empty state (formulaire) ── */
.empty-state {
  background:#fff; border:2px dashed #c8d6e8; border-radius:18px;
  padding:64px 40px; text-align:center;
}
.empty-state i { font-size:2.8rem; color:#c8d6e8; display:block; margin-bottom:16px; }
.empty-state h3 { font-size:1.05rem; font-weight:700; color:#64748b; margin:0 0 8px; }
.empty-state p  { font-size:.88rem; color:#94a3b8; margin:0; line-height:1.6; }

/* ── Profile grid ── */
.profile-grid {
  display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:22px;
}
.profile-cell {
  background:#f8fafc; border:1px solid #e2e8f0;
  border-radius:11px; padding:11px 14px;
}
.profile-cell-lbl {
  font-size:.67rem; font-weight:800; color:#94a3b8;
  letter-spacing:.9px; text-transform:uppercase; margin-bottom:4px;
}
.profile-cell-val { font-size:.98rem; font-weight:700; color:#1a1f2e; }

/* ── Result panel ── */
.result-panel { border-radius:18px; overflow:hidden; box-shadow:0 6px 32px rgba(0,0,0,.11); margin:0 0 20px; }
.result-hdr { display:flex; align-items:center; gap:16px; padding:22px 28px; }
.result-hdr-risk { background:linear-gradient(135deg,#7f1d1d,#dc2626); }
.result-hdr-safe { background:linear-gradient(135deg,#052e16,#16a34a); }
.result-hdr i { font-size:1.8rem; color:rgba(255,255,255,.92); }
.result-hdr-text h2 { color:#fff; font-size:1.25rem; font-weight:800; margin:0 0 3px; }
.result-hdr-text p  { color:rgba(255,255,255,.76); font-size:.84rem; margin:0; }
.result-body { background:#fff; padding:24px 28px; }

/* ── Chips ── */
.chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:20px; }
.chip {
  display:inline-flex; align-items:center; gap:7px;
  padding:7px 16px; border-radius:100px; font-size:.82rem;
  font-weight:700; border:1.5px solid transparent;
}
.chip i { font-size:.74rem; }
.chip-risk    { background:#fef2f2; color:#b91c1c; border-color:#fecaca; }
.chip-safe    { background:#f0fdf4; color:#15803d; border-color:#bbf7d0; }

/* ── Prob bar ── */
.prob-bar { margin:14px 0 20px; }
.prob-bar-row { display:flex; justify-content:space-between; font-size:.82rem; color:#64748b; margin-bottom:7px; }
.prob-bar-row b { font-size:.96rem; color:#1a1f2e; font-weight:800; }
.prob-track { height:10px; background:#e9eef5; border-radius:100px; overflow:hidden; }
.prob-fill-risk { height:100%; border-radius:100px; background:linear-gradient(90deg,#f87171,#dc2626); }
.prob-fill-safe { height:100%; border-radius:100px; background:linear-gradient(90deg,#4ade80,#16a34a); }

/* ── Factors ── */
.factors { margin:4px 0 16px; }
.factor-grp-title {
  font-size:.72rem; font-weight:800; letter-spacing:1.5px; text-transform:uppercase;
  color:#64748b; margin:16px 0 9px;
}
.factor-item {
  display:flex; align-items:flex-start; gap:10px;
  padding:11px 15px; border-radius:11px; font-size:.88rem;
  line-height:1.55; margin-bottom:7px;
}
.factor-risk { background:#fef2f2; border-left:3px solid #dc2626; color:#7f1d1d; }
.factor-safe { background:#f0fdf4; border-left:3px solid #16a34a; color:#052e16; }
.factor-item i { margin-top:2px; flex-shrink:0; font-size:.88rem; }

/* ── Interventions ── */
.interventions {
  background:#fffbeb; border:1px solid #fde68a;
  border-radius:13px; padding:18px 20px; margin:16px 0;
}
.interventions h4 {
  margin:0 0 14px; font-size:.88rem; font-weight:800; color:#78350f;
  display:flex; align-items:center; gap:9px;
}
.intv-item {
  background:#fff; border-left:3px solid #f59e0b;
  border-radius:9px; padding:12px 15px; margin-bottom:10px;
}
.intv-item:last-child { margin-bottom:0; }
.intv-item h5 { margin:0 0 5px; font-size:.85rem; font-weight:800; color:#92400e;
  display:flex; align-items:center; gap:7px; }
.intv-item p  { margin:0; font-size:.84rem; color:#374151; line-height:1.58; }

/* ── Disclaimer ── */
.disclaimer {
  background:#f8fafc; border:1px solid #e2e8f0;
  border-radius:11px; padding:12px 16px; margin-top:18px;
  font-size:.8rem; color:#64748b; line-height:1.62;
  display:flex; gap:10px; align-items:flex-start;
}
.disclaimer i { flex-shrink:0; margin-top:2px; color:#94a3b8; font-size:.9rem; }

/* ── Map legend ── */
.map-legend {
  display:flex; gap:24px; flex-wrap:wrap; margin-bottom:18px;
  background:#fff; border:1px solid #e2e8f2; border-radius:12px;
  padding:12px 18px;
}
.legend-item {
  display:flex; align-items:center; gap:9px;
  font-size:.86rem; font-weight:600; color:#374151;
}
.legend-dot { width:13px; height:13px; border-radius:50%; flex-shrink:0; }

/* ── History empty ── */
.hist-empty {
  background:#fff; border:2px dashed #c8d6e8; border-radius:16px;
  padding:56px 40px; text-align:center;
}
.hist-empty i { font-size:2.2rem; color:#cbd5e1; display:block; margin-bottom:14px; }
.hist-empty p { color:#94a3b8; font-size:.92rem; margin:0; line-height:1.65; }
.stDataFrame { border-radius:13px !important; overflow:hidden !important; }

/* ── Export row ── */
.export-row { display:flex; gap:12px; margin-top:16px; flex-wrap:wrap; }

/* ── Footer ── */
.app-footer {
  text-align:center; margin-top:60px; padding:26px 0 14px;
  border-top:1px solid #dde4ef; color:#94a3b8; font-size:.82rem; line-height:1.9;
}
.app-footer a { color:#0f6b8e; text-decoration:none; font-weight:700; }
.app-footer a:hover { text-decoration:underline; }

/* ── Responsive ── */
@media(max-width:900px) {
  .hero { padding:36px 24px 40px; }
  .hero h1 { font-size:1.8rem; }
  .profile-grid { grid-template-columns:repeat(2,1fr); }
  .stApp .block-container { padding:0 1rem 2rem; }
}
</style>""", unsafe_allow_html=True)

# LocalStorage bridge (persistence inter-sessions)
LS_KEY = "nutriscreen_history_v2"

_LS_SCRIPT = f"""
<script>
(function() {{
  // Restore from localStorage → Streamlit query param trick via hidden input
  const key = "{LS_KEY}";
  const raw = localStorage.getItem(key);
  if (raw) {{
    // Inject into a hidden input that Python reads via URL hash
    const el = document.getElementById("ls-payload");
    if (el) el.value = raw;
  }}

  // Listen for save events dispatched by Streamlit custom component
  window.addEventListener("message", function(e) {{
    if (e.data && e.data.type === "nutriscreen_save") {{
      localStorage.setItem(key, e.data.payload);
    }}
    if (e.data && e.data.type === "nutriscreen_clear") {{
      localStorage.removeItem(key);
    }}
  }});
}})();
</script>
<input type="hidden" id="ls-payload" value="">
"""

def _ls_save(data: list):
    """Save list to localStorage via postMessage."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    payload_escaped = payload.replace("\\", "\\\\").replace("`", "\\`")
    script = f"""
    <script>
    (function() {{
      const payload = `{payload_escaped}`;
      localStorage.setItem("{LS_KEY}", payload);
    }})();
    </script>
    """
    components.html(script, height=0, scrolling=False)

def _ls_load_component():
    """Render hidden bridge to read localStorage."""
    bridge = f"""
    <!DOCTYPE html>
    <html><body>
    <script>
    (function() {{
      const key = "{LS_KEY}";
      const raw = localStorage.getItem(key);
      const out = document.getElementById("out");
      if (raw && out) {{ out.textContent = raw; }}
    }})();
    </script>
    <div id="out" style="display:none;"></div>
    </body></html>
    """
    return components.html(bridge, height=0, scrolling=False)

# Chargement des modèles
MODEL_DIR = Path("models")

@st.cache_resource(show_spinner=False)
def load_resources():
    sc = joblib.load(MODEL_DIR / "scaler.joblib")
    ms = joblib.load(MODEL_DIR / "model_stunting.joblib")
    mw = joblib.load(MODEL_DIR / "model_wasting.joblib")
    mu = joblib.load(MODEL_DIR / "model_underweight.joblib")
    mg = joblib.load(MODEL_DIR / "model_global.joblib")
    with open(MODEL_DIR / "feature_cols.json") as f:
        fc = json.load(f)
    return sc, ms, mw, mu, mg, fc

@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_excel("Dataset_Malnutrition_Cameroun_2018.xlsx", sheet_name="data")

try:
    scaler, model_stunting, model_wasting, model_underweight, model_global, FEATURE_COLS = load_resources()
    df_hist = load_dataset()
except Exception as e:
    st.error(f"Impossible de charger les ressources : {e}")
    st.stop()

# Initialisation session state
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []
if "_ls_loaded" not in st.session_state:
    st.session_state["_ls_loaded"] = False

# Restore depuis localStorage au premier chargement via query_params
if not st.session_state["_ls_loaded"]:
    ls_raw = st.query_params.get("ls", None)
    if ls_raw:
        try:
            restored = json.loads(ls_raw)
            if isinstance(restored, list) and restored:
                st.session_state["predictions"] = restored
        except Exception:
            pass
    st.session_state["_ls_loaded"] = True

# Helpers
def lbl_reg(c): return REGION_COORDS.get(int(c), {}).get("nom", str(c))
def lbl_sex(c): return LABELS_SEXE.get(int(c), str(c))
def lbl_mil(c): return LABELS_MILIEU.get(int(c), str(c))
def lbl_edu(c): return LABELS_EDUCATION.get(int(c), str(c))
def lbl_ric(c): return LABELS_RICHESSE.get(int(c), str(c))
def lbl_rel(c): return LABELS_RELIGION.get(int(c), str(c))

def age_txt(m):
    a, r = int(m)//12, int(m)%12
    if a == 0: return f"{r} mois"
    if r == 0: return f"{a} an{'s' if a>1 else ''}"
    return f"{a} an{'s' if a>1 else ''} et {r} mois"

def build_X(d):
    QUAL = ["region","milieu_residence","education_mere","source_eau",
            "type_toilettes","index_richesse","religion","sexe_enfant"]
    row = pd.DataFrame([d])
    enc = pd.get_dummies(row, columns=QUAL, drop_first=True)
    for c in FEATURE_COLS:
        if c not in enc.columns:
            enc[c] = 0
    return scaler.transform(enc[FEATURE_COLS].astype(float))

def predict_all(X):
    out = {}
    for k, m in [("stunting",model_stunting),("wasting",model_wasting),
                  ("underweight",model_underweight),("global",model_global)]:
        out[k] = {"pred":int(m.predict(X)[0]),
                  "prob":float(m.predict_proba(X)[0][1])*100}
    return out

def add_jitter(lat, lon, mag=0.85):
    return lat+np.random.uniform(-mag,mag), lon+np.random.uniform(-mag,mag)

def build_popup(row, is_new=False):
    rc    = int(row.get("region", 0))
    malnu = int(row.get("Y_global_malnutrition", row.get("Y_global_malnutrition_PRED", 0)))
    sc_   = "#dc2626" if malnu == 1 else "#16a34a"
    stat  = "A risque de sous-nutrition" if malnu == 1 else "Etat nutritionnel normal"
    badge = "NOUVELLE EVALUATION" if is_new else "EDS 2018"
    icon  = "circle-exclamation" if malnu == 1 else "circle-check"
    return f"""<div style="font-family:Inter,sans-serif;width:288px;font-size:13px;line-height:1.5;">
      <div style="background:{sc_};color:#fff;padding:9px 13px;border-radius:8px 8px 0 0;font-weight:700;text-align:center;">
        <i class="fas fa-{icon}"></i> {stat}</div>
      <div style="padding:12px 14px;background:#fafafa;border-radius:0 0 8px 8px;">
        <span style="background:#e8edf4;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;color:#374151;">{badge}</span>
        <table style="width:100%;border-collapse:collapse;margin-top:9px;">
          <tr><td style="padding:3px 0;color:#64748b;"><i class="fas fa-baby" style="color:#3498db;width:18px;"></i> Age enfant</td><td style="text-align:right;font-weight:600;">{age_txt(row.get("age_enfant_mois",0))}</td></tr>
          <tr><td style="padding:3px 0;color:#64748b;"><i class="fas fa-venus-mars" style="color:#9b59b6;width:18px;"></i> Sexe</td><td style="text-align:right;font-weight:600;">{lbl_sex(row.get("sexe_enfant",0))}</td></tr>
          <tr><td style="padding:3px 0;color:#64748b;"><i class="fas fa-location-dot" style="color:#e74c3c;width:18px;"></i> Region</td><td style="text-align:right;font-weight:600;">{lbl_reg(rc)}</td></tr>
          <tr><td style="padding:3px 0;color:#64748b;"><i class="fas fa-house" style="color:#f39c12;width:18px;"></i> Milieu</td><td style="text-align:right;font-weight:600;">{lbl_mil(row.get("milieu_residence",0))}</td></tr>
          <tr><td style="padding:3px 0;color:#64748b;"><i class="fas fa-user-nurse" style="color:#e84393;width:18px;"></i> Age mere</td><td style="text-align:right;font-weight:600;">{int(row.get("age_mere",0))} ans</td></tr>
          <tr><td style="padding:3px 0;color:#64748b;"><i class="fas fa-calculator" style="color:#00cec9;width:18px;"></i> IMC mere</td><td style="text-align:right;font-weight:600;">{float(row.get("imc_mere",0)):.1f} kg/m2</td></tr>
          <tr><td style="padding:3px 0;color:#64748b;"><i class="fas fa-graduation-cap" style="color:#2e86c1;width:18px;"></i> Education</td><td style="text-align:right;font-weight:600;">{lbl_edu(row.get("education_mere",0))}</td></tr>
          <tr><td style="padding:3px 0;color:#64748b;"><i class="fas fa-coins" style="color:#f1c40f;width:18px;"></i> Richesse</td><td style="text-align:right;font-weight:600;">{lbl_ric(row.get("index_richesse",0))}</td></tr>
        </table>
      </div></div>"""

def render_result(results, d):
    gl = results["global"]
    prob, is_risk = gl["prob"], gl["pred"] == 1
    hdr_cls = "result-hdr-risk" if is_risk else "result-hdr-safe"
    icon    = "triangle-exclamation" if is_risk else "circle-check"
    titre   = "Risque de sous nutrition detecte" if is_risk else "Etat nutritionnel satisfaisant"
    sous    = f"Evaluation globale — {datetime.now().strftime('%d/%m/%Y a %H:%M')}"

    chips = "".join(
        f'<span class="chip {"chip-risk" if results[k]["pred"]==1 else "chip-safe"}">'
        f'<i class="fas fa-{"circle-xmark" if results[k]["pred"]==1 else "circle-check"}"></i>'
        f'{full} — {results[k]["prob"]:.0f}%</span>'
        for k, full in [("stunting","Retard de croissance"),
                        ("wasting","Amaigrissement"),
                        ("underweight","Insuffisance ponderale")]
    )

    fill_cls = "prob-fill-risk" if is_risk else "prob-fill-safe"
    bar = (f'<div class="prob-bar">'
           f'<div class="prob-bar-row"><span>Probabilite de risque global</span><b>{prob:.1f}%</b></div>'
           f'<div class="prob-track"><div class="{fill_cls}" style="width:{min(prob,100):.1f}%"></div></div>'
           f'</div>')

    prot, risq = [], []
    edu=d["education_mere"]; ric=d["index_richesse"]; mil=d["milieu_residence"]
    age=d["age_enfant_mois"]; imc=d["imc_mere"]; itv=d["intervalle_naissance"]; reg=d["region"]
    if edu>=2: prot.append(f"Education maternelle {lbl_edu(edu).lower()} — facteur protecteur majeur.")
    else: risq.append(f"Education maternelle {lbl_edu(edu).lower()} — facteur de risque nutritionnel significatif.")
    if ric>=4: prot.append(f"Menage {lbl_ric(ric).lower()} — acces a une alimentation diversifiee.")
    elif ric<=2: risq.append(f"Menage {lbl_ric(ric).lower()} — acces limite a une alimentation equilibree.")
    if mil==2: risq.append("Residence en milieu rural — acces reduit aux soins et a l'alimentation.")
    else: prot.append("Residence en milieu urbain — meilleur acces aux soins et a l'alimentation.")
    if 6<=age<=23: risq.append(f"Enfant de {age_txt(age)} — periode critique de diversification alimentaire.")
    elif age>23: prot.append(f"Enfant de {age_txt(age)} — periode post-sevrage moins vulnerable.")
    if imc<18.5: risq.append(f"IMC maternel {imc:.1f} — insuffisance ponderale maternelle probable.")
    elif 18.5<=imc<=25: prot.append(f"IMC maternel {imc:.1f} — corpulence normale, bon etat nutritionnel maternel.")
    if 0<itv<24: risq.append(f"Intervalle inter-naissance de {itv} mois — espacement court, risque accru.")
    if reg in [5,7,1,4]: risq.append(f"Region {lbl_reg(reg)} — prevalence de sous nutrition parmi les plus elevees au Cameroun.")
    else: prot.append(f"Region {lbl_reg(reg)} — prevalence nutritionnelle moderee.")

    fh = '<div class="factors">'
    if prot:
        fh += '<div class="factor-grp-title">Facteurs protecteurs</div>'
        fh += "".join(f'<div class="factor-item factor-safe"><i class="fas fa-shield-halved"></i><span>{t}</span></div>' for t in prot)
    if risq:
        fh += '<div class="factor-grp-title">Facteurs de risque identifies</div>'
        fh += "".join(f'<div class="factor-item factor-risk"><i class="fas fa-circle-exclamation"></i><span>{t}</span></div>' for t in risq)
    fh += '</div>'

    intv = ""
    if is_risk:
        items = ""
        if results["stunting"]["pred"]==1:
            items += f'<div class="intv-item"><h5><i class="fas fa-bone"></i>Retard de croissance (Stunting) — {results["stunting"]["prob"]:.0f}%</h5><p>Malnutrition chronique. Diversification alimentaire riche en proteines, zinc, vitamines A et D. Supplementation et suivi pediatrique regulier. Processus a long terme.</p></div>'
        if results["wasting"]["pred"]==1:
            items += f'<div class="intv-item"><h5><i class="fas fa-bolt"></i>Amaigrissement aigu (Wasting) — {results["wasting"]["prob"]:.0f}%</h5><p><strong>Urgence medicale.</strong> Aliments Therapeutiques Prets a l\'Emploi (ATPE/Plumpy\'Nut), rehydratation si necessaire. Consultation medicale immediate pour ecarter toute infection sous-jacente.</p></div>'
        if results["underweight"]["pred"]==1:
            items += f'<div class="intv-item"><h5><i class="fas fa-scale-unbalanced"></i>Insuffisance ponderale — {results["underweight"]["prob"]:.0f}%</h5><p>Augmentation de l\'apport calorique quotidien, ajout de corps gras sains, repas plus frequents et surveillance stricte de la courbe de poids.</p></div>'
        if items:
            intv = f'<div class="interventions"><h4><i class="fas fa-kit-medical"></i> Interventions recommandees</h4>{items}</div>'

    disc = ('<div class="disclaimer"><i class="fas fa-circle-info"></i>'
            '<span>Ce resultat est une estimation statistique produite par un modele IA entraine sur les donnees '
            'EDS Cameroun 2018. Il ne constitue pas un diagnostic medical et ne remplace pas l\'evaluation '
            'd\'un professionnel de sante qualifie.</span></div>')

    return (f'<div class="result-panel">'
            f'<div class="result-hdr {hdr_cls}"><i class="fas fa-{icon}"></i>'
            f'<div class="result-hdr-text"><h2>{titre}</h2><p>{sous}</p></div></div>'
            f'<div class="result-body"><div class="chips">{chips}</div>{bar}{fh}{intv}{disc}</div></div>')

def generate_pdf_result(info, results):
    """PDF du résultat d'une prédiction."""
    pdf = FPDF(); pdf.add_page()
    pdf.set_fill_color(27, 58, 92)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",14)
    pdf.set_xy(0,8); pdf.cell(210,10,"NutriScreen Cameroun Rapport de depistage",align="C",ln=True)
    pdf.set_font("Helvetica","",9)
    pdf.cell(210,6,f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}",align="C",ln=True)
    pdf.set_text_color(26,31,46); pdf.ln(8)
    gl = results["global"]
    is_risk = gl["pred"]==1
    if is_risk:
        pdf.set_fill_color(220,38,38)
    else:
        pdf.set_fill_color(22,163,74)
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",12)
    label = "RISQUE DE SOUS NUTRITION DETECTE" if is_risk else "ETAT NUTRITIONNEL SATISFAISANT"
    pdf.cell(0,12,f"{label}  ({gl['prob']:.1f}%)",fill=True,ln=True,align="C")
    pdf.set_text_color(26,31,46); pdf.ln(6)
    pdf.set_font("Helvetica","B",11); pdf.cell(0,8,"Sous-types evalues",ln=True)
    pdf.set_font("Helvetica","",10)
    for k,l in [("stunting","Retard de croissance"),("wasting","Amaigrissement"),("underweight","Insuffisance ponderale")]:
        r=results[k]
        status="OUI" if r["pred"]==1 else "NON"
        pdf.cell(0,7,f"  {l} : {status}  ({r['prob']:.1f}%)",ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica","B",11); pdf.cell(0,8,"Profil de l'enfant evalue",ln=True)
    pdf.set_font("Helvetica","",10)
    rows = [
        ("Region", lbl_reg(info["region"])),
        ("Sexe", lbl_sex(info["sexe_enfant"])),
        ("Age enfant", age_txt(info["age_enfant_mois"])),
        ("Rang de naissance", str(int(info["rang_naissance"]))),
        ("Intervalle inter-naissance", f"{int(info['intervalle_naissance'])} mois"),
        ("Age mere", f"{info['age_mere']} ans"),
        ("Taille mere", f"{info['taille_mere']:.1f} cm"),
        ("Poids mere", f"{info['poids_mere']:.1f} kg"),
        ("IMC mere", f"{info['imc_mere']:.1f} kg/m2"),
        ("Education mere", lbl_edu(info["education_mere"])),
        ("Niveau de richesse", lbl_ric(info["index_richesse"])),
        ("Religion", lbl_rel(info["religion"])),
        ("Milieu", lbl_mil(info["milieu_residence"])),
        ("Source d'eau", SOURCE_EAU_LABELS.get(info["source_eau"],"N/A")),
        ("Type de toilettes", TOILETTES_LABELS.get(info["type_toilettes"],"N/A")),
    ]
    for i,(lbl_,val) in enumerate(rows):
        fill = i%2==0
        if fill: pdf.set_fill_color(248,250,252)
        else: pdf.set_fill_color(255,255,255)
        pdf.cell(90,7,f"  {lbl_}",fill=True)
        pdf.cell(0,7,f"  {val}",fill=True,ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica","I",8)
    pdf.set_text_color(100,116,139)
    pdf.multi_cell(0,5,"Avertissement : Ce rapport est genere par un modele IA entraine sur les donnees EDS Cameroun 2018. "
                       "Il ne constitue pas un diagnostic medical et ne remplace pas l'evaluation d'un professionnel de sante qualifie.")
    return bytes(pdf.output())

def generate_pdf_history(df_disp):
    """PDF de l'historique complet."""
    pdf = FPDF(orientation="L"); pdf.add_page()
    pdf.set_fill_color(255, 0, 0)
    pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",13)
    pdf.set_xy(0,6); pdf.cell(297,10,"NutriScreen Cameroun Historique des evaluations",align="C",ln=True)
    pdf.set_text_color(0, 0, 0); pdf.ln(4)
    cols = list(df_disp.columns)
    col_w = 280//len(cols)
    pdf.set_font("Helvetica","B",7); pdf.set_fill_color(240,244,248)
    for c in cols:
        pdf.cell(col_w,8,str(c)[:16],border=1,fill=True)
    pdf.ln()
    pdf.set_font("Helvetica","",7)
    for i,row in df_disp.iterrows():
        fill = i%2==0
        pdf.set_fill_color(248,250,252)
        for c in cols:
            pdf.cell(col_w,7,str(row[c])[:18],border=1,fill=fill)
        pdf.ln()
    return bytes(pdf.output())

def save_to_dataset(row_dict, results):
    path = Path("Dataset_Malnutrition_Cameroun_2018.xlsx")
    try:
        df_ex = pd.read_excel(path, sheet_name="data")
        new_row = {**row_dict,
                   "Y1_retard_croissance":      results["stunting"]["pred"],
                   "Y2_amaigrissement":         results["wasting"]["pred"],
                   "Y3_insuffisance_ponderale": results["underweight"]["pred"],
                   "Y_global_malnutrition":     results["global"]["pred"]}
        cols = [c for c in df_ex.columns if c in new_row]
        df_up = pd.concat([df_ex, pd.DataFrame([{c:new_row[c] for c in cols}])], ignore_index=True)
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df_up.to_excel(w, sheet_name="data", index=False)
    except Exception:
        pass

# HERO
n_total  = len(df_hist)
n_risque = int(df_hist["Y_global_malnutrition"].sum()) if "Y_global_malnutrition" in df_hist.columns else 0
prev_pct = round(n_risque/n_total*100,1) if n_total else 0

st.markdown(f"""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">
      <i class="fas fa-stethoscope"></i>&nbsp; Sante Infantile &bull; Intelligence Artificielle &bull; Cameroun
    </div>
    <h1>Depistage de la <span>Sous nutrition Infantile</span><br>au Cameroun</h1>
    <p class="hero-desc">
      Systeme d'aide au depistage precoce base sur un modele de <strong>Stacking Classifier</strong>
      (Random Forest + Gradient Boosting &rarr; Regression Logistique) entraine sur les
      <strong>{n_total:,} enfants</strong> de l'Enquete Demographique et de Sante
      (EDS-V Cameroun 2018). Evaluation simultanee du retard de croissance (stunting),
      de l'amaigrissement (wasting) et de l'insuffisance ponderale (underweight),
      avec recommandations d'intervention personnalisees.
    </p>
    <div class="hero-stats">
      <div class="hero-stat">
        <div class="hero-stat-val">{n_total:,}</div>
        <div class="hero-stat-lbl">Enfants EDS 2018</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-val">{prev_pct}%</div>
        <div class="hero-stat-lbl">Prevalence sous nutrition</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-val">60</div>
        <div class="hero-stat-lbl">Features utilisees</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-val">AUC&nbsp;&gt;&nbsp;0.90</div>
        <div class="hero-stat-lbl">Performance modeles</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-val">4</div>
        <div class="hero-stat-lbl">Modeles Stacking</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# LocalStorage : sauvegarde JS 
if st.session_state["predictions"]:
    _ls_save(st.session_state["predictions"])


# TABS
tab1, tab2, tab3 = st.tabs([
    "  Evaluation  ",
    "  Carte du Cameroun  ",
    "  Historique des evaluations  ",
])


# TAB 1 — EVALUATION DU PROFILE ET PREDICTION
with tab1:
    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        # ── Bloc Enfant
        st.markdown(
            '<div class="form-block">'
            '<div class="form-block-title"><i class="fas fa-baby"></i>Informations sur l\'enfant</div>',
            unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sexe       = st.selectbox("Sexe de l'enfant", list(LABELS_SEXE.keys()),
                                       format_func=lambda x: LABELS_SEXE[x])
            age_enfant = st.number_input("Age (mois)", 0, 59, 12, step=1)
        with c2:
            rang       = st.number_input("Rang de naissance", 1, 15, 1, step=1)
            intervalle = st.number_input("Intervalle inter-naissance (mois)", 0, 183, 24, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Bloc Mère
        st.markdown(
            '<div class="form-block">'
            '<div class="form-block-title"><i class="fas fa-user-nurse"></i>Informations sur la mere</div>',
            unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age_mere    = st.number_input("Age de la mere (annees)", 15, 49, 28, step=1)
            taille_mere = st.number_input("Taille de la mere (cm)", 108.0, 200.0, 161.0, step=0.5)
        with c2:
            poids_mere = st.number_input("Poids de la mere (kg)", 34.0, 132.0, 63.0, step=0.5)
            imc_mere   = poids_mere / ((taille_mere / 100) ** 2)
            if imc_mere < 18.5:
                tag = '<span class="imc-box-tag imc-low"><i class="fas fa-arrow-down"></i> Maigreur</span>'
            elif imc_mere <= 25:
                tag = '<span class="imc-box-tag imc-normal"><i class="fas fa-check"></i> Normal</span>'
            elif imc_mere <= 30:
                tag = '<span class="imc-box-tag imc-high"><i class="fas fa-arrow-up"></i> Surpoids</span>'
            else:
                tag = '<span class="imc-box-tag imc-high"><i class="fas fa-triangle-exclamation"></i> Obesite</span>'
            st.markdown(
                f'<div class="imc-box">'
                f'<div class="imc-box-label"><i class="fas fa-calculator"></i> IMC calcule automatiquement</div>'
                f'<div class="imc-box-val">{imc_mere:.2f} kg/m&sup2;</div>{tag}</div>',
                unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            education = st.selectbox("Niveau d'education", list(LABELS_EDUCATION.keys()),
                                      format_func=lambda x: LABELS_EDUCATION[x])
        with c2:
            religion  = st.selectbox("Religion", list(LABELS_RELIGION.keys()),
                                      format_func=lambda x: LABELS_RELIGION[x])
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Bloc Ménage
        st.markdown(
            '<div class="form-block">'
            '<div class="form-block-title"><i class="fas fa-house-chimney"></i>Menage &amp; localisation</div>',
            unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            region = st.selectbox("Region d'origine", list(REGION_COORDS.keys()),
                                   format_func=lambda x: REGION_COORDS[x]["nom"])
            milieu = st.radio("Milieu de residence", [1,2],
                               format_func=lambda x: LABELS_MILIEU[x], horizontal=True)
        with c2:
            richesse   = st.select_slider("Niveau de richesse du menage",
                                           options=[1,2,3,4,5],
                                           format_func=lambda x: LABELS_RICHESSE[x])
            source_eau = st.selectbox("Source d'eau principale",
                                       sorted(SOURCE_EAU_LABELS.keys()),
                                       format_func=lambda x: SOURCE_EAU_LABELS[x])
        type_toilettes = st.selectbox("Type de toilettes / assainissement",
                                       sorted(TOILETTES_LABELS.keys()),
                                       format_func=lambda x: TOILETTES_LABELS[x])
        st.markdown('</div>', unsafe_allow_html=True)

        clicked = st.button("Lancer l'evaluation", use_container_width=True)

    # ── Colonne résultats
    with col_res:
        if not clicked and not st.session_state["predictions"]:
            st.markdown("""
            <div class="empty-state">
              <i class="fas fa-stethoscope"></i>
              <h3>Aucune evaluation en cours</h3>
              <p>Renseignez le formulaire a gauche<br>et lancez l'evaluation pour obtenir<br>une analyse nutritionnelle complete.</p>
            </div>""", unsafe_allow_html=True)

        if clicked:
            input_dict = {
                "age_mere":age_mere, "poids_mere":poids_mere, "taille_mere":taille_mere,
                "imc_mere":imc_mere, "age_enfant_mois":age_enfant, "rang_naissance":rang,
                "intervalle_naissance":intervalle, "region":region,
                "milieu_residence":milieu, "education_mere":education,
                "source_eau":source_eau, "type_toilettes":type_toilettes,
                "index_richesse":richesse, "religion":religion, "sexe_enfant":sexe,
            }
            try:
                X_sc    = build_X(input_dict)
                results = predict_all(X_sc)

                # Profil
                pm = [("Region",lbl_reg(region)),("Age enfant",age_txt(age_enfant)),
                      ("Sexe",lbl_sex(sexe)),("Milieu",lbl_mil(milieu)),
                      ("Age mere",f"{age_mere} ans"),("IMC mere",f"{imc_mere:.1f}"),
                      ("Education",lbl_edu(education)),("Richesse",lbl_ric(richesse))]
                cells = "".join(
                    f'<div class="profile-cell"><div class="profile-cell-lbl">{l}</div>'
                    f'<div class="profile-cell-val">{v}</div></div>' for l,v in pm)
                st.markdown(f'<div class="profile-grid">{cells}</div>', unsafe_allow_html=True)

                st.markdown(render_result(results, input_dict), unsafe_allow_html=True)

                # Sauvegarde
                record = {**input_dict,
                          "Y_global_malnutrition_PRED": results["global"]["pred"],
                          "probabilite": results["global"]["prob"],
                          "prob_stunting":    results["stunting"]["prob"],
                          "prob_wasting":     results["wasting"]["prob"],
                          "prob_underweight": results["underweight"]["prob"],
                          "Y1_retard_croissance":      results["stunting"]["pred"],
                          "Y2_amaigrissement":         results["wasting"]["pred"],
                          "Y3_insuffisance_ponderale": results["underweight"]["pred"],
                          "date": datetime.now().strftime("%d/%m/%Y %H:%M")}
                st.session_state["predictions"].append(record)
                _ls_save(st.session_state["predictions"])
                save_to_dataset(input_dict, results)

            except Exception as e:
                st.error(f"Erreur lors de l'evaluation : {e}")

        elif st.session_state["predictions"]:
            last = st.session_state["predictions"][-1]
            res_last = {
                "global":      {"pred":last["Y_global_malnutrition_PRED"], "prob":last["probabilite"]},
                "stunting":    {"pred":last["Y1_retard_croissance"],       "prob":last.get("prob_stunting",0)},
                "wasting":     {"pred":last["Y2_amaigrissement"],          "prob":last.get("prob_wasting",0)},
                "underweight": {"pred":last["Y3_insuffisance_ponderale"],  "prob":last.get("prob_underweight",0)},
            }
            st.info("Dernier resultat — Relancez une evaluation pour mettre a jour.")
            st.markdown(render_result(res_last, last), unsafe_allow_html=True)

# TAB 2 — CARTE
with tab2:
    st.markdown(
        '<div class="sec-hdr"><i class="fas fa-map-location-dot"></i>'
        ' Repartition geographique — EDS 2018 &amp; evaluations de la session</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Visualisation des enfants de l\'enquete EDS 2018 sur le territoire '
        'camerounais. Les points verts indiquent un etat nutritionnel normal, les rouges un risque de sous nutrition. '
        'Cliquez sur un marqueur pour le profil complet. Les evaluations de la session courante '
        'apparaissent en etoile.</div>', unsafe_allow_html=True)

    st.markdown("""<div class="map-legend">
      <div class="legend-item"><div class="legend-dot" style="background:#16a34a;"></div>Etat normal (EDS 2018)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#dc2626;"></div>A risque (EDS 2018)</div>
      <div class="legend-item"><i class="fas fa-star" style="color:#f59e0b;"></i>&nbsp;Nouvelle evaluation (session)</div>
    </div>""", unsafe_allow_html=True)

    # Carte rapide : tuile legere, pas de filtre, rendu direct
    m = folium.Map(
        location=[5.5, 12.3], zoom_start=6,
        tiles="CartoDB positron",
        control_scale=False,
        prefer_canvas=True,       # canvas renderer = bien plus rapide
    )

    # Regions : cercles + labels
    for rid, ri in REGION_COORDS.items():
        folium.Circle(
            location=[ri["lat"], ri["lon"]], radius=55000,
            color=ri["color"], fill=True, fill_color=ri["color"],
            fill_opacity=0.13, weight=1.5,
            tooltip=ri["nom"]
        ).add_to(m)
        folium.Marker(
            location=[ri["lat"]+0.35, ri["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10.5px;font-weight:800;color:{ri["color"]};'
                     f'text-shadow:0 1px 3px rgba(255,255,255,.9);white-space:nowrap;'
                     f'pointer-events:none;">{ri["nom"]}</div>',
                icon_size=(120,20), icon_anchor=(60,10))
        ).add_to(m)

    # Clusters séparés pour les performances
    cluster_normal = MarkerCluster(
        name="Etat normal (EDS 2018)",
        options={"maxClusterRadius":40, "disableClusteringAtZoom":9}
    ).add_to(m)
    cluster_risque = MarkerCluster(
        name="A risque (EDS 2018)",
        options={"maxClusterRadius":40, "disableClusteringAtZoom":9}
    ).add_to(m)

    np.random.seed(42)
    # Sous-échantillonnage pour vitesse (max 1500 points par classe)
    df_norm = df_hist[df_hist["Y_global_malnutrition"]==0].sample(min(1500,len(df_hist[df_hist["Y_global_malnutrition"]==0])), random_state=42)
    df_risk = df_hist[df_hist["Y_global_malnutrition"]==1].sample(min(1500,len(df_hist[df_hist["Y_global_malnutrition"]==1])), random_state=42)

    for _, row in df_norm.iterrows():
        rid = row.get("region",0)
        if rid not in REGION_COORDS: continue
        base = REGION_COORDS[rid]
        jlat, jlon = add_jitter(base["lat"], base["lon"])
        folium.CircleMarker(
            location=[jlat,jlon], radius=5, color="#16a34a",
            fill=True, fill_color="#16a34a", fill_opacity=0.75, weight=0,
            popup=folium.Popup(build_popup(row,False), max_width=310)
        ).add_to(cluster_normal)

    for _, row in df_risk.iterrows():
        rid = row.get("region",0)
        if rid not in REGION_COORDS: continue
        base = REGION_COORDS[rid]
        jlat, jlon = add_jitter(base["lat"], base["lon"])
        folium.CircleMarker(
            location=[jlat,jlon], radius=5, color="#dc2626",
            fill=True, fill_color="#dc2626", fill_opacity=0.75, weight=0,
            popup=folium.Popup(build_popup(row,False), max_width=310)
        ).add_to(cluster_risque)

    # Nouvelles évaluations de session
    for rec in st.session_state["predictions"]:
        rid = rec.get("region",0)
        if rid not in REGION_COORDS: continue
        base = REGION_COORDS[rid]
        jlat, jlon = add_jitter(base["lat"], base["lon"], mag=0.25)
        is_r = int(rec.get("Y_global_malnutrition_PRED",0))==1
        folium.Marker(
            location=[jlat,jlon],
            icon=folium.Icon(color="red" if is_r else "green", icon="star", prefix="fa"),
            popup=folium.Popup(build_popup(rec,True), max_width=310)
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=660, use_container_width=True, returned_objects=[])


# TAB 3 — HISTORIQUE
with tab3:
    st.markdown(
        '<div class="sec-hdr"><i class="fas fa-clock-rotate-left"></i>'
        ' Historique des evaluations de la session</div>',
        unsafe_allow_html=True)

    n_sess = len(st.session_state["predictions"])
    if n_sess == 0:
        st.markdown("""<div class="hist-empty">
          <i class="fas fa-folder-open"></i>
          <p>Aucune evaluation realisee dans cette session.<br>
          Rendez-vous dans l'onglet <strong>Evaluation</strong> pour commencer.<br>
          <small>Les evaluations sont conservees localement et restaurees apres rafraichissement.</small></p>
        </div>""", unsafe_allow_html=True)
    else:
        df_p = pd.DataFrame(st.session_state["predictions"])
        disp = pd.DataFrame({
            "Date":              df_p.get("date",""),
            "Region":            df_p["region"].apply(lbl_reg),
            "Sexe":              df_p["sexe_enfant"].apply(lbl_sex),
            "Age enfant (mois)": df_p["age_enfant_mois"],
            "Age mere (ans)":    df_p["age_mere"],
            "Taille mere (cm)":  df_p["taille_mere"].round(1),
            "Poids mere (kg)":   df_p["poids_mere"].round(1),
            "IMC mere":          df_p["imc_mere"].round(1),
            "Religion":          df_p["religion"].apply(lbl_rel),
            "Education":         df_p["education_mere"].apply(lbl_edu),
            "Richesse":          df_p["index_richesse"].apply(lbl_ric),
            "Milieu":            df_p["milieu_residence"].apply(lbl_mil),
            "Stunting":          df_p["Y1_retard_croissance"].apply(lambda x: "Oui" if x==1 else "Non"),
            "Wasting":           df_p["Y2_amaigrissement"].apply(lambda x: "Oui" if x==1 else "Non"),
            "Underweight":       df_p["Y3_insuffisance_ponderale"].apply(lambda x: "Oui" if x==1 else "Non"),
            "Resultat":          df_p["Y_global_malnutrition_PRED"].apply(lambda x: "A risque" if x==1 else "Normal"),
            "Probabilite (%)":   df_p["probabilite"].round(1),
        })

        st.dataframe(disp, use_container_width=True, height=380)

        # Exports côte à côte
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            buf_xl = BytesIO()
            disp.to_excel(buf_xl, index=False, sheet_name="historique")
            st.download_button(
                label="Exporter Excel",
                data=buf_xl.getvalue(),
                file_name=f"nutriscreen_historique_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c2:
            try:
                pdf_hist = generate_pdf_history(disp)
                st.download_button(
                    label="Exporter PDF",
                    data=pdf_hist,
                    file_name=f"nutriscreen_historique_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
            except Exception:
                pass
        with c3:
            if st.button("Effacer l'historique"):
                st.session_state["predictions"] = []
                components.html(
                    f'<script>localStorage.removeItem("{LS_KEY}");</script>',
                    height=0)
                st.rerun()


# FOOTER
st.markdown("""
<div class="app-footer">
  <i class="fas fa-code"></i>
  Developpe par <a href="https://github.com/teuzem" target="_blank">NGOUMTSOP TEUZEM Yeiayel</a>
  &nbsp;&mdash;&nbsp; NutriScreen Cameroun &nbsp;&mdash;&nbsp; EDS-V 2018
  <br>
  Modele Stacking Classifier (RF + GBM &rarr; LogReg) &bull; 60 features &bull; AUC &gt; 0.90
  &bull; Donnees : Enquete Demographique et de Sante du Cameroun 2018
</div>
""", unsafe_allow_html=True)
