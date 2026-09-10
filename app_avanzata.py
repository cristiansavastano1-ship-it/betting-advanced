import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import time
from datetime import date
from scipy.stats import poisson

st.set_page_config(page_title="COMBO - Advanced Betting Model", page_icon="⚽", layout="wide")

# ====================== CONFIG ======================
CAMPIONATI_DOMESTICI = {
    "Italia - Serie A": {"id_fd": "I1"},
    "Inghilterra - Premier League": {"id_fd": "E0"},
    "Spagna - La Liga": {"id_fd": "SP1"},
    "Germania - Bundesliga": {"id_fd": "D1"},
    "Francia - Ligue 1": {"id_fd": "F1"},
}

CAMPIONATI_COPPE = {
    "🌍 UEFA Champions League": {"code": "CL"},
    "🌍 UEFA Europa League": {"code": "EL"},
    "🌍 UEFA Conference League": {"code": "UEL"},  # UECL a volte non disponibile free
}

HEADERS_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

K_SHRINKAGE = 12

# ====================== UTILS ======================
def normalizza_nome_squadra(nome):
    nome = str(nome)
    for s in [" FC", " CF", " AFC", " AC", " SC", " CFC", " FK", " SK"]:
        nome = nome.replace(s, "")
    return nome.strip().lower()

def nomi_corrispondono(a, b):
    na, nb = normalizza_nome_squadra(a), normalizza_nome_squadra(b)
    return na == nb or na in nb or nb in na

def codici_stagione(oggi=None):
    oggi = oggi or date.today()
    anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    fmt = lambda a: f"{a % 100:02d}{(a + 1) % 100:02d}"
    return fmt(anno), fmt(anno - 1)

def tau_dixon_coles(gc, gt, lc, lt, rho):
    if gc == 0 and gt == 0: return 1 - (lc * lt * rho)
    if gc == 0 and gt == 1: return 1 + (lc * rho)
    if gc == 1 and gt == 0: return 1 + (lt * rho)
    if gc == 1 and gt == 1: return 1 - rho
    return 1.0

def media_ewma(serie, span):
    serie = serie.dropna()
    if len(serie) == 0: return None
    return serie.ewm(span=span, min_periods=1).mean().iloc[-1]

def media_pesata_decadimento(df, colonna, data_riferimento, emivita):
    if colonna not in df.columns: return None
    sub = df[[colonna, 'Date_parsed']].dropna()
    if len(sub) == 0: return None
    giorni = (data_riferimento - sub['Date_parsed']).dt.days.clip(lower=0)
    pesi = 0.5 ** (giorni / emivita)
    tot = pesi.sum()
    return sub[colonna].mean() if tot <= 0 else (sub[colonna] * pesi).sum() / tot

def scarica_csv_robusto(url, tentativi=3):
    varianti = [url]
    if "://www." in url:
        varianti.append(url.replace("://www.", "://"))
    else:
        varianti.append(url.replace("://", "://www."))
    ultimo = None
    for _ in range(tentativi):
        for u in varianti:
            try:
                r = requests.get(u, headers=HEADERS_BROWSER, timeout=15)
                r.raise_for_status()
                return pd.read_csv(io.StringIO(r.text)), None
            except Exception as e:
                ultimo = str(e)
        time.sleep(1.5)
    return None, ultimo

# ====================== DATA LOADERS ======================
@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_campionato(id_fd):
    corr, prec = codici_stagione()
    frames = []
    for cod in [prec, corr]:
        url = f"https://football-data.co.uk/mmz4281/{cod}/{id_fd}.csv"
        df, _ = scarica_csv_robusto(url)
        if df is not None:
            df.columns = df.columns.str.strip()
            frames.append(df)
    if not frames:
        return None
    dati = pd.concat(frames, ignore_index=True, sort=False)
    dati['Date_parsed'] = pd.to_datetime(dati['Date'], errors='coerce', dayfirst=True)
    return dati.dropna(subset=['Date_parsed']).sort_values('Date_parsed').reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def carica_tutti_i_campionati():
    tutti = []
    for info in CAMPIONATI_DOMESTICI.values():
        df = carica_dati_campionato(info["id_fd"])
        if df is not None:
            tutti.append(df)
    return pd.concat(tutti, ignore_index=True) if tutti else pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def carica_fixture_future(id_fd):
    df, _ = scarica_csv_robusto("https://football-data.co.uk/fixtures.csv")
    if df is None:
        return pd.DataFrame()
    fx = df.copy()
    fx.columns = fx.columns.str.strip()
    if 'Div' not in fx.columns:
        return pd.DataFrame()
    fx = fx[fx['Div'] == id_fd].copy()
    fx['Date_parsed'] = pd.to_datetime(fx['Date'], errors='coerce', dayfirst=True)
    oggi = pd.Timestamp(date.today())
    return fx[fx['Date_parsed'] >= oggi].sort_values('Date_parsed').reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_api_europee(codice, api_key):
    if not api_key:
        return "NO_KEY"
    headers = {"X-Auth-Token": api_key}
    url = f"https://api.football-data.org/v4/competitions/{codice}/matches"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 403:
            return "ERRORE_403"
        r.raise_for_status()
        matches = r.json().get("matches", [])
        rows = []
        for m in matches:
            status = m['status']
            fthg = ftag = None
            if status == 'FINISHED':
                fthg = m['score']['fullTime']['home']
                ftag = m['score']['fullTime']['away']
            rows.append({
                'Date': m['utcDate'][:10],
                'HomeTeam': m['homeTeam']['name'],
                'AwayTeam': m['awayTeam']['name'],
                'FTHG': fthg,
                'FTAG': ftag,
                'Status': status,
            })
        df = pd.DataFrame(rows)
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.sort_values('Date_parsed').reset_index(drop=True)
    except Exception as e:
        return str(e)

# ====================== MODELLO ======================
def estrai_partite_squadra(squadra, df_coppa, df_globale):
    for df in [df_coppa, df_globale]:
        if df is None or df.empty:
            continue
        mask = df['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra)) | \
               df['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
        f = df[mask]
        if len(f) > 0:
            return f
    return pd.DataFrame()

def calcola_modello(giocate, squadra_casa, squadra_trasferta, rho, ewma_span, emivita, df_globale, data_rif=None):
    giocate = giocate.dropna(subset=['FTHG', 'FTAG']) if giocate is not None else pd.DataFrame()
    if data_rif is None:
        data_rif = giocate['Date_parsed'].max() if not giocate.empty else pd.Timestamp(date.today())

    base = giocate
    if len(giocate) < 25 and df_globale is not None and not df_globale.empty:
        base = pd.concat([giocate, df_globale.dropna(subset=['FTHG', 'FTAG'])], ignore_index=True)

    m_gc = media_pesata_decadimento(base, 'FTHG', data_rif, emivita) or 1.55
    m_gt = media_pesata_decadimento(base, 'FTAG', data_rif, emivita) or 1.20

    forma_c = estrai_partite_squadra(squadra_casa, giocate, df_globale)
    forma_t = estrai_partite_squadra(squadra_trasferta, giocate, df_globale)
    n_c, n_t = len(forma_c), len(forma_t)

    gf_c = media_ewma(forma_c['FTHG'], ewma_span) if n_c else None
    gs_c = media_ewma(forma_c['FTAG'], ewma_span) if n_c else None
    gf_t = media_ewma(forma_t['FTAG'], ewma_span) if n_t else None
    gs_t = media_ewma(forma_t['FTHG'], ewma_span) if n_t else None

    peso_c = n_c / (n_c + K_SHRINKAGE)
    peso_t = n_t / (n_t + K_SHRINKAGE)

    att_c = peso_c * ((gf_c / max(0.1, m_gc)) if gf_c else 1.0) + (1 - peso_c) * 1.0
    dif_c = peso_c * ((gs_c / max(0.1, m_gt)) if gs_c else 1.0) + (1 - peso_c) * 1.0
    att_t = peso_t * ((gf_t / max(0.1, m_gt)) if gf_t else 1.0) + (1 - peso_t) * 1.0
    dif_t = peso_t * ((gs_t / max(0.1, m_gc)) if gs_t else 1.0) + (1 - peso_t) * 1.0

    lam_c = max(0.25, att_c * dif_t * m_gc)
    lam_t = max(0.25, att_t * dif_c * m_gt)

    griglia = []
    p1 = px = p2 = p_goal = p_nogoal = 0.0
    under = {1.5: 0.0, 2.5: 0.0, 3.5: 0.0}
    multi_c = {k: 0.0 for k in ["0-1", "0-2", "1-2", "1-3", "2-3", "2-4"]}
    multi_t = multi_c.copy()
    combo = {"1+Goal": 0.0, "1+Over2.5": 0.0, "X+Under2.5": 0.0, "2+Goal": 0.0}

    tot = 0.0
    for gc in range(8):
        for gt in range(8):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            tot += p
            segno = "X" if gc == gt else ("1" if gc > gt else "2")
            griglia.append({"gc": gc, "gt": gt, "p": p, "segno": segno, "ris": f"{gc}-{gt}"})

            if segno == "1": p1 += p
            elif segno == "X": px += p
            else: p2 += p
            if gc > 0 and gt > 0: p_goal += p
            else: p_nogoal += p
            for lim in under:
                if gc + gt < lim: under[lim] += p
            for key, (lo, hi) in [("0-1",(0,1)),("0-2",(0,2)),("1-2",(1,2)),("1-3",(1,3)),("2-3",(2,3)),("2-4",(2,4))]:
                if lo <= gc <= hi: multi_c[key] += p
                if lo <= gt <= hi: multi_t[key] += p
            if segno == "1" and gc > 0 and gt > 0: combo["1+Goal"] += p
            if segno == "1" and gc + gt > 2.5: combo["1+Over2.5"] += p
            if segno == "X" and gc + gt < 2.5: combo["X+Under2.5"] += p
            if segno == "2" and gc > 0 and gt > 0: combo["2+Goal"] += p

    if tot > 0:
        f = 100 / tot
        p1, px, p2 = p1*f, px*f, p2*f
        p_goal, p_nogoal = p_goal*f, p_nogoal*f
        under = {k: v*f for k,v in under.items()}
        multi_c = {k: v*f for k,v in multi_c.items()}
        multi_t = {k: v*f for k,v in multi_t.items()}
        combo = {k: v*f for k,v in combo.items()}
        for r in griglia:
            r["p"] *= f

    # Top risultati esatti
    top_risultati = sorted(griglia, key=lambda x: x["p"], reverse=True)[:12]

    return {
        "prob_1": p1, "prob_X": px, "prob_2": p2,
        "prob_goal": p_goal, "prob_nogoal": p_nogoal,
        "prob_under": under,
        "multigol_casa": multi_c, "multigol_trasf": multi_t,
        "combo": combo,
        "griglia": griglia,
        "top_risultati": top_risultati,
        "n_casa": n_c, "n_trasf": n_t,
        "lam_c": lam_c, "lam_t": lam_t
    }

def calcola_combo_libera(griglia, segno=None, soglia=None, tipo=None, gg=None):
    tot = 0.0
    for r in griglia:
        ok = True
        if segno and r["segno"] != segno: ok = False
        if ok and soglia is not None:
            tot_g = r["gc"] + r["gt"]
            if tipo == "Over" and not (tot_g > soglia): ok = False
            if tipo == "Under" and not (tot_g < soglia): ok = False
        if ok and gg:
            entrambe = r["gc"] > 0 and r["gt"] > 0
            if gg == "Goal" and not entrambe: ok = False
            if gg == "NoGoal" and entrambe: ok = False
        if ok: tot += r["p"]
    return tot

def kelly(prob, quota, frazione=0.5):
    """Kelly Criterion frazionario"""
    if quota <= 1 or prob <= 0:
        return 0.0
    b = quota - 1
    q = 1 - prob
    k = (b * prob - q) / b
    return max(0.0, k * frazione)

# ====================== UI ======================
st.title("⚽ COMBO — Advanced Betting Model")
st.caption("Dixon-Coles + EWMA + Shrinkage | Solo fonti gratuite | Nessun risultato fisso")

with st.sidebar:
    st.header("⚙️ Parametri")
    api_key = st.text_input("API Key football-data.org (opzionale per Coppe)", type="password",
                            help="Gratuita su football-data.org/client/register")
    rho = st.slider("Dixon-Coles ρ", -0.20, 0.10, -0.08, 0.01)
    ewma_span = st.slider("Finestra Forma (EWMA)", 3, 12, 6)
    emivita = st.slider("Decadimento (giorni)", 60, 300, 150)
    kelly_frac = st.slider("Frazione Kelly", 0.1, 1.0, 0.4, 0.05)

categoria = st.radio("Categoria", ["Campionati Nazionali", "Coppe Europee"], horizontal=True)

if categoria == "Campionati Nazionali":
    campionato = st.selectbox("Campionato", list(CAMPIONATI_DOMESTICI.keys()))
    id_fd = CAMPIONATI_DOMESTICI[campionato]["id_fd"]
    with st.spinner("Caricamento dati..."):
        dati = carica_dati_campionato(id_fd)
        future = carica_fixture_future(id_fd)
    df_globale = pd.DataFrame()
    is_coppa = False
else:
    if not api_key:
        st.warning("Per le Coppe Europee inserisci la chiave API gratuita di football-data.org nella sidebar.")
        st.info("Champions League è gratuita. Europa/Conference hanno copertura limitata nel piano free.")
        st.stop()
    campionato = st.selectbox("Coppa", list(CAMPIONATI_COPPE.keys()))
    code = CAMPIONATI_COPPE[campionato]["code"]
    with st.spinner("Caricamento Coppe + dati di supporto..."):
        risultato = carica_dati_api_europee(code, api_key)
        df_globale = carica_tutti_i_campionati()
    if risultato == "ERRORE_403":
        st.error("Accesso negato (403). Questa competizione non è inclusa nel piano gratuito.")
        st.stop()
    if isinstance(risultato, str):
        st.error(f"Errore: {risultato}")
        st.stop()
    dati = risultato
    future = dati[dati['Status'] != 'FINISHED'] if 'Status' in dati.columns else pd.DataFrame()
    is_coppa = True

if dati is None or len(dati) == 0:
    st.error("Nessun dato disponibile.")
    st.stop()

# Costruzione lista partite
opzioni = []
mappa = []
if not future.empty:
    for _, r in future.iterrows():
        opzioni.append(f"FUTURA ({r.get('Date', '?')}): {r.get('HomeTeam')} vs {r.get('AwayTeam')}")
        mappa.append(r.to_dict())
storiche = dati[dati['FTHG'].notna()].tail(12) if 'FTHG' in dati.columns else pd.DataFrame()
for _, r in storiche.iterrows():
    opzioni.append(f"RECENTE ({r.get('Date', '?')}): {r.get('HomeTeam')} vs {r.get('AwayTeam')}")
    mappa.append(r.to_dict())

if not opzioni:
    st.warning("Nessuna partita trovata.")
    st.stop()

scelta = st.selectbox("Seleziona partita", opzioni)
partita = mappa[opzioni.index(scelta)]

# No look-ahead
data_rif = partita.get('Date_parsed')
if pd.notna(data_rif):
    dati_f = dati[(dati['FTHG'].notna()) & (dati['Date_parsed'] < data_rif)].copy() if 'Date_parsed' in dati.columns else dati
    glob_f = df_globale[(df_globale['FTHG'].notna()) & (df_globale['Date_parsed'] < data_rif)].copy() if not df_globale.empty else df_globale
else:
    dati_f = dati[dati['FTHG'].notna()].copy() if 'FTHG' in dati.columns else dati
    glob_f = df_globale

modello = calcola_modello(dati_f, partita['HomeTeam'], partita['AwayTeam'],
                          rho, ewma_span, emivita, glob_f, data_rif)

st.subheader(f"📊 {partita['HomeTeam']} vs {partita['AwayTeam']}")

# Avviso qualità dati
avvisi = []
if modello["n_casa"] < 6: avvisi.append(f"{partita['HomeTeam']} ({modello['n_casa']} partite)")
if modello["n_trasf"] < 6: avvisi.append(f"{partita['AwayTeam']} ({modello['n_trasf']} partite)")
if avvisi:
    st.warning("⚠️ Dati scarsi per: " + " • ".join(avvisi) + " → il modello usa più la media di lega (shrinkage).")

# 1X2
col1, col2, col3 = st.columns(3)
col1.metric("1 (Casa)", f"{modello['prob_1']:.1f}%")
col2.metric("X", f"{modello['prob_X']:.1f}%")
col3.metric("2 (Trasferta)", f"{modello['prob_2']:.1f}%")

# Top risultati esatti
st.markdown("### 🎯 Risultati Esatti più Probabili")
df_top = pd.DataFrame(modello["top_risultati"])[["ris", "p", "segno"]]
df_top.columns = ["Risultato", "Prob %", "Segno"]
df_top["Prob %"] = df_top["Prob %"].round(1).astype(str) + "%"
st.dataframe(df_top, use_container_width=True, hide_index=True)

# Goal / Under-Over
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Goal / No Goal")
    st.write(f"**Goal**: {modello['prob_goal']:.1f}%")
    st.write(f"**No Goal**: {modello['prob_nogoal']:.1f}%")
with c2:
    st.markdown("#### Under / Over")
    for lim, pu in modello["prob_under"].items():
        st.write(f"Under {lim}: **{pu:.1f}%** | Over {lim}: **{100-pu:.1f}%**")

# Combo libera
st.markdown("### 🎯 Costruisci la tua Combo")
cc1, cc2, cc3 = st.columns(3)
segno_c = cc1.selectbox("Segno", ["Nessuno", "1", "X", "2"])
soglia_c = cc2.selectbox("Soglia gol", ["Nessuna", "1.5", "2.5", "3.5"])
tipo_c = cc2.radio("Tipo", ["Over", "Under"], horizontal=True, disabled=(soglia_c=="Nessuna"))
gg_c = cc3.selectbox("Gol/NoGol", ["Nessuno", "Goal", "NoGoal"])

if segno_c != "Nessuno" or soglia_c != "Nessuna" or gg_c != "Nessuno":
    prob = calcola_combo_libera(
        modello["griglia"],
        None if segno_c == "Nessuno" else segno_c,
        None if soglia_c == "Nessuna" else float(soglia_c),
        tipo_c if soglia_c != "Nessuna" else None,
        None if gg_c == "Nessuno" else gg_c
    )
    q_equa = 100 / prob if prob > 0 else 0
    pezzi = [p for p in [segno_c if segno_c!="Nessuno" else None,
                         f"{tipo_c} {soglia_c}" if soglia_c!="Nessuna" else None,
                         gg_c if gg_c!="Nessuno" else None] if p]
    st.success(f"**{' + '.join(pezzi)}** → **{prob:.1f}%** (Quota equa ≈ {q_equa:.2f})")

# Kelly (esempio su 1X2)
st.markdown("### 💰 Kelly Criterion (esempio)")
st.caption("Inserisci le quote del bookmaker per calcolare la puntata consigliata (frazione Kelly)")
k1, k2, k3 = st.columns(3)
q1 = k1.number_input("Quota 1", min_value=1.01, value=2.10, step=0.05)
qx = k2.number_input("Quota X", min_value=1.01, value=3.40, step=0.05)
q2 = k3.number_input("Quota 2", min_value=1.01, value=3.50, step=0.05)

kelly_1 = kelly(modello["prob_1"]/100, q1, kelly_frac)
kelly_x = kelly(modello["prob_X"]/100, qx, kelly_frac)
kelly_2 = kelly(modello["prob_2"]/100, q2, kelly_frac)

st.write(f"**Kelly 1**: {kelly_1*100:.1f}% del bankroll" if kelly_1 > 0 else "Kelly 1: nessun value")
st.write(f"**Kelly X**: {kelly_x*100:.1f}% del bankroll" if kelly_x > 0 else "Kelly X: nessun value")
st.write(f"**Kelly 2**: {kelly_2*100:.1f}% del bankroll" if kelly_2 > 0 else "Kelly 2: nessun value")

st.divider()
st.caption("Modello statistico a scopo informativo. Non è consiglio finanziario. Gioca responsabilmente.")
