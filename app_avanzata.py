import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import time
from datetime import date
from scipy.stats import poisson

st.set_page_config(
    page_title="COMBO - Advanced Betting Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CONFIGURAZIONE ======================
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
    "🌍 UEFA Conference League": {"code": "UEL"},
}

HEADERS_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

K_SHRINKAGE = 10

# ====================== FUNZIONI DI BASE ======================
def normalizza_nome_squadra(nome):
    nome = str(nome)
    for suffisso in [" FC", " CF", " AFC", " AC", " SC", " CFC", " FK", " SK"]:
        nome = nome.replace(suffisso, "")
    return nome.strip().lower()

def nomi_corrispondono(a, b):
    na = normalizza_nome_squadra(a)
    nb = normalizza_nome_squadra(b)
    return na == nb or na in nb or nb in na

def codici_stagione(oggi=None):
    oggi = oggi or date.today()
    anno_inizio = oggi.year if oggi.month >= 7 else oggi.year - 1
    fmt = lambda a: f"{a % 100:02d}{(a + 1) % 100:02d}"
    return fmt(anno_inizio), fmt(anno_inizio - 1)

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

def scarica_csv_robusto(url, tentativi=3, attesa=2):
    varianti = [url]
    if "://www." in url:
        varianti.append(url.replace("://www.", "://"))
    else:
        varianti.append(url.replace("://", "://www."))
    ultimo_errore = None
    for tentativo in range(tentativi):
        for u in varianti:
            try:
                resp = requests.get(u, headers=HEADERS_BROWSER, timeout=15)
                resp.raise_for_status()
                return pd.read_csv(io.StringIO(resp.text)), None
            except Exception as e:
                ultimo_errore = str(e)
        if tentativo < tentativi - 1:
            time.sleep(attesa)
    return None, ultimo_errore

# ====================== CARICAMENTO DATI ======================
@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_campionato(id_fd):
    corr, prec = codici_stagione()
    frames = []
    for codice in [prec, corr]:
        url = f"https://football-data.co.uk/mmz4281/{codice}/{id_fd}.csv"
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
    headers = {"X-Auth-Token": api_key}
    url = f"https://api.football-data.org/v4/competitions/{codice}/matches"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 403:
            return "ERRORE_403"
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
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

# ====================== ESTRAZIONE ======================
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

def estrai_scontri_diretti(squadra_casa, squadra_trasferta, df_coppa, df_globale):
    frames = []
    if df_coppa is not None and not df_coppa.empty:
        frames.append(df_coppa)
    if df_globale is not None and not df_globale.empty:
        frames.append(df_globale)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if 'FTHG' not in df.columns:
        return pd.DataFrame()
    mask_dir = df['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra_casa)) & \
               df['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra_trasferta))
    mask_inv = df['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra_trasferta)) & \
               df['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra_casa))
    h2h = df[df['FTHG'].notna() & df['FTAG'].notna() & (mask_dir | mask_inv)].copy()
    if 'Date_parsed' in h2h.columns:
        h2h = h2h.sort_values('Date_parsed', ascending=False)
    return h2h.head(6)

# ====================== MODELLO ======================
def calcola_modello_completo(giocate, squadra_casa, squadra_trasferta, rho, ewma_span, emivita, df_globale, data_rif=None):
    giocate = giocate.dropna(subset=['FTHG', 'FTAG']) if giocate is not None else pd.DataFrame()
    if data_rif is None:
        data_rif = giocate['Date_parsed'].max() if not giocate.empty else pd.Timestamp(date.today())

    base = giocate
    if len(giocate) < 20 and df_globale is not None and not df_globale.empty:
        base = pd.concat([giocate, df_globale.dropna(subset=['FTHG', 'FTAG'])], ignore_index=True)

    m_gol_casa = media_pesata_decadimento(base, 'FTHG', data_rif, emivita) or 1.55
    m_gol_trasf = media_pesata_decadimento(base, 'FTAG', data_rif, emivita) or 1.20

    forma_casa = estrai_partite_squadra(squadra_casa, giocate, df_globale)
    forma_trasf = estrai_partite_squadra(squadra_trasferta, giocate, df_globale)
    n_casa, n_trasf = len(forma_casa), len(forma_trasf)

    gf_casa = media_ewma(forma_casa['FTHG'], ewma_span) if n_casa else None
    gs_casa = media_ewma(forma_casa['FTAG'], ewma_span) if n_casa else None
    gf_trasf = media_ewma(forma_trasf['FTAG'], ewma_span) if n_trasf else None
    gs_trasf = media_ewma(forma_trasf['FTHG'], ewma_span) if n_trasf else None

    peso_casa = n_casa / (n_casa + K_SHRINKAGE)
    peso_trasf = n_trasf / (n_trasf + K_SHRINKAGE)

    att_casa = peso_casa * ((gf_casa / max(0.1, m_gol_casa)) if gf_casa is not None else 1.0) + (1 - peso_casa)
    dif_casa = peso_casa * ((gs_casa / max(0.1, m_gol_trasf)) if gs_casa is not None else 1.0) + (1 - peso_casa)
    att_trasf = peso_trasf * ((gf_trasf / max(0.1, m_gol_trasf)) if gf_trasf is not None else 1.0) + (1 - peso_trasf)
    dif_trasf = peso_trasf * ((gs_trasf / max(0.1, m_gol_casa)) if gs_trasf is not None else 1.0) + (1 - peso_trasf)

    lam_c = max(0.25, att_casa * dif_trasf * m_gol_casa)
    lam_t = max(0.25, att_trasf * dif_casa * m_gol_trasf)

    griglia = []
    p1 = px = p2 = p_goal = p_nogoal = 0.0
    under = {1.5: 0.0, 2.5: 0.0, 3.5: 0.0}
    multi_c = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    multi_t = multi_c.copy()

    combo_classiche = {
        "1 + Goal": 0.0, "1 + No Goal": 0.0,
        "X + Goal": 0.0, "X + No Goal": 0.0,
        "2 + Goal": 0.0, "2 + No Goal": 0.0,
        "1 + Over 2.5": 0.0, "1 + Under 2.5": 0.0,
        "X + Over 2.5": 0.0, "X + Under 2.5": 0.0,
        "2 + Over 2.5": 0.0, "2 + Under 2.5": 0.0,
    }

    super_combo = {}

    tot = 0.0
    for gc in range(8):
        for gt in range(8):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            tot += p
            segno = "X" if gc == gt else ("1" if gc > gt else "2")
            entrambe_segnano = gc > 0 and gt > 0
            tot_gol = gc + gt

            griglia.append({"gc": gc, "gt": gt, "p": p, "segno": segno})

            if segno == "1": p1 += p
            elif segno == "X": px += p
            else: p2 += p

            if entrambe_segnano:
                p_goal += p
            else:
                p_nogoal += p

            for lim in under:
                if tot_gol < lim:
                    under[lim] += p

            for key, (lo, hi) in [("0-1",(0,1)), ("0-2",(0,2)), ("1-2",(1,2)), ("1-3",(1,3)), ("2-3",(2,3)), ("2-4",(2,4))]:
                if lo <= gc <= hi: multi_c[key] += p
                if lo <= gt <= hi: multi_t[key] += p

            # Combo classiche
            if segno == "1" and entrambe_segnano: combo_classiche["1 + Goal"] += p
            if segno == "1" and not entrambe_segnano: combo_classiche["1 + No Goal"] += p
            if segno == "X" and entrambe_segnano: combo_classiche["X + Goal"] += p
            if segno == "X" and not entrambe_segnano: combo_classiche["X + No Goal"] += p
            if segno == "2" and entrambe_segnano: combo_classiche["2 + Goal"] += p
            if segno == "2" and not entrambe_segnano: combo_classiche["2 + No Goal"] += p

            if segno == "1" and tot_gol > 2.5: combo_classiche["1 + Over 2.5"] += p
            if segno == "1" and tot_gol < 2.5: combo_classiche["1 + Under 2.5"] += p
            if segno == "X" and tot_gol > 2.5: combo_classiche["X + Over 2.5"] += p
            if segno == "X" and tot_gol < 2.5: combo_classiche["X + Under 2.5"] += p
            if segno == "2" and tot_gol > 2.5: combo_classiche["2 + Over 2.5"] += p
            if segno == "2" and tot_gol < 2.5: combo_classiche["2 + Under 2.5"] += p

            # Super Combo
            gg = "Goal" if entrambe_segnano else "No Goal"
            ou = "Over 2.5" if tot_gol > 2.5 else "Under 2.5"
            chiave = f"{segno} + {gg} + {ou}"
            super_combo[chiave] = super_combo.get(chiave, 0.0) + p

    if tot > 0:
        f = 100.0 / tot
        p1, px, p2 = p1 * f, px * f, p2 * f
        p_goal, p_nogoal = p_goal * f, p_nogoal * f
        under = {k: v * f for k, v in under.items()}
        multi_c = {k: v * f for k, v in multi_c.items()}
        multi_t = {k: v * f for k, v in multi_t.items()}
        combo_classiche = {k: v * f for k, v in combo_classiche.items()}
        super_combo = {k: v * f for k, v in super_combo.items()}
        for r in griglia:
            r["p"] *= f

    return {
        "prob_1": p1, "prob_X": px, "prob_2": p2,
        "prob_goal": p_goal, "prob_nogoal": p_nogoal,
        "prob_under": under,
        "multigol_casa": multi_c,
        "multigol_trasf": multi_t,
        "combo_classiche": combo_classiche,
        "super_combo": super_combo,
        "n_casa": n_casa,
        "n_trasf": n_trasf,
    }

def crea_tabella(dati_dict, nome_col="Mercato"):
    df = pd.DataFrame(list(dati_dict.items()), columns=[nome_col, "Probabilità (%)"])
    df["Probabilità (%)"] = df["Probabilità (%)"].round(1)
    df = df.sort_values("Probabilità (%)", ascending=False).reset_index(drop=True)
    df["Probabilità (%)"] = df["Probabilità (%)"].astype(str) + "%"
    return df

# ====================== INTERFACCIA ======================
st.title("⚽ COMBO — Advanced Betting Model")
st.caption("Dixon-Coles + EWMA + Shrinkage • Solo fonti gratuite")

with st.sidebar:
    st.header("⚙️ Configurazione")
    api_key = st.text_input("Chiave API football-data.org (per le Coppe)", type="password")
    st.divider()
    rho = st.slider("Dixon-Coles ρ", -0.20, 0.10, -0.10, 0.01)
    ewma_span = st.slider("Finestra Forma", 3, 12, 6)
    emivita = st.slider("Decadimento (giorni)", 60, 300, 150)

categoria = st.radio("Categoria", ["Campionati Nazionali", "Coppe Europee"], horizontal=True)

if categoria == "Campionati Nazionali":
    campionato = st.selectbox("Campionato", list(CAMPIONATI_DOMESTICI.keys()))
    id_fd = CAMPIONATI_DOMESTICI[campionato]["id_fd"]
    with st.spinner("Caricamento dati..."):
        dati = carica_dati_campionato(id_fd)
        future = carica_fixture_future(id_fd)
    df_globale = pd.DataFrame()
else:
    if not api_key:
        st.warning("Inserisci la chiave API gratuita per le Coppe.")
        st.stop()

    campionato = st.selectbox("Coppa", list(CAMPIONATI_COPPE.keys()))
    code = CAMPIONATI_COPPE[campionato]["code"]

    with st.spinner("Caricamento dati Coppe..."):
        risultato = carica_dati_api_europee(code, api_key)
        df_globale = carica_tutti_i_campionati()

    # === FIX ERRORE CHAMPIONS ===
    if isinstance(risultato, str):
        if risultato == "ERRORE_403":
            st.error("Questa competizione non è disponibile nel piano gratuito dell'API.")
            st.stop()
        else:
            st.error(f"Errore di connessione: {risultato}")
            st.stop()

    # Se arriviamo qui è un DataFrame
    dati = risultato
    future = dati[dati["Status"] != "FINISHED"] if "Status" in dati.columns else pd.DataFrame()

if dati is None or len(dati) == 0:
    st.error("Nessun dato disponibile.")
    st.stop()

# Lista partite
opzioni = []
mappa = []

if not future.empty:
    for _, r in future.iterrows():
        opzioni.append(f"FUTURA ({r.get('Date', '?')}): {r.get('HomeTeam')} vs {r.get('AwayTeam')}")
        mappa.append(r.to_dict())

storiche = dati[dati["FTHG"].notna()].tail(12) if "FTHG" in dati.columns else pd.DataFrame()
for _, r in storiche.iterrows():
    opzioni.append(f"RECENTE ({r.get('Date', '?')}): {r.get('HomeTeam')} vs {r.get('AwayTeam')}")
    mappa.append(r.to_dict())

if not opzioni:
    st.warning("Nessuna partita trovata.")
    st.stop()

scelta = st.selectbox("Seleziona la partita", opzioni)
partita = mappa[opzioni.index(scelta)]

# Filtro no-look-ahead
data_rif = partita.get("Date_parsed")
if pd.notna(data_rif):
    dati_f = dati[(dati["FTHG"].notna()) & (dati["Date_parsed"] < data_rif)].copy() if "Date_parsed" in dati.columns else dati
    glob_f = df_globale[(df_globale["FTHG"].notna()) & (df_globale["Date_parsed"] < data_rif)].copy() if not df_globale.empty else df_globale
else:
    dati_f = dati[dati["FTHG"].notna()].copy() if "FTHG" in dati.columns else dati
    glob_f = df_globale

modello = calcola_modello_completo(
    dati_f, partita["HomeTeam"], partita["AwayTeam"],
    rho, ewma_span, emivita, glob_f, data_rif
)

# ====================== VISUALIZZAZIONE ======================
st.markdown(f"## {partita['HomeTeam']}  vs  {partita['AwayTeam']}")

avvisi = []
if modello["n_casa"] < 6:
    avvisi.append(f"{partita['HomeTeam']} ({modello['n_casa']} partite)")
if modello["n_trasf"] < 6:
    avvisi.append(f"{partita['AwayTeam']} ({modello['n_trasf']} partite)")
if avvisi:
    st.warning("Dati limitati per: " + " • ".join(avvisi))

# 1X2
st.markdown("### Esito Finale (1X2)")
c1, c2, c3 = st.columns(3)
c1.metric("1 - Casa", f"{modello['prob_1']:.1f}%")
c2.metric("X - Pareggio", f"{modello['prob_X']:.1f}%")
c3.metric("2 - Trasferta", f"{modello['prob_2']:.1f}%")

# Goal / Under-Over
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Goal / No Goal")
    st.dataframe(crea_tabella({
        "Goal": modello["prob_goal"],
        "No Goal": modello["prob_nogoal"]
    }, "Mercato"), use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Under / Over")
    oo = {}
    for lim, pu in modello["prob_under"].items():
        oo[f"Under {lim}"] = pu
        oo[f"Over {lim}"] = 100 - pu
    st.dataframe(crea_tabella(oo, "Linea"), use_container_width=True, hide_index=True)

# Multigol
col3, col4 = st.columns(2)
with col3:
    st.markdown("### Multigol Casa")
    st.dataframe(crea_tabella(modello["multigol_casa"], "Intervallo"), use_container_width=True, hide_index=True)
with col4:
    st.markdown("### Multigol Ospite")
    st.dataframe(crea_tabella(modello["multigol_trasf"], "Intervallo"), use_container_width=True, hide_index=True)

# Combo Classiche
st.markdown("### Combo Classiche")
st.dataframe(crea_tabella(modello["combo_classiche"], "Combo"), use_container_width=True, hide_index=True)

# Top 5 Super Combo
st.markdown("### Top 5 Super Combo")
top5_super = dict(sorted(modello["super_combo"].items(), key=lambda x: x[1], reverse=True)[:5])
st.dataframe(crea_tabella(top5_super, "Super Combo"), use_container_width=True, hide_index=True)

# Scontri diretti
st.markdown("### Ultimi Scontri Diretti")
h2h = estrai_scontri_diretti(partita["HomeTeam"], partita["AwayTeam"], dati_f, glob_f)
if not h2h.empty:
    cols = [c for c in ["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam"] if c in h2h.columns]
    st.dataframe(h2h[cols], use_container_width=True, hide_index=True)
else:
    st.info("Nessun precedente recente trovato.")

st.divider()
st.caption("Modello statistico a scopo informativo. Non costituisce consiglio di scommessa.")
