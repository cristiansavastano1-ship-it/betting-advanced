
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import date
from scipy.stats import poisson

st.set_page_config(page_title="Advanced Betting Model", page_icon="⚽", layout="centered")

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
    "🌍 UEFA Conference League": {"code": "UECL"},
}

HEADERS_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def codici_stagione(oggi=None):
    oggi = oggi or date.today()
    anno_inizio_corrente = oggi.year if oggi.month >= 7 else oggi.year - 1
    anno_inizio_precedente = anno_inizio_corrente - 1
    fmt = lambda a: f"{a % 100:02d}{(a + 1) % 100:02d}"
    return fmt(anno_inizio_corrente), fmt(anno_inizio_precedente)

def tau_dixon_coles(gc, gt, lc, lt, rho):
    if gc == 0 and gt == 0: return 1 - (lc * lt * rho)
    elif gc == 0 and gt == 1: return 1 + (lc * rho)
    elif gc == 1 and gt == 0: return 1 + (lt * rho)
    elif gc == 1 and gt == 1: return 1 - rho
    return 1.0

def media_ewma(serie, span):
    serie = serie.dropna()
    if len(serie) == 0: return None
    return serie.ewm(span=span, min_periods=1).mean().iloc[-1]

def media_pesata_decadimento(df, colonna, data_riferimento, emivita):
    sub = df[[colonna, 'Date_parsed']].dropna()
    if len(sub) == 0: return None
    giorni = (data_riferimento - sub['Date_parsed']).dt.days.clip(lower=0)
    pesi = 0.5 ** (giorni / emivita)
    tot = pesi.sum()
    return sub[colonna].mean() if tot <= 0 else (sub[colonna] * pesi).sum() / tot

def calcola_modello_completo(giocate, squadra_casa, squadra_trasferta, rho, ewma_span, emivita, data_riferimento=None):
    n_storico = len(giocate)
    if n_storico < 10: return None
    if data_riferimento is None or pd.isna(data_riferimento):
        data_riferimento = giocate['Date_parsed'].max()

    m_gol_casa = media_pesata_decadimento(giocate, 'FTHG', data_riferimento, emivita)
    m_gol_trasf = media_pesata_decadimento(giocate, 'FTAG', data_riferimento, emivita)
    if m_gol_casa is None or m_gol_trasf is None: return None

    forma_casa = giocate[giocate['HomeTeam'] == squadra_casa]
    forma_trasf = giocate[giocate['AwayTeam'] == squadra_trasferta]

    gf_casa_rec = media_ewma(forma_casa['FTHG'], ewma_span) or m_gol_casa
    gs_casa_rec = media_ewma(forma_casa['FTAG'], ewma_span) or m_gol_trasf
    gf_trasf_rec = media_ewma(forma_trasf['FTAG'], ewma_span) or m_gol_trasf
    gs_trasf_rec = media_ewma(forma_trasf['FTHG'], ewma_span) or m_gol_casa

    tiri_casa = (media_ewma(forma_casa['HST'], ewma_span) if 'HST' in forma_casa.columns and not forma_casa['HST'].dropna().empty else 4.0) or 4.0
    corner_casa = (media_ewma(forma_casa['HC'], ewma_span) if 'HC' in forma_casa.columns and not forma_casa['HC'].dropna().empty else 5.0) or 5.0
    tiri_trasf = (media_ewma(forma_trasf['AST'], ewma_span) if 'AST' in forma_trasf.columns and not forma_trasf['AST'].dropna().empty else 3.5) or 3.5
    corner_trasf = (media_ewma(forma_trasf['AC'], ewma_span) if 'AC' in forma_trasf.columns and not forma_trasf['AC'].dropna().empty else 4.5) or 4.5

    lam_c = (gf_casa_rec / max(0.1, m_gol_casa)) * (gs_trasf_rec / max(0.1, m_gol_trasf)) * m_gol_casa
    lam_t = (gf_trasf_rec / max(0.1, m_gol_trasf)) * (gs_casa_rec / max(0.1, m_gol_casa)) * m_gol_trasf

    risultati = []
    prob_1, prob_x, prob_2 = 0.0, 0.0, 0.0
    prob_goal, prob_nogoal = 0.0, 0.0
    limiti_under = [1.5, 2.5, 3.5]
    prob_under = {l: 0.0 for l in limiti_under}
    
    multigol_casa = {"1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    multigol_trasf = {"1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    combo_stats = {"1_e_gol": 0.0, "1_e_over25": 0.0, "x_e_under25": 0.0, "2_e_gol": 0.0}

    for gc in range(8):
        for gt in range(8):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            segno = 'X' if gc == gt else ('1' if gc > gt else '2')
            
            if segno == '1': prob_1 += p
            elif segno == 'X': prob_x += p
            else: prob_2 += p
            
            if gc > 0 and gt > 0: prob_goal += p
            else: prob_nogoal += p
            
            for l in limiti_under:
                if gc + gt < l: prob_under[l] += p
                
            for mg_key, (mi_c, ma_c) in [("1-2", (1,2)), ("1-3", (1,3)), ("2-3", (2,3)), ("2-4", (2,4))]:
                if mi_c <= gc <= ma_c: multigol_casa[mg_key] += p
                if mi_c <= gt <= ma_c: multigol_trasf[mg_key] += p

            if segno == '1' and gc > 0 and gt > 0: combo_stats["1_e_gol"] += p
            if segno == '1' and (gc + gt) > 2.5: combo_stats["1_e_over25"] += p
            if segno == 'X' and (gc + gt) < 2.5: combo_stats["x_e_under25"] += p
            if segno == '2' and gc > 0 and gt > 0: combo_stats["2_e_gol"] += p

            risultati.append({'res': f"{gc}-{gt}", 'p': p, 'segno': segno})

    tot = sum(r['p'] for r in risultati)
    if tot > 0:
        f = 100.0 / tot
        for r in risultati: r['p'] *= f
        prob_1, prob_x, prob_2 = prob_1*f, prob_x*f, prob_2*f
        prob_goal, prob_nogoal = prob_goal*f, prob_nogoal*f
        prob_under = {l: v*f for l, v in prob_under.items()}
        multigol_casa = {k: v*f for k, v in multigol_casa.items()}
        multigol_trasf = {k: v*f for k, v in multigol_trasf.items()}
        combo_stats = {k: v*f for k, v in combo_stats.items()}

    risultati_ordinati = sorted(risultati, key=lambda x: x['p'], reverse=True)

    return {
        "lambda_casa": lam_c, "lambda_trasferta": lam_t,
        "prob_1": prob_1, "prob_X": prob_x, "prob_2": prob_2,
        "prob_goal": prob_goal, "prob_nogoal": prob_nogoal,
        "prob_under": prob_under, "multigol_casa": multigol_casa, "multigol_trasf": multigol_trasf,
        "combo": combo_stats, "angoli_stimati": f"{corner_casa + corner_trasf:.1f}",
        "tiri_stimati": f"{tiri_casa + tiri_trasf:.1f}", "risultati": risultati_ordinati
    }

def scarica_csv_robusto(url):
    try:
        resp = requests.get(url, headers=HEADERS_BROWSER, timeout=15)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text)), None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_campionato(id_fd):
    codice_corrente, codice_precedente = codici_stagione()
    frames = []
    for codice in [codice_precedente, codice_corrente]:
        url = f"https://football-data.co.uk/mmz4281/{codice}/{id_fd}.csv"
        df, _ = scarica_csv_robusto(url)
        if df is not None:
            df.columns = df.columns.str.strip()
            frames.append(df)
    if frames:
        dati = pd.concat(frames, ignore_index=True, sort=False)
        dati['Date_parsed'] = pd.to_datetime(dati['Date'], errors='coerce', dayfirst=True)
        return dati.dropna(subset=['Date_parsed']).sort_values('Date_parsed').reset_index(drop=True)
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def carica_fixture_future(id_fd):
    df, _ = scarica_csv_robusto("https://football-data.co.uk/fixtures.csv")
    if df is not None:
        fx = df.copy()
        fx.columns = fx.columns.str.strip()
        if 'Div' in fx.columns:
            fx = fx[fx['Div'] == id_fd].copy()
            fx['Date_parsed'] = pd.to_datetime(fx['Date'], errors='coerce', dayfirst=True)
            oggi = pd.Timestamp(date.today())
            return fx[fx['Date_parsed'] >= oggi].sort_values('Date_parsed').reset_index(drop=True)
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_api_europee(codice_competizione, api_key):
    headers = {"X-Auth-Token": api_key}
    url_matches = f"https://api.football-data.org/v4/competitions/{codice_competizione}/matches"
    try:
        resp = requests.get(url_matches, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches", [])
        
        rows = []
        for m in matches:
            home = m['homeTeam']['name']
            away = m['awayTeam']['name']
            date_str = m['utcDate'][:10]
            status = m['status']
            fthg, ftag = None, None
            if status == 'FINISHED':
                fthg = m['score']['fullTime']['home']
                ftag = m['score']['fullTime']['away']
            
            rows.append({
                'Date': date_str,
                'HomeTeam': home,
                'AwayTeam': away,
                'FTHG': fthg,
                'FTAG': ftag,
                'Status': status
            })
        df = pd.DataFrame(rows)
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception as e:
        return None

st.title("⚽ Advanced Pro Betting Analyzer")
st.caption("Modello Statistico con Parametri Dinamici, Ordinamento Decrescente & Coppe Europee")

with st.sidebar:
    st.header("⚙️ Configurazione & API")
    api_key_input = st.text_input("Chiave API football-data.org (per Coppe)", type="password", help="Necessaria per Champions, Europa e Conference League")
    st.divider()
    rho_val = st.slider("Correzione Dixon-Coles (ρ)", -0.20, 0.10, -0.10, 0.01)
    ewma_span_val = st.slider("Finestra Forma Recente (EWMA)", 2, 15, 6, 1)
    emivita_val = st.slider("Decadimento Temporale (Giorni)", 30, 365, 180, 10)

scelta_categoria = st.radio("Categoria Torneo", ["Campionati Nazionali (Gratuiti)", "Coppe Europee (Richiede API Key)"], horizontal=True)

if scelta_categoria == "Campionati Nazionali (Gratuiti)":
    campionato = st.selectbox("Seleziona Campionato", list(CAMPIONATI_DOMESTICI.keys()))
    info = CAMPIONATI_DOMESTICI[campionato]
    id_fd = info["id_fd"]
    
    with st.spinner("Caricamento dataset campionato..."):
        dati = carica_dati_campionato(id_fd)
        fixture_future = carica_fixture_future(id_fd)
        
    if dati is None or len(dati) == 0:
        st.error("Impossibile scaricare i dati. Riprova tra poco.")
        st.stop()
        
    opzioni_partite = []
    mappa_partite = []
    if not fixture_future.empty:
        for _, r in fixture_future.iterrows():
            opzioni_partite.append(f"FUTURA ({r.get('Date','?')}): {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}")
            mappa_partite.append(r.to_dict())

    storiche = dati[dati['FTHG'].notna()].tail(15)
    for _, r in storiche.iterrows():
        opzioni_partite.append(f"RECENTE ({r.get('Date','?')}): {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}")
        mappa_partite.append(r.to_dict())
        
else:
    if not api_key_input:
        st.warning("⚠️ Inserisci la tua chiave API di football-data.org nella barra laterale per sbloccare le coppe europee.")
        st.stop()
    campionato = st.selectbox("Seleziona Coppe", list(CAMPIONATI_COPPE.keys()))
    info = CAMPIONATI_COPPE[campionato]
    code_api = info["code"]
    
    with st.spinner("Connessione alle API delle Coppe Europee..."):
        dati = carica_dati_api_europee(code_api, api_key_input)
        
    if dati is None or len(dati) == 0:
        st.error("Errore nel recupero dati API o chiave non valida.")
        st.stop()
        
    opzioni_partite = []
    mappa_partite = []
    for _, r in dati.iterrows():
        stato_txt = "FUTURA" if r['Status'] != 'FINISHED' else "GIOCATA"
        opzioni_partite.append(f"{stato_txt} ({r['Date']}): {r['HomeTeam']} vs {r['AwayTeam']}")
        mappa_partite.append(r.to_dict())

if not opzioni_partite:
    st.warning("Nessuna partita disponibile al momento.")
else:
    scelta = st.selectbox("Seleziona Partita", opzioni_partite)
    idx_sel = opzioni_partite.index(scelta)
    partita_sel = mappa_partite[idx_sel]
    
    modello = calcola_modello_completo(dati, partita_sel['HomeTeam'], partita_sel['AwayTeam'], rho_val, ewma_span_val, emivita_val)
    
    if modello is None:
        st.error("Campione insufficiente per elaborare le statistiche di questa partita.")
    else:
        st.subheader(f"📊 Analisi Match: {partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("1 (Casa)", f"{modello['prob_1']:.1f}%")
        c2.metric("X (Pareggio)", f"{modello['prob_X']:.1f}%")
        c3.metric("2 (Trasferta)", f"{modello['prob_2']:.1f}%")
        
        st.divider()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**⚽ Goal / No Goal**")
            st.write(f"- Goal: **{modello['prob_goal']:.1f}%**")
            st.write(f"- No Goal: **{modello['prob_nogoal']:.1f}%**")
        with col_b:
            st.markdown("**📉 Under / Over**")
            for soglia, prob in sorted(modello['prob_under'].items(), key=lambda x: x[1], reverse=True):
                st.write(f"- Under {soglia}: **{prob:.1f}%** | Over {soglia}: **{100-prob:.1f}%**")

        st.divider()

        st.markdown("**🔥 Top Risultati Esatti (Ordinati per % decrescente)**")
        top_risultati = modello['risultati'][:5]
        for tr in top_risultati:
            st.write(f"- Risultato **{tr['res']}** ({tr['segno']}): **{tr['p']:.1f}%**")

        st.divider()
        st.markdown("**🎯 Statistiche Match (Stimate)**")
        st.info(f"Angoli Totali Stimati: **{modello['angoli_stimati']}** | Tiri in Porta Totali Stimati: **{modello['tiri_stimati']}**")
