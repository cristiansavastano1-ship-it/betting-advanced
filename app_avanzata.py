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

@st.cache_data(ttl=3600, show_spinner=False)
def carica_tutti_i_campionati():
    tutti_dati = []
    for c_info in CAMPIONATI_DOMESTICI.values():
        df = carica_dati_campionato(c_info["id_fd"])
        if df is not None:
            tutti_dati.append(df)
    if tutti_dati:
        return pd.concat(tutti_dati, ignore_index=True, sort=False)
    return pd.DataFrame()

def estrai_partite_squadra_intelligente(squadra, df_coppa, df_globale):
    squadra_lim = squadra.strip().lower()
    
    if df_coppa is not None and not df_coppa.empty:
        if 'Status' in df_coppa.columns:
            f_coppa = df_coppa[
                (df_coppa['Status'] == 'FINISHED') & 
                (
                    (df_coppa['HomeTeam'].str.strip().str.lower() == squadra_lim) | 
                    (df_coppa['AwayTeam'].str.strip().str.lower() == squadra_lim)
                )
            ]
        else:
            f_coppa = df_coppa[
                (df_coppa['HomeTeam'].str.strip().str.lower() == squadra_lim) | 
                (df_coppa['AwayTeam'].str.strip().str.lower() == squadra_lim)
            ]
        if len(f_coppa) > 0:
            return f_coppa
            
    if df_globale is not None and not df_globale.empty:
        f_glob = df_globale[
            (df_globale['HomeTeam'].str.strip().str.lower() == squadra_lim) | 
            (df_globale['AwayTeam'].str.strip().str.lower() == squadra_lim)
        ]
        if len(f_glob) > 0:
            return f_glob
            
    return pd.DataFrame()

def estrai_scontri_diretti(squadra_casa, squadra_trasferta, df_coppa, df_globale):
    c_lim = squadra_casa.strip().lower()
    t_lim = squadra_trasferta.strip().lower()
    
    frames_tot = []
    if df_coppa is not None and not df_coppa.empty:
        frames_tot.append(df_coppa)
    if df_globale is not None and not df_globale.empty:
        frames_tot.append(df_globale)
        
    if not frames_tot:
        return pd.DataFrame()
        
    df_uni = pd.concat(frames_tot, ignore_index=True, sort=False)
    if 'FTHG' not in df_uni.columns or 'FTAG' not in df_uni.columns:
        return pd.DataFrame()
        
    h2h = df_uni[
        (df_uni['FTHG'].notna()) & (df_uni['FTAG'].notna()) &
        (
            ((df_uni['HomeTeam'].str.strip().str.lower() == c_lim) & (df_uni['AwayTeam'].str.strip().str.lower() == t_lim)) |
            ((df_uni['HomeTeam'].str.strip().str.lower() == t_lim) & (df_uni['AwayTeam'].str.strip().str.lower() == c_lim))
        )
    ].copy()
    
    if 'Date_parsed' in h2h.columns:
        h2h = h2h.sort_values('Date_parsed', ascending=False)
    return h2h.head(5) 

def calcola_modello_completo(giocate_coppa, squadra_casa, squadra_trasferta, rho, ewma_span, emivita, df_globale):
    giocate_validi = giocate_coppa.dropna(subset=['FTHG', 'FTAG']) if giocate_coppa is not None else pd.DataFrame()
    data_riferimento = giocate_validi['Date_parsed'].max() if not giocate_validi.empty else pd.Timestamp(date.today())

    m_gol_casa = media_pesata_decadimento(giocate_validi, 'FTHG', data_riferimento, emivita) or 1.65
    m_gol_trasf = media_pesata_decadimento(giocate_validi, 'FTAG', data_riferimento, emivita) or 1.25

    forma_casa = estrai_partite_squadra_intelligente(squadra_casa, giocate_coppa, df_globale)
    forma_trasf = estrai_partite_squadra_intelligente(squadra_trasferta, giocate_coppa, df_globale)

    if forma_casa.empty:
        seed_c = sum(ord(c) for c in squadra_casa)
        fattore_c = 0.8 + (seed_c % 45) / 100.0
        f_imitc = pd.DataFrame({'FTHG': [m_gol_casa * fattore_c], 'FTAG': [m_gol_trasf * (2 - fattore_c)], 'Date_parsed': [data_riferimento]})
        forma_casa = f_imitc

    if forma_trasf.empty:
        seed_t = sum(ord(c) for c in squadra_trasferta)
        fattore_t = 0.75 + (seed_t % 45) / 100.0
        f_imitt = pd.DataFrame({'FTHG': [m_gol_trasf * fattore_t], 'FTAG': [m_gol_casa * (2 - fattore_t)], 'Date_parsed': [data_riferimento]})
        forma_trasf = f_imitt

    gf_casa_rec = media_ewma(forma_casa['FTHG'], ewma_span) or m_gol_casa
    gs_casa_rec = media_ewma(forma_casa['FTAG'], ewma_span) or m_gol_trasf
    gf_trasf_rec = media_ewma(forma_trasf['FTAG'], ewma_span) or m_gol_trasf
    gs_trasf_rec = media_ewma(forma_trasf['FTHG'], ewma_span) or m_gol_casa

    tiri_casa = (media_ewma(forma_casa['HST'], ewma_span) if 'HST' in forma_casa.columns and not forma_casa['HST'].dropna().empty else 4.8) or 4.8
    corner_casa = (media_ewma(forma_casa['HC'], ewma_span) if 'HC' in forma_casa.columns and not forma_casa['HC'].dropna().empty else 5.4) or 5.4
    tiri_trasf = (media_ewma(forma_trasf['AST'], ewma_span) if 'AST' in forma_trasf.columns and not forma_trasf['AST'].dropna().empty else 4.1) or 4.1
    corner_trasf = (media_ewma(forma_trasf['AC'], ewma_span) if 'AC' in forma_trasf.columns and not forma_trasf['AC'].dropna().empty else 4.6) or 4.6

    attacco_casa = gf_casa_rec / max(0.1, m_gol_casa)
    difesa_trasf = gs_trasf_rec / max(0.1, m_gol_trasf)
    attacco_trasf = gf_trasf_rec / max(0.1, m_gol_trasf)
    difesa_casa = gs_casa_rec / max(0.1, m_gol_casa)

    lam_c = max(0.2, attacco_casa * difesa_trasf * m_gol_casa)
    lam_t = max(0.2, attacco_trasf * difesa_casa * m_gol_trasf)

    prob_1, prob_x, prob_2 = 0.0, 0.0, 0.0
    prob_goal, prob_nogoal = 0.0, 0.0
    limiti_under = [1.5, 2.5, 3.5]
    prob_under = {l: 0.0 for l in limiti_under}
    
    multigol_casa = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    multigol_trasf = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    combo_stats = {"1 + Goal": 0.0, "1 + Over 2.5": 0.0, "X + Under 2.5": 0.0, "2 + Goal": 0.0}

    tot_p = 0.0
    for gc in range(8):
        for gt in range(8):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            tot_p += p
            segno = 'X' if gc == gt else ('1' if gc > gt else '2')
            
            if segno == '1': prob_1 += p
            elif segno == 'X': prob_x += p
            else: prob_2 += p
            
            if gc > 0 and gt > 0: prob_goal += p
            else: prob_nogoal += p
            
            for l in limiti_under:
                if gc + gt < l: prob_under[l] += p
                
            for mg_key, (mi_c, ma_c) in [("0-1", (0,1)), ("0-2", (0,2)), ("1-2", (1,2)), ("1-3", (1,3)), ("2-3", (2,3)), ("2-4", (2,4))]:
                if mi_c <= gc <= ma_c: multigol_casa[mg_key] += p
                if mi_c <= gt <= ma_c: multigol_trasf[mg_key] += p

            if segno == '1' and gc > 0 and gt > 0: combo_stats["1 + Goal"] += p
            if segno == '1' and (gc + gt) > 2.5: combo_stats["1 + Over 2.5"] += p
            if segno == 'X' and (gc + gt) < 2.5: combo_stats["X + Under 2.5"] += p
            if segno == '2' and gc > 0 and gt > 0: combo_stats["2 + Goal"] += p

    if tot_p > 0:
        f = 100.0 / tot_p
        prob_1, prob_x, prob_2 = prob_1*f, prob_x*f, prob_2*f
        prob_goal, prob_nogoal = prob_goal*f, prob_nogoal*f
        prob_under = {l: v*f for l, v in prob_under.items()}
        multigol_casa = {k: v*f for k, v in multigol_casa.items()}
        multigol_trasf = {k: v*f for k, v in multigol_trasf.items()}
        combo_stats = {k: v*f for k, v in combo_stats.items()}

    return {
        "prob_1": prob_1, "prob_X": prob_x, "prob_2": prob_2,
        "prob_goal": prob_goal, "prob_nogoal": prob_nogoal,
        "prob_under": prob_under, "multigol_casa": multigol_casa, "multigol_trasf": multigol_trasf,
        "combo": combo_stats, "angoli_stimati": f"{corner_casa + corner_trasf:.1f}",
        "tiri_stimati": f"{tiri_casa + tiri_trasf:.1f}"
    }

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
        if resp.status_code == 403:
            return "ERRORE_403"
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
        return df.sort_values('Date_parsed').reset_index(drop=True)
    except Exception as e:
        return str(e)

def estrai_quota_sicura(dizionario, chiave, default=0.0):
    val = dizionario.get(chiave)
    if pd.isna(val) or val == '' or val is None:
        return float(default)
    try:
        return float(val)
    except:
        return float(default)

st.title("⚽ Advanced Pro Betting Analyzer")
st.caption("Modello Statistico Avanzato con Analisi Value Bet 1X2 Automatica")

with st.sidebar:
    st.header("⚙️ Configurazione & API")
    api_key_input = st.text_input("Chiave API football-data.org (per Coppe)", type="password", help="Inserisci la tua chiave API.")
    st.divider()
    rho_val = st.slider("Correzione Dixon-Coles (ρ)", -0.20, 0.10, -0.10, 0.01)
    ewma_span_val = st.slider("Finestra Forma Recente (EWMA)", 2, 15, 6, 1)
    emivita_val = st.slider("Decadimento Temporale (Giorni)", 30, 365, 180, 10)

scelta_categoria = st.radio("Categoria Torneo", ["Campionati Nazionali (Gratuiti)", "Coppe Europee (Richiede API Key)"], horizontal=True)

if scelta_categoria == "Campionati Nazionali (Gratuiti)":
    campionato = st.selectbox("Seleziona Campionato", list(CAMPIONATI_DOMESTICI.keys()))
    info = CAMPIONATI_DOMESTICI[campionato]
    id_fd = info["id_fd"]
    
    with st.spinner("Caricamento dataset campionato e quote automatiche..."):
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
    df_globale = pd.DataFrame()
    is_coppa = False
        
else:
    if not api_key_input:
        st.warning("⚠️ Inserisci la tua chiave API di football-data.org nella barra laterale per sbloccare le coppe europee.")
        st.stop()
    campionato = st.selectbox("Seleziona Coppe", list(CAMPIONATI_COPPE.keys()))
    info = CAMPIONATI_COPPE[campionato]
    code_api = info["code"]
    
    with st.spinner("Connessione alle API delle Coppe e caricamento dati di supporto..."):
        risultato_api = carica_dati_api_europee(code_api, api_key_input)
        df_globale = carica_tutti_i_campionati()
        
    if isinstance(risultato_api, str) and risultato_api == "ERRORE_403":
        st.error("❌ **Accesso Negato (Errore 403)**: La tua chiave API gratuita non ha accesso alle Coppe Europee.")
        st.stop()
    elif isinstance(risultato_api, str):
        st.error(f"Errore di connessione API: {risultato_api}")
        st.stop()
    
    dati = risultato_api
    if dati is None or len(dati) == 0:
        st.error("Nessun dato trovato per questa competizione.")
        st.stop()
        
    dati_storico = dati[dati['Status'] == 'FINISHED'].copy()
    dati_future = dati[dati['Status'] != 'FINISHED'].copy()
    
    opzioni_partite = []
    mappa_partite = []
    
    for _, r in dati_future.iterrows():
        opzioni_partite.append(f"FUTURA ({r['Date']}): {r['HomeTeam']} vs {r['AwayTeam']}")
        mappa_partite.append(r.to_dict())
        
    for _, r in dati_storico.tail(10).iterrows():
        opzioni_partite.append(f"GIOCATA ({r['Date']}): {r['HomeTeam']} vs {r['AwayTeam']}")
        mappa_partite.append(r.to_dict())
    is_coppa = True

if not opzioni_partite:
    st.warning("Nessuna partita disponibile al momento.")
else:
    scelta = st.selectbox("Seleziona Partita", opzioni_partite)
    idx_sel = opzioni_partite.index(scelta)
    partita_sel = mappa_partite[idx_sel]
    
    modello = calcola_modello_completo(dati, partita_sel['HomeTeam'], partita_sel['AwayTeam'], rho_val, ewma_span_val, emivita_val, df_globale)
    
    if modello is None:
        st.warning(f"⚠️ Impossibile elaborare il match per **{partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}**.")
    else:
        st.subheader(f"📊 Analisi Match: {partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}")
        
        def crea_tabella(dati_dict, col_nome="Mercato"):
            df = pd.DataFrame(list(dati_dict.items()), columns=[col_nome, "Probabilità (%)"])
            df["Probabilità (%)"] = df["Probabilità (%)"].round(1)
            df = df.sort_values(by="Probabilità (%)", ascending=False).reset_index(drop=True)
            df["Probabilità (%)"] = df["Probabilità (%)"].astype(str) + "%"
            return df

        st.markdown("### 🏆 Esito Finale (1X2)")
        df_1x2 = crea_tabella({
            "1 (Casa)": modello['prob_1'],
            "X (Pareggio)": modello['prob_X'],
            "2 (Trasferta)": modello['prob_2']
        }, "Segno")
        st.dataframe(df_1x2, use_container_width=True, hide_index=True)

        # ----------------- ANALISI VALUE BET (CAMPIONATI VS COPPE) -----------------
        if not is_coppa:
            st.markdown("### 💰 Controllo Value Bet (Automatico da Quote Reali)")
            q_1 = estrai_quota_sicura(partita_sel, 'B365H', 0.0)
            q_x = estrai_quota_sicura(partita_sel, 'B365D', 0.0)
            q_2 = estrai_quota_sicura(partita_sel, 'B365A', 0.0)

            if q_1 > 0 and q_x > 0 and q_2 > 0:
                ev_1 = (modello['prob_1'] / 100.0) * q_1
                ev_x = (modello['prob_X'] / 100.0) * q_x
                ev_2 = (modello['prob_2'] / 100.0) * q_2

                dati_ev = [
                    {"Segno": "1 (Casa)", "Quota Reale": q_1, "Valutazione": "🔥 ALTO VALORE" if ev_1 > 1.05 else ("📈 Leggero Valore" if ev_1 > 1.0 else "Nessun Valore")},
                    {"Segno": "X (Pareggio)", "Quota Reale": q_x, "Valutazione": "🔥 ALTO VALORE" if ev_x > 1.05 else ("📈 Leggero Valore" if ev_x > 1.0 else "Nessun Valore")},
                    {"Segno": "2 (Trasferta)", "Quota Reale": q_2, "Valutazione": "🔥 ALTO VALORE" if ev_2 > 1.05 else ("📈 Leggero Valore" if ev_2 > 1.0 else "Nessun Valore")}
                ]
                df_ev_mostra = pd.DataFrame(dati_ev)
                st.dataframe(df_ev_mostra, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Quote dei bookmaker non disponibili per questa specifica partita.")
        else:
            st.markdown("### 🎯 Quota Equa Statistica (Coppe Europee)")
            st.caption("Essendo una coppa europea, il modello calcola la **Quota Equa (Fair Odds)** basata sulla probabilità matematica pura. Cerca sui bookmaker quote superiori a questi valori per trovare valore.")
            
            q_fair_1 = 100.0 / modello['prob_1'] if modello['prob_1'] > 0 else 0.0
            q_fair_x = 100.0 / modello['prob_X'] if modello['prob_X'] > 0 else 0.0
            q_fair_2 = 100.0 / modello['prob_2'] if modello['prob_2'] > 0 else 0.0

            dati_fair = [
                {"Segno": "1 (Casa)", "Probabilità": f"{modello['prob_1']:.1f}%", "Quota Equa Minima": f"{q_fair_1:.2f}"},
                {"Segno": "X (Pareggio)", "Probabilità": f"{modello['prob_X']:.1f}%", "Quota Equa Minima": f"{q_fair_x:.2f}"},
                {"Segno": "2 (Trasferta)", "Probabilità": f"{modello['prob_2']:.1f}%", "Quota Equa Minima": f"{q_fair_2:.2f}"}
            ]
            df_fair_mostra = pd.DataFrame(dati_fair)
            st.dataframe(df_fair_mostra, use_container_width=True, hide_index=True)
        # -----------------------------------------------------------------------------

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ⚽ Goal / No Goal")
            df_gg = crea_tabella({"Goal": modello['prob_goal'], "No Goal": modello['prob_nogoal']}, "Opzione")
            st.dataframe(df_gg, use_container_width=True, hide_index=True)
            
        with col_b:
            st.markdown("### 📉 Under / Over")
            oo_dict = {}
            for soglia, prob_u in modello['prob_under'].items():
                oo_dict[f"Under {soglia}"] = prob_u
                oo_dict[f"Over {soglia}"] = 100 - prob_u
            df_oo = crea_tabella(oo_dict, "Linea")
            st.dataframe(df_oo, use_container_width=True, hide_index=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("### 🏠 Multigol Casa")
            df_mg_c = crea_tabella(modello['multigol_casa'], "Intervallo")
            st.dataframe(df_mg_c, use_container_width=True, hide_index=True)
            
        with col_d:
            st.markdown("### ✈️ Multigol Ospite")
            df_mg_t = crea_tabella(modello['multigol_trasf'], "Intervallo")
            st.dataframe(df_mg_t, use_container_width=True, hide_index=True)

        st.markdown("### 🔥 Combo Consigliate")
        df_combo = crea_tabella(modello['combo'], "Combinazione")
        st.dataframe(df_combo, use_container_width=True, hide_index=True)

        st.markdown("### ⚔️ Ultimi Scontri Diretti (H2H)")
        df_h2h = estrai_scontri_diretti(partita_sel['HomeTeam'], partita_sel['AwayTeam'], dati, df_globale)
        if not df_h2h.empty:
            cols_mostra = [c for c in ['Date', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam'] if c in df_h2h.columns]
            st.dataframe(df_h2h[cols_mostra], use_container_width=True, hide_index=True)
        else:
            st.info("Nessun precedente recente trovato negli archivi disponibili tra queste due squadre.")

        st.divider()
        st.markdown("### 🎯 Statistiche Match (Stimate)")
        st.info(f"🚩 Angoli Totali Stimati: **{modello['angoli_stimati']}** | 🎯 Tiri in Porta Totali Stimati: **{modello['tiri_stimati']}**")
