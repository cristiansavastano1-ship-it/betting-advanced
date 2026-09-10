import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import date
from scipy.stats import poisson

st.set_page_config(page_title="COMBO Betting Pro", page_icon="⚡", layout="centered")

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

def normalizza_nome(nome):
    if not isinstance(nome, str): return ""
    return nome.lower().replace("fc", "").replace("cf", "").replace("united", "utd").replace(".", "").strip()

def estrai_partite_squadra_intelligente(squadra, df_coppa, df_globale, data_limite):
    s_norm = normalizza_nome(squadra)
    
    def filtra_df(df):
        if df is None or df.empty: return pd.DataFrame()
        sub = df[df['Date_parsed'] < data_limite].copy()
        if 'Status' in sub.columns:
            sub = sub[sub['Status'] == 'FINISHED']
        else:
            sub = sub.dropna(subset=['FTHG', 'FTAG'])
            
        return sub[
            sub['HomeTeam'].apply(lambda x: s_norm in normalizza_nome(x)) | 
            sub['AwayTeam'].apply(lambda x: s_norm in normalizza_nome(x))
        ]

    res = filtra_df(df_coppa)
    if len(res) >= 3:
        return res, "Coppa"
        
    res_glob = filtra_df(df_globale)
    if not res_glob.empty:
        return res_glob, "Globale/Domestico"
        
    return res, "Insufficiente"

def estrai_scontri_diretti(squadra_casa, squadra_trasferta, df_coppa, df_globale, data_limite):
    c_norm = normalizza_nome(squadra_casa)
    t_norm = normalizza_nome(squadra_trasferta)
    
    frames_tot = []
    if df_coppa is not None and not df_coppa.empty: frames_tot.append(df_coppa)
    if df_globale is not None and not df_globale.empty: frames_tot.append(df_globale)
    if not frames_tot: return pd.DataFrame()
        
    df_uni = pd.concat(frames_tot, ignore_index=True, sort=False)
    df_uni = df_uni[df_uni['Date_parsed'] < data_limite].dropna(subset=['FTHG', 'FTAG'])
    
    h2h = df_uni[
        (
            (df_uni['HomeTeam'].apply(lambda x: c_norm in normalizza_nome(x)) & df_uni['AwayTeam'].apply(lambda x: t_norm in normalizza_nome(x))) |
            (df_uni['HomeTeam'].apply(lambda x: t_norm in normalizza_nome(x)) & df_uni['AwayTeam'].apply(lambda x: c_norm in normalizza_nome(x)))
        )
    ].copy()
    
    return h2h.sort_values('Date_parsed', ascending=False).head(5)

def calcola_modello_completo(giocate_coppa, squadra_casa, squadra_trasferta, data_partita, rho, ewma_span, emivita, df_globale):
    data_limite = pd.to_datetime(data_partita) if data_partita else pd.Timestamp(date.today())
    
    # Filtro rigoroso No-Look-Ahead sul passato
    df_storico_valido = giocate_coppa[giocate_coppa['Date_parsed'] < data_limite].dropna(subset=['FTHG', 'FTAG']) if giocate_coppa is not None else pd.DataFrame()
    
    m_gol_casa = media_pesata_decadimento(df_storico_valido, 'FTHG', data_limite, emivita) or 1.60
    m_gol_trasf = media_pesata_decadimento(df_storico_valido, 'FTAG', data_limite, emivita) or 1.15

    forma_casa, fonte_c = estrai_partite_squadra_intelligente(squadra_casa, giocate_coppa, df_globale, data_limite)
    forma_trasf, fonte_t = estrai_partite_squadra_intelligente(squadra_trasferta, giocate_coppa, df_globale, data_limite)

    avvisi = []
    if fonte_c == "Insufficiente":
        avvisi.warning(f"⚠️ Storico limitato per **{squadra_casa}**: applicato shrinkage di lega (nessun dato inventato).")
        forma_casa = pd.DataFrame({'FTHG': [m_gol_casa], 'FTAG': [m_gol_trasf], 'Date_parsed': [data_limite]})
    if fonte_t == "Insufficiente":
        avvisi.warning(f"⚠️ Storico limitato per **{squadra_trasferta}**: applicato shrinkage di lega (nessun dato inventato).")
        forma_trasf = pd.DataFrame({'FTHG': [m_gol_trasf], 'FTAG': [m_gol_casa], 'Date_parsed': [data_limite]})

    gf_casa_rec = media_ewma(forma_casa['FTHG'], ewma_span) or m_gol_casa
    gs_casa_rec = media_ewma(forma_casa['FTAG'], ewma_span) or m_gol_trasf
    gf_trasf_rec = media_ewma(forma_trasf['FTAG'], ewma_span) or m_gol_trasf
    gs_trasf_rec = media_ewma(forma_trasf['FTHG'], ewma_span) or m_gol_casa

    tiri_casa = (media_ewma(forma_casa['HST'], ewma_span) if 'HST' in forma_casa.columns and not forma_casa['HST'].dropna().empty else 4.5)
    corner_casa = (media_ewma(forma_casa['HC'], ewma_span) if 'HC' in forma_casa.columns and not forma_casa['HC'].dropna().empty else 5.0)
    tiri_trasf = (media_ewma(forma_trasf['AST'], ewma_span) if 'AST' in forma_trasf.columns and not forma_trasf['AST'].dropna().empty else 4.0)
    corner_trasf = (media_ewma(forma_trasf['AC'], ewma_span) if 'AC' in forma_trasf.columns and not forma_trasf['AC'].dropna().empty else 4.5)

    attacco_casa = gf_casa_rec / max(0.1, m_gol_casa)
    difesa_trasf = gs_trasf_rec / max(0.1, m_gol_trasf)
    attacco_trasf = gf_trasf_rec / max(0.1, m_gol_trasf)
    difesa_casa = gs_casa_rec / max(0.1, m_gol_casa)

    lam_c = max(0.1, attacco_casa * difesa_trasf * m_gol_casa)
    lam_t = max(0.1, attacco_trasf * defesa_casa * m_gol_trasf if 'defesa_casa' in locals() else attacco_trasf * difesa_casa * m_gol_trasf)

    prob_1, prob_x, prob_2 = 0.0, 0.0, 0.0
    prob_goal, prob_nogoal = 0.0, 0.0
    limiti_under = [1.5, 2.5, 3.5]
    prob_under = {l: 0.0 for l in limiti_under}
    
    multigol_casa = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    multigol_trasf = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    
    # Motore Combo Avanzato flessibile
    combo_stats = {}

    tot_p = 0.0
    for gc in range(8):
        for gt in range(8):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            tot_p += p
            segno = 'X' if gc == gt else ('1' if gc > gt else '2')
            
            if segno == '1': prob_1 += p
            elif segno == 'X': prob_x += p
            else: prob_2 += p
            
            is_goal = (gc > 0 and gt > 0)
            if is_goal: prob_goal += p
            else: prob_nogoal += p
            
            tot_gol = gc + gt
            for l in limiti_under:
                if tot_gol < l: prob_under[l] += p
                
            for mg_key, (mi_c, ma_c) in [("0-1", (0,1)), ("0-2", (0,2)), ("1-2", (1,2)), ("1-3", (1,3)), ("2-3", (2,3)), ("2-4", (2,4))]:
                if mi_c <= gc <= ma_c: multigol_casa[mg_key] += p
                if mi_c <= gt <= ma_c: multigol_trasf[mg_key] += p

            # Generazione dinamica Combo avanzate richieste
            # 1 + Over 2.5 + Goal
            if segno == '1' and tot_gol > 2.5 and is_goal:
                combo_stats["1 + Over 2.5 + Goal"] = combo_stats.get("1 + Over 2.5 + Goal", 0.0) + p
            # 1 + Under 2.5 + No Goal
            if segno == '1' and tot_gol < 2.5 and not is_goal:
                combo_stats["1 + Under 2.5 + No Goal"] = combo_stats.get("1 + Under 2.5 + No Goal", 0.0) + p
            # 1 + Over 2.5
            if segno == '1' and tot_gol > 2.5:
                combo_stats["1 + Over 2.5"] = combo_stats.get("1 + Over 2.5", 0.0) + p
            # 1 + Goal
            if segno == '1' and is_goal:
                combo_stats["1 + Goal"] = combo_stats.get("1 + Goal", 0.0) + p
            # X + Under 2.5
            if segno == 'X' and tot_gol < 2.5:
                combo_stats["X + Under 2.5"] = combo_stats.get("X + Under 2.5", 0.0) + p
            # 2 + Goal
            if segno == '2' and is_goal:
                combo_stats["2 + Goal"] = combo_stats.get("2 + Goal", 0.0) + p

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
        if resp.status_code == 403: return "ERRORE_403"
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for m in data.get("matches", []):
            home = m['homeTeam']['name']
            away = m['awayTeam']['name']
            date_str = m['utcDate'][:10]
            status = m['status']
            fthg, ftag = None, None
            if status == 'FINISHED':
                fthg = m['score']['fullTime']['home']
                ftag = m['score']['fullTime']['away']
            rows.append({'Date': date_str, 'HomeTeam': home, 'AwayTeam': away, 'FTHG': fthg, 'FTAG': ftag, 'Status': status})
        df = pd.DataFrame(rows)
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.sort_values('Date_parsed').reset_index(drop=True)
    except Exception as e:
        return str(e)

def estrai_quota_sicura(dizionario, chiave, default=0.0):
    val = dizionario.get(chiave)
    if pd.isna(val) or val == '' or val is None: return float(default)
    try: return float(val)
    except: return float(default)

st.title("⚡ COMBO Betting Pro")
st.caption("Modello Statistico Avanzato No-Look-Ahead con Motore Combo & Stime Tiri/Angoli")

with st.sidebar:
    st.header("⚙️ Configurazione")
    api_key_input = st.text_input("Chiave API football-data.org (Coppe)", type="password")
    st.divider()
    rho_val = st.slider("Correzione Dixon-Coles (ρ)", -0.20, 0.10, -0.10, 0.01)
    ewma_span_val = st.slider("Finestra Forma (EWMA)", 2, 15, 6, 1)
    emivita_val = st.slider("Decadimento Temporale (Giorni)", 30, 365, 180, 10)

scelta_categoria = st.radio("Categoria Torneo", ["Campionati Nazionali", "Coppe Europee"], horizontal=True)

if scelta_categoria == "Campionati Nazionali":
    campionato = st.selectbox("Seleziona Campionato", list(CAMPIONATI_DOMESTICI.keys()))
    info = CAMPIONATI_DOMESTICI[campionato]
    id_fd = info["id_fd"]
    
    with st.spinner("Caricamento dataset..."):
        dati = carica_dati_campionato(id_fd)
        fixture_future = carica_fixture_future(id_fd)
        
    if dati is None or len(dati) == 0:
        st.error("Errore di caricamento dati.")
        st.stop()
        
    opzioni_partite, mappa_partite = [], []
    if not fixture_future.empty:
        for _, r in fixture_future.iterrows():
            opzioni_partite.append(f"FUTURA ({r.get('Date','?')}): {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}")
            mappa_partite.append(r.to_dict())
    for _, r in dati[dati['FTHG'].notna()].tail(15).iterrows():
        opzioni_partite.append(f"RECENTE ({r.get('Date','?')}): {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}")
        mappa_partite.append(r.to_dict())
    df_globale = pd.DataFrame()
    is_coppa = False
else:
    if not api_key_input:
        st.warning("⚠️ Inserisci la chiave API per le coppe.")
        st.stop()
    campionato = st.selectbox("Seleziona Coppe", list(CAMPIONATI_COPPE.keys()))
    info = CAMPIONATI_COPPE[campionato]
    
    with st.spinner("Connessione API Coppe..."):
        risultato_api = carica_dati_api_europee(info["code"], api_key_input)
        df_globale = carica_tutti_i_campionati()
        
    if isinstance(risultato_api, str):
        st.error(f"Errore API: {risultato_api}")
        st.stop()
        
    dati = risultato_api
    opzioni_partite, mappa_partite = [], []
    for _, r in dati[dati['Status'] != 'FINISHED'].iterrows():
        opzioni_partite.append(f"FUTURA ({r['Date']}): {r['HomeTeam']} vs {r['AwayTeam']}")
        mappa_partite.append(r.to_dict())
    for _, r in dati[dati['Status'] == 'FINISHED'].tail(10).iterrows():
        opzioni_partite.append(f"GIOCATA ({r['Date']}): {r['HomeTeam']} vs {r['AwayTeam']}")
        mappa_partite.append(r.to_dict())
    is_coppa = True

if not opzioni_partite:
    st.warning("Nessuna partita disponibile.")
else:
    scelta = st.selectbox("Seleziona Partita", opzioni_partite)
    partita_sel = mappa_partite[opzioni_partite.index(scelta)]
    
    modello = calcola_modello_completo(dati, partita_sel['HomeTeam'], partita_sel['AwayTeam'], partita_sel.get('Date'), rho_val, ewma_span_val, emivita_val, df_globale)
    
    if modello:
        st.subheader(f"📊 Analisi COMBO: {partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}")
        
        def crea_tabella(dati_dict, col_nome="Mercato"):
            if not dati_dict: return pd.DataFrame(columns=[col_nome, "Probabilità (%)"])
            df = pd.DataFrame(list(dati_dict.items()), columns=[col_nome, "Probabilità (%)"])
            df["Probabilità (%)"] = df["Probabilità (%)"].round(1)
            df = df.sort_values(by="Probabilità (%)", ascending=False).reset_index(drop=True)
            df["Probabilità (%)"] = df["Probabilità (%)"].astype(str) + "%"
            return df

        st.markdown("### 🏆 Esito Finale (1X2)")
        st.dataframe(crea_tabella({"1 (Casa)": modello['prob_1'], "X (Pareggio)": modello['prob_X'], "2 (Trasferta)": modello['prob_2']}, "Segno"), use_container_width=True, hide_index=True)

        # Controllo Value Bet con normalizzazione Overround
        if not is_coppa:
            st.markdown("### 💰 Controllo Value Bet (Normalizzato)")
            q_1, q_x, q_2 = estrai_quota_sicura(partita_sel, 'B365H'), estrai_quota_sicura(partita_sel, 'B365D'), estrai_quota_sicura(partita_sel, 'B365A')
            if q_1 > 0 and q_x > 0 and q_2 > 0:
                implied_sum = (1/q_1) + (1/q_x) + (1/q_2) # Overround
                ev_1 = (modello['prob_1'] / 100.0) * (q_1 * implied_sum)
                ev_x = (modello['prob_X'] / 100.0) * (q_x * implied_sum)
                ev_2 = (modello['prob_2'] / 100.0) * (q_2 * implied_sum)
                dati_ev = [
                    {"Segno": "1", "Quota": q_1, "Valutazione": "🔥 OTTIMO VALORE" if ev_1 > 1.05 else "Nessun Valore"},
                    {"Segno": "X", "Quota": q_x, "Valutazione": "🔥 OTTIMO VALORE" if ev_x > 1.05 else "Nessun Valore"},
                    {"Segno": "2", "Quota": q_2, "Valutazione": "🔥 OTTIMO VALORE" if ev_2 > 1.05 else "Nessun Valore"}
                ]
                st.dataframe(pd.DataFrame(dati_ev), use_container_width=True, hide_index=True)
            else:
                st.info("Quote non disponibili.")
        else:
            st.markdown("### 🎯 Quote Eque Statistiche (Coppe)")
            df_fair = pd.DataFrame([
                {"Segno": "1", "Prob": f"{modello['prob_1']:.1f}%", "Quota Equa Minima": f"{100/modello['prob_1']:.2f}" if modello['prob_1']>0 else "N/A"},
                {"Segno": "X", "Prob": f"{modello['prob_X']:.1f}%", "Quota Equa Minima": f"{100/modello['prob_X']:.2f}" if modello['prob_X']>0 else "N/A"},
                {"Segno": "2", "Prob": f"{modello['prob_2']:.1f}%", "Quota Equa Minima": f"{100/modello['prob_2']:.2f}" if modello['prob_2']>0 else "N/A"}
            ])
            st.dataframe(df_fair, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ⚽ Goal / No Goal")
            st.dataframe(crea_tabella({"Goal": modello['prob_goal'], "No Goal": modello['prob_nogoal']}, "Opzione"), use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("### 📉 Under / Over")
            oo_dict = {f"Under {s}": p for s, p in modello['prob_under'].items()}
            oo_dict.update({f"Over {s}": 100 - p for s, p in modello['prob_under'].items()})
            st.dataframe(crea_tabella(oo_dict, "Linea"), use_container_width=True, hide_index=True)

        st.markdown("### ⚡ COMBO Avanzate Consigliate")
        st.dataframe(crea_tabella(modello['combo'], "Combinazione"), use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Statistiche Match (Tiri & Angoli)")
        st.info(f"🚩 Angoli Totali Stimati: **{modello['angoli_stimati']}** | 🎯 Tiri in Porta Totali Stimati: **{modello['tiri_stimati']}**")

        st.markdown("### ⚔️ Ultimi Scontri Diretti (H2H)")
        df_h2h = estrai_scontri_diretti(partita_sel['HomeTeam'], partita_sel['AwayTeam'], dati, df_globale, pd.to_datetime(partita_sel.get('Date')))
        if not df_h2h.empty:
            st.dataframe(df_h2h[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam']], use_container_width=True, hide_index=True)
        else:
            st.info("Nessun precedente trovato prima di questa data.")
