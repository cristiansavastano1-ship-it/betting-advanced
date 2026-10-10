import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import time
import os
import pickle
from datetime import datetime, date
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="COMBO — V15.2 Operativo Prossime Partite", page_icon="⚽", layout="centered")

AUDIT_SCORE_MAX = 12
EPS_PROB = 1e-9

CAMPIONATI_DOMESTICI = {
    "Italia - Serie A": {"id_fd": "I1"},
    "Inghilterra - Premier League": {"id_fd": "E0"},
    "Spagna - La Liga": {"id_fd": "SP1"},
    "Germania - Bundesliga": {"id_fd": "D1"},
    "Francia - Ligue 1": {"id_fd": "F1"},
}

CAMPIONATI_COPPE = {
    "🌍 UEFA Champions League": {"code": "CL", "gratis_confermato": True},
    "🌍 UEFA Europa League": {"code": "EL", "gratis_confermato": False},
    "🌍 UEFA Conference League": {"code": "UECL", "gratis_confermato": False},
}

HEADERS_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

K_SHRINKAGE = 10  # stesso principio già validato nell'App Risultati Fissi


# =====================================================================
# 🔧 FIX #4 — MATCHING SQUADRE "MORBIDO"
# Prima: confronto testuale esatto. Con le coppe europee (nomi da
# football-data.org, es. "Manchester United FC") contro i nomi domestici
# (es. "Man United") il match esatto falliva spesso in silenzio, facendo
# scattare il generatore di dati finti (ora rimosso, vedi FIX #1).
# =====================================================================
def normalizza_nome_squadra(nome):
    nome = str(nome)
    for suffisso in [" FC", " CF", " AFC", " AC", " SC", " CFC"]:
        nome = nome.replace(suffisso, "")
    return nome.strip().lower()


def nomi_corrispondono(a, b):
    na, nb = normalizza_nome_squadra(a), normalizza_nome_squadra(b)
    return na == nb or na in nb or nb in na


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
    if colonna not in df.columns: return None
    sub = df[[colonna, 'Date_parsed']].dropna()
    if len(sub) == 0: return None
    giorni = (data_riferimento - sub['Date_parsed']).dt.days.clip(lower=0)
    pesi = 0.5 ** (giorni / emivita)
    tot = pesi.sum()
    return sub[colonna].mean() if tot <= 0 else (sub[colonna] * pesi).sum() / tot


# =====================================================================
# 🔧 Download robusto: header da browser + retry + fallback www/no-www
# (stessa correzione già validata nell'App Risultati Fissi, dopo il
# disservizio di football-data.co.uk causato dal blocco geografico su
# "www" per il traffico non-UK)
# =====================================================================
def scarica_csv_robusto(url, tentativi=3, attesa_secondi=2):
    """FIX #11 — BOM (Byte Order Mark): fixtures.csv inizia con un carattere
    invisibile Unicode che, se non gestito, si attacca al nome della prima
    colonna ("Div" diventa "\\ufeffDiv"), facendo fallire in silenzio ogni
    controllo tipo "if 'Div' in colonne" — nessuna fixture futura veniva mai
    trovata, su nessun campionato, per questo motivo esatto. 'utf-8-sig' lo
    rimuove in fase di decodifica; non fa danno se il file non ha il BOM."""
    varianti_url = [url]
    if "://www." in url:
        varianti_url.append(url.replace("://www.", "://"))
    elif "://" in url:
        varianti_url.append(url.replace("://", "://www."))

    ultimo_errore = None
    for tentativo in range(tentativi):
        for url_prova in varianti_url:
            try:
                resp = requests.get(url_prova, headers=HEADERS_BROWSER, timeout=15)
                resp.raise_for_status()
                testo = resp.content.decode('utf-8-sig', errors='replace')
                df = pd.read_csv(io.StringIO(testo))
                df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]  # rete di sicurezza extra
                return df, None
            except Exception as e:
                ultimo_errore = str(e)
        if tentativo < tentativi - 1:
            time.sleep(attesa_secondi)
    return None, ultimo_errore


@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_campionato(id_fd):
    """FIX B — aggiunto il tag 'Stagione' (precedente/corrente), necessario
    per poter validare il value bet SOLO sui risultati reali più recenti,
    non su quelli già usati per calibrare le medie di lega."""
    codice_corrente, codice_precedente = codici_stagione()
    frames = []
    for codice, label in [(codice_precedente, 'precedente'), (codice_corrente, 'corrente')]:
        url = f"https://football-data.co.uk/mmz4281/{codice}/{id_fd}.csv"
        df, _ = scarica_csv_robusto(url)
        if df is not None:
            df.columns = df.columns.str.strip()
            df['Stagione'] = label
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
    """Matching morbido (FIX #4): prima prova nel dataset della competizione
    stessa, poi nel dataset globale domestico come fallback."""
    if df_coppa is not None and not df_coppa.empty:
        maschera = df_coppa['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra)) | \
                   df_coppa['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
        f_coppa = df_coppa[maschera]
        if len(f_coppa) > 0:
            return f_coppa

    if df_globale is not None and not df_globale.empty:
        maschera = df_globale['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra)) | \
                   df_globale['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
        f_glob = df_globale[maschera]
        if len(f_glob) > 0:
            return f_glob

    return pd.DataFrame()


# =====================================================================
# 🔧 FIX CRITICO #12 — ATTRIBUZIONE CORRETTA DI GOL/TIRI/CORNER
# BUG PRECEDENTE: le partite di una squadra venivano raccolte sia in casa
# sia in trasferta, ma poi il codice leggeva sempre la colonna "di casa"
# (FTHG) come "gol fatti". Per le partite giocate in TRASFERTA, FTHG sono
# i gol dell'AVVERSARIO — quindi circa metà dei dati di attacco erano in
# realtà dati di difesa e viceversa. Effetto: una squadra che segna 3 e
# subisce 0 risultava 1.5/1.5, cioè perfettamente nella media — il modello
# perdeva quasi del tutto la capacità di distinguere squadre forti e deboli,
# e finiva sotto la semplice baseline "vince sempre la squadra di casa".
# Qui i valori vengono attribuiti guardando, partita per partita, se la
# squadra giocava in casa o fuori.
# =====================================================================
def serie_squadra(df, squadra, tipo):
    """tipo: 'gol_fatti', 'gol_subiti', 'tiri_fatti', 'corner_fatti'.
    Ritorna una Series con i valori attribuiti correttamente alla squadra,
    indipendentemente dal fatto che giocasse in casa o in trasferta."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    colonne_casa = {'gol_fatti': 'FTHG', 'gol_subiti': 'FTAG', 'tiri_fatti': 'HST', 'corner_fatti': 'HC'}
    colonne_trasf = {'gol_fatti': 'FTAG', 'gol_subiti': 'FTHG', 'tiri_fatti': 'AST', 'corner_fatti': 'AC'}
    col_c, col_t = colonne_casa[tipo], colonne_trasf[tipo]

    if col_c not in df.columns or col_t not in df.columns:
        return pd.Series(dtype=float)

    gioca_in_casa = df['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra))
    valori = df[col_c].where(gioca_in_casa, df[col_t])
    return valori.dropna()


def estrai_scontri_diretti(squadra_casa, squadra_trasferta, df_coppa, df_globale):
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

    maschera_diretto = df_uni['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra_casa)) & \
                        df_uni['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra_trasferta))
    maschera_inverso = df_uni['HomeTeam'].apply(lambda x: nomi_corrispondono(x, squadra_trasferta)) & \
                        df_uni['AwayTeam'].apply(lambda x: nomi_corrispondono(x, squadra_casa))

    h2h = df_uni[(df_uni['FTHG'].notna()) & (df_uni['FTAG'].notna()) & (maschera_diretto | maschera_inverso)].copy()

    if 'Date_parsed' in h2h.columns:
        h2h = h2h.sort_values('Date_parsed', ascending=False)
    return h2h.head(5)


# =====================================================================
# 🔧 FIX #1 — VIA IL GENERATORE DI DATI FINTI, DENTRO VERO SHRINKAGE
# Prima: se una squadra non aveva partite trovate, il codice INVENTAVA un
# profilo attacco/difesa calcolato dalla somma dei codici ASCII del nome
# squadra — un numero pseudo-casuale spacciato per statistica.
# Ora: quando i dati specifici sono pochi o assenti, la stima converge
# gradualmente verso la media di lega (shrinkage, stesso principio già
# validato nell'App Risultati Fissi) invece di inventare un profilo finto.
# Il chiamante riceve n_casa/n_trasf per mostrare un avviso onesto quando
# il campione specifico è scarso.
# =====================================================================
def calcola_modello_completo(giocate_coppa, squadra_casa, squadra_trasferta, rho, ewma_span,
                              emivita, df_globale, data_riferimento=None):
    giocate_validi = giocate_coppa.dropna(subset=['FTHG', 'FTAG']) if giocate_coppa is not None else pd.DataFrame()
    if data_riferimento is None:
        data_riferimento = giocate_validi['Date_parsed'].max() if not giocate_validi.empty else pd.Timestamp(date.today())

    # Baseline di lega/competizione. Se il campione della competizione è
    # troppo piccolo (tipico per le coppe a inizio stagione), lo arricchiamo
    # con il dataset domestico globale per una stima più stabile.
    base_per_media = giocate_validi
    if len(giocate_validi) < 20 and df_globale is not None and not df_globale.empty:
        base_per_media = pd.concat([giocate_validi, df_globale.dropna(subset=['FTHG', 'FTAG'])], ignore_index=True, sort=False)

    m_gol_casa = media_pesata_decadimento(base_per_media, 'FTHG', data_riferimento, emivita) or 1.65
    m_gol_trasf = media_pesata_decadimento(base_per_media, 'FTAG', data_riferimento, emivita) or 1.25
    m_tiri_casa_lega = media_pesata_decadimento(base_per_media, 'HST', data_riferimento, emivita) or 4.8
    m_tiri_trasf_lega = media_pesata_decadimento(base_per_media, 'AST', data_riferimento, emivita) or 4.1
    m_corner_casa_lega = media_pesata_decadimento(base_per_media, 'HC', data_riferimento, emivita) or 5.4
    m_corner_trasf_lega = media_pesata_decadimento(base_per_media, 'AC', data_riferimento, emivita) or 4.6

    forma_casa = estrai_partite_squadra_intelligente(squadra_casa, giocate_coppa, df_globale)
    forma_trasf = estrai_partite_squadra_intelligente(squadra_trasferta, giocate_coppa, df_globale)
    n_casa, n_trasf = len(forma_casa), len(forma_trasf)

    # FIX CRITICO #12 — i valori vengono attribuiti alla squadra giusta
    # guardando, partita per partita, se giocava in casa o in trasferta.
    gf_casa_rec = media_ewma(serie_squadra(forma_casa, squadra_casa, 'gol_fatti'), ewma_span) if n_casa else None
    gs_casa_rec = media_ewma(serie_squadra(forma_casa, squadra_casa, 'gol_subiti'), ewma_span) if n_casa else None
    gf_trasf_rec = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'gol_fatti'), ewma_span) if n_trasf else None
    gs_trasf_rec = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'gol_subiti'), ewma_span) if n_trasf else None

    tiri_casa = media_ewma(serie_squadra(forma_casa, squadra_casa, 'tiri_fatti'), ewma_span) if n_casa else None
    corner_casa = media_ewma(serie_squadra(forma_casa, squadra_casa, 'corner_fatti'), ewma_span) if n_casa else None
    tiri_trasf = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'tiri_fatti'), ewma_span) if n_trasf else None
    corner_trasf = media_ewma(serie_squadra(forma_trasf, squadra_trasferta, 'corner_fatti'), ewma_span) if n_trasf else None

    # Shrinkage: peso -> 0 quando n_casa/n_trasf sono pochi o zero, quindi la
    # stima converge verso il rapporto neutro 1.0 (= "come la media") invece
    # di un profilo inventato. Peso -> 1 quando il campione è ampio, quindi ci
    # si fida del dato specifico della squadra.
    peso_casa = n_casa / (n_casa + K_SHRINKAGE)
    peso_trasf = n_trasf / (n_trasf + K_SHRINKAGE)

    # FIX CRITICO #13 — riferimento corretto per i rapporti attacco/difesa.
    # I dati di una squadra mescolano partite in casa e in trasferta, quindi
    # vanno confrontati con la media di lega COMPLESSIVA (casa+trasferta), non
    # con quella specifica di casa o di trasferta. Prima si confrontava il dato
    # misto della squadra di casa con la media "solo casa" (più alta) e quello
    # della squadra ospite con la media "solo trasferta" (più bassa): risultato,
    # le squadre di casa risultavano sistematicamente sottovalutate e le ospiti
    # sopravvalutate — per questo il modello prevedeva più vittorie in trasferta
    # che in casa, il contrario di come funziona davvero il calcio.
    # Il vantaggio del fattore campo resta comunque applicato, più sotto, dal
    # fatto che lambda_casa usa m_gol_casa e lambda_trasferta usa m_gol_trasf.
    m_gol_complessiva = (m_gol_casa + m_gol_trasf) / 2.0

    rapp_attacco_casa = (gf_casa_rec / max(0.1, m_gol_complessiva)) if gf_casa_rec is not None else 1.0
    rapp_difesa_casa = (gs_casa_rec / max(0.1, m_gol_complessiva)) if gs_casa_rec is not None else 1.0
    rapp_attacco_trasf = (gf_trasf_rec / max(0.1, m_gol_complessiva)) if gf_trasf_rec is not None else 1.0
    rapp_difesa_trasf = (gs_trasf_rec / max(0.1, m_gol_complessiva)) if gs_trasf_rec is not None else 1.0

    attacco_casa = peso_casa * rapp_attacco_casa + (1 - peso_casa) * 1.0
    difesa_casa = peso_casa * rapp_difesa_casa + (1 - peso_casa) * 1.0
    attacco_trasf = peso_trasf * rapp_attacco_trasf + (1 - peso_trasf) * 1.0
    difesa_trasf = peso_trasf * rapp_difesa_trasf + (1 - peso_trasf) * 1.0

    # Stessa logica del FIX #13 anche per tiri e angoli: i dati della squadra
    # sono misti casa+trasferta, quindi il riferimento verso cui convergono
    # (shrinkage) dev'essere la media complessiva, non quella specifica.
    m_tiri_complessiva = (m_tiri_casa_lega + m_tiri_trasf_lega) / 2.0
    m_corner_complessiva = (m_corner_casa_lega + m_corner_trasf_lega) / 2.0

    tiri_casa_finale = peso_casa * tiri_casa + (1 - peso_casa) * m_tiri_complessiva if tiri_casa is not None else m_tiri_casa_lega
    corner_casa_finale = peso_casa * corner_casa + (1 - peso_casa) * m_corner_complessiva if corner_casa is not None else m_corner_casa_lega
    tiri_trasf_finale = peso_trasf * tiri_trasf + (1 - peso_trasf) * m_tiri_complessiva if tiri_trasf is not None else m_tiri_trasf_lega
    corner_trasf_finale = peso_trasf * corner_trasf + (1 - peso_trasf) * m_corner_complessiva if corner_trasf is not None else m_corner_trasf_lega

    lam_c = max(0.2, attacco_casa * difesa_trasf * m_gol_casa)
    lam_t = max(0.2, attacco_trasf * difesa_casa * m_gol_trasf)

    prob_1, prob_x, prob_2 = 0.0, 0.0, 0.0
    prob_goal, prob_nogoal = 0.0, 0.0
    limiti_under = [1.5, 2.5, 3.5]
    prob_under = {l: 0.0 for l in limiti_under}
    multigol_casa = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    multigol_trasf = {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0, "1-3": 0.0, "2-3": 0.0, "2-4": 0.0}
    combo_stats = {"1 + Goal": 0.0, "1 + Over 2.5": 0.0, "X + Under 2.5": 0.0, "2 + Goal": 0.0}
    griglia_risultati = []  # FIX #5 — griglia completa per il motore combo libero

    tot_p = 0.0
    for gc in range(AUDIT_SCORE_MAX + 1):
        for gt in range(AUDIT_SCORE_MAX + 1):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            tot_p += p
            segno = 'X' if gc == gt else ('1' if gc > gt else '2')
            griglia_risultati.append({"gc": gc, "gt": gt, "p": p, "segno": segno})

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
        for r in griglia_risultati:
            r["p"] *= f

    return {
        "prob_1": prob_1, "prob_X": prob_x, "prob_2": prob_2,
        "prob_goal": prob_goal, "prob_nogoal": prob_nogoal,
        "prob_under": prob_under, "multigol_casa": multigol_casa, "multigol_trasf": multigol_trasf,
        "combo": combo_stats, "griglia": griglia_risultati,
        "angoli_stimati": f"{corner_casa_finale + corner_trasf_finale:.1f}",
        "tiri_stimati": f"{tiri_casa_finale + tiri_trasf_finale:.1f}",
        "n_casa": n_casa, "n_trasf": n_trasf,
    }


# =====================================================================
# 🔧 FIX #5 — MOTORE COMBO LIBERO
# Calcola la probabilità congiunta di qualunque combinazione di segno +
# soglia gol (Over/Under) + Gol/No Gol, leggendo direttamente dalla griglia
# di risultati già calcolata — corretto per costruzione (somma le celle
# della griglia Poisson che soddisfano TUTTE le condizioni insieme, non
# moltiplica probabilità come se fossero eventi indipendenti, cosa che
# gol/segno NON sono).
# =====================================================================
def calcola_combo_libera(griglia, segno=None, soglia_gol=None, tipo_soglia=None, gol_nogol=None):
    tot = 0.0
    for r in griglia:
        gc, gt, p = r["gc"], r["gt"], r["p"]
        ok = True
        if segno and r["segno"] != segno:
            ok = False
        if ok and soglia_gol is not None and tipo_soglia:
            tot_g = gc + gt
            if tipo_soglia == "Over" and not (tot_g > soglia_gol): ok = False
            if tipo_soglia == "Under" and not (tot_g < soglia_gol): ok = False
        if ok and gol_nogol:
            entrambe_segnano = gc > 0 and gt > 0
            if gol_nogol == "Goal" and not entrambe_segnano: ok = False
            if gol_nogol == "NoGoal" and entrambe_segnano: ok = False
        if ok:
            tot += p
    return tot


@st.cache_data(ttl=1800, show_spinner=False)
def carica_fixture_future(id_fd):
    """Scarica e normalizza le sole fixture future del campionato richiesto.

    Fonte gratuita football-data.co.uk. Se la fonte non è raggiungibile,
    restituisce un DataFrame vuoto: non genera partite o risultati artificiali.
    """
    df, errore = scarica_csv_robusto("https://football-data.co.uk/fixtures.csv")
    if df is None or df.empty:
        return pd.DataFrame()

    fx = df.copy()
    fx.columns = [str(c).replace('\ufeff', '').strip() for c in fx.columns]
    colonne_richieste = {'Div', 'Date', 'HomeTeam', 'AwayTeam'}
    if not colonne_richieste.issubset(set(fx.columns)):
        return pd.DataFrame()

    fx['Div'] = fx['Div'].astype(str).str.strip()
    fx = fx[fx['Div'].eq(str(id_fd))].copy()
    if fx.empty:
        return pd.DataFrame()

    fx['Date_parsed'] = pd.to_datetime(fx['Date'], errors='coerce', dayfirst=True)
    fx = fx.dropna(subset=['Date_parsed', 'HomeTeam', 'AwayTeam'])
    fx = fx[(fx['HomeTeam'].astype(str).str.strip() != '') &
            (fx['AwayTeam'].astype(str).str.strip() != '')]
    oggi = pd.Timestamp(date.today()).normalize()
    fx = fx[fx['Date_parsed'].dt.normalize() >= oggi]
    return fx.sort_values(['Date_parsed', 'HomeTeam']).drop_duplicates(
        subset=['Date_parsed', 'HomeTeam', 'AwayTeam']
    ).reset_index(drop=True)


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
            status = m['status']
            fthg, ftag = None, None
            if status == 'FINISHED':
                fthg = m['score']['fullTime']['home']
                ftag = m['score']['fullTime']['away']
            rows.append({
                'Date': m['utcDate'][:10], 'HomeTeam': m['homeTeam']['name'], 'AwayTeam': m['awayTeam']['name'],
                'FTHG': fthg, 'FTAG': ftag, 'Status': status,
            })
        df = pd.DataFrame(rows)
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.sort_values('Date_parsed').reset_index(drop=True)
    except Exception as e:
        return str(e)


# =====================================================================
# 🔧 FIX #3 — VALUE BET CORRETTO (overround + media multi-bookmaker)
# Prima: EV = probabilità_modello × quota_grezza di UN SOLO bookmaker
# (Bet365), senza depurare la quota dal margine del bookmaker — lo stesso
# bug dell'overround già corretto nell'App Risultati Fissi. Ora: media di
# tutti i bookmaker disponibili nel file, quota "equa" depurata dal margine.
# =====================================================================
def classifica_colonne_quote(colonne):
    apertura_h = [c for c in colonne if c.endswith('H') and not c.endswith('CH') and c not in ['FTHG', 'HTHG', 'PTHG']]
    apertura_d = [c for c in colonne if c.endswith('D') and not c.endswith('CD') and c not in ['FTHG', 'FTAG', 'HTHG', 'HTAG']]
    apertura_a = [c for c in colonne if c.endswith('A') and not c.endswith('CA') and c not in ['FTAG', 'HTAG', 'PTAG']]
    return apertura_h, apertura_d, apertura_a


def quote_mercato_normalizzate(riga, colonne_h, colonne_d, colonne_a):
    vh = [riga[c] for c in colonne_h if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    vd = [riga[c] for c in colonne_d if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    va = [riga[c] for c in colonne_a if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    if not (vh and vd and va): return None
    qh, qd, qa = sum(vh)/len(vh), sum(vd)/len(vd), sum(va)/len(va)
    pih, pid, pia = 1/qh, 1/qd, 1/qa
    over = pih + pid + pia
    return {"q_casa_equa": over/pih, "q_x_equa": over/pid, "q_trasf_equa": over/pia,
            "overround": over, "n_bookmakers": len(vh)}


# =====================================================================
# 🔧 PUNTO C — STIMA APPROSSIMATA DELLA QUOTA COMBO
# Nessun bookmaker pubblica una quota per combo libere tipo "1 + Over 2.5 +
# Goal" nei file gratuiti — solo per i mercati singoli (1X2, Over/Under
# 2.5). Qui stimiamo una quota "come se" i mercati fossero indipendenti
# (moltiplicando le quote eque dei singoli mercati disponibili) — è
# un'APPROSSIMAZIONE, non un dato di mercato reale: segno e gol totali
# nella stessa partita sono in una certa misura correlati, quindi il numero
# vero si discosterà da questo. Il componente Gol/No Gol non ha una quota
# disponibile nel file, quindi non entra nella stima — etichettato chiaramente.
# =====================================================================
def classifica_colonne_over_under(colonne, soglia="2.5"):
    over_cols = [c for c in colonne if c.endswith(f'>{soglia}')]
    under_cols = [c for c in colonne if c.endswith(f'<{soglia}')]
    return over_cols, under_cols


def quote_over_under_normalizzate(riga, colonne_over, colonne_under):
    v_over = [riga[c] for c in colonne_over if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    v_under = [riga[c] for c in colonne_under if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    if not (v_over and v_under): return None
    q_over, q_under = sum(v_over)/len(v_over), sum(v_under)/len(v_under)
    pi_over, pi_under = 1/q_over, 1/q_under
    overround = pi_over + pi_under
    return {"q_over_equa": overround/pi_over, "q_under_equa": overround/pi_under, "n_bookmakers": len(v_over)}


def stima_quota_combo_approssimata(segno, soglia_gol, tipo_soglia, quote_1x2, quote_ou_25):
    """Ritorna (quota_stimata_o_None, lista_componenti_non_prezzate)."""
    fattori = []
    non_prezzate = []
    if segno:
        if quote_1x2:
            mappa = {"1": quote_1x2["q_casa_equa"], "X": quote_1x2["q_x_equa"], "2": quote_1x2["q_trasf_equa"]}
            fattori.append(mappa[segno])
        else:
            non_prezzate.append(f"segno {segno}")
    if soglia_gol is not None and tipo_soglia:
        if soglia_gol == 2.5 and quote_ou_25:
            fattori.append(quote_ou_25["q_over_equa"] if tipo_soglia == "Over" else quote_ou_25["q_under_equa"])
        else:
            non_prezzate.append(f"{tipo_soglia} {soglia_gol}")
    quota = None
    if fattori:
        quota = 1.0
        for f in fattori:
            quota *= f
    return quota, non_prezzate


# =====================================================================
# 🔧 BACKTEST SENZA FILTRO — accuratezza pura del modello
# Diverso dal value bet (che filtra per EV/soglia): qui valutiamo semplicemente
# "quante volte la previsione principale del modello (il segno più probabile)
# ha indovinato il risultato vero?", su TUTTE le partite disponibili, senza
# nessun filtro — la metrica più diretta e senza sorprese sulla bontà di base
# del modello, utile per mandarmi i numeri e controllarli insieme.
# =====================================================================
def esegui_backtest_senza_filtro(dati_completi, rho, ewma_span, emivita, usa_oos, id_fd=None,
                                  usa_calibrazione=False, richiedi_accordo_mercato=False):
    """richiedi_accordo_mercato (IDEA #1): valuta la previsione principale SOLO
    sulle partite dove il modello è d'accordo col favorito del mercato (stesso
    segno con la quota più bassa) — un filtro di fiducia, non una correzione
    della stima come il blending già scartato. Riduce il campione ma, se
    l'idea è valida, dovrebbe alzare l'accuratezza sul sottoinsieme rimasto."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    ha_stagione = 'Stagione' in tutte.columns

    if usa_oos and ha_stagione:
        indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
    else:
        indici = list(range(15, len(tutte)))

    if not indici:
        return None

    calib_info = carica_calibratore(id_fd, "1x2") if (usa_calibrazione and id_fd) else None
    calibratore = calib_info["calibratore"] if calib_info else None
    colonne_h, colonne_d, colonne_a = classifica_colonne_quote(tutte.columns)

    n_partite, n_corrette = 0, 0
    n_scartate_disaccordo, n_scartate_no_quote = 0, 0
    per_segno = {"1": {"previste": 0, "corrette": 0}, "X": {"previste": 0, "corrette": 0}, "2": {"previste": 0, "corrette": 0}}

    for i in indici:
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue
        if calibratore is not None:
            m = applica_calibrazione_1x2(m, calibratore)

        probabilita = {"1": m['prob_1'], "X": m['prob_X'], "2": m['prob_2']}
        previsione_principale = max(probabilita, key=probabilita.get)

        if richiedi_accordo_mercato:
            quote = quote_mercato_normalizzate(partita, colonne_h, colonne_d, colonne_a)
            if quote is None:
                n_scartate_no_quote += 1
                continue
            quote_per_segno = {"1": quote["q_casa_equa"], "X": quote["q_x_equa"], "2": quote["q_trasf_equa"]}
            favorito_mercato = min(quote_per_segno, key=quote_per_segno.get)  # quota più bassa = favorito
            if previsione_principale != favorito_mercato:
                n_scartate_disaccordo += 1
                continue

        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        n_partite += 1
        per_segno[previsione_principale]["previste"] += 1
        if previsione_principale == esito:
            n_corrette += 1
            per_segno[previsione_principale]["corrette"] += 1

    return {"n_partite": n_partite, "n_corrette": n_corrette, "per_segno": per_segno,
            "n_scartate_disaccordo": n_scartate_disaccordo, "n_scartate_no_quote": n_scartate_no_quote}


# =====================================================================
# 🔧 PUNTO B — CALIBRAZIONE POST-HOC (1X2 + le 12 combo automatiche)
# Stessa tecnica isotonic regression già validata nell'App Risultati Fissi.
# Un correttore per 1X2 (corregge "Esito Finale" e il Value Bet), più uno
# separato per ciascuna delle 12 combinazioni automatiche (segno × Gol/No
# Gol × Over/Under 2.5) — la verità per allenarle è già nei risultati reali
# che scarichiamo (nessuna fonte dati nuova serve). Le combo libere con
# soglie diverse da 2.5 restano SENZA calibrazione: te lo segnaliamo,
# non lo nascondiamo.
# =====================================================================
LE_12_COMBO = [(s, g, t) for s in ["1", "X", "2"] for g in ["Goal", "NoGoal"] for t in ["Over", "Under"]]



# =====================================================================
# V13 — PLATT SOLO O/U 2.5
# Walk-forward: il fit usa esclusivamente partite precedenti alla partita
# da analizzare. Non modifica 1X2, Goal/No Goal, Multigol o la griglia.
# =====================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def _v13_training_ou_platt(dati, rho, ewma_span, emivita, data_riferimento, n_train=100):
    """Costruisce il training O/U 2.5 senza look-ahead."""
    if dati is None or len(dati) == 0 or 'FTHG' not in dati.columns or 'FTAG' not in dati.columns:
        return pd.DataFrame()
    df = dati.copy()
    if 'Date_parsed' not in df.columns:
        return pd.DataFrame()
    df = df[df['FTHG'].notna() & df['FTAG'].notna() & df['Date_parsed'].notna()].copy()
    if data_riferimento is not None and pd.notna(data_riferimento):
        df = df[df['Date_parsed'] < data_riferimento].copy()
    df = df.sort_values('Date_parsed').tail(max(int(n_train) + 40, int(n_train))).copy()
    rows = []
    for _, r in df.iterrows():
        d = r['Date_parsed']
        prior = df[df['Date_parsed'] < d].copy()
        if len(prior) < 20:
            continue
        try:
            m = calcola_modello_completo(
                prior,
                r['HomeTeam'], r['AwayTeam'],
                rho, ewma_span, emivita, pd.DataFrame(),
                data_riferimento=d
            )
            if not m or 'prob_under' not in m or 2.5 not in m['prob_under']:
                continue
            p_over = 1.0 - float(m['prob_under'][2.5]) / 100.0
            y_over = int(float(r['FTHG']) + float(r['FTAG']) > 2.5)
            if np.isfinite(p_over):
                rows.append({'p_raw': float(np.clip(p_over, 0.001, 0.999)), 'y_over': y_over, 'Date': d})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('Date').tail(int(n_train)).reset_index(drop=True)


def _v13_logit(p):
    """Logit numerically stable per una probabilita 0..1."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _v13_fit_platt_ou(df_train):
    """Fit Platt standard e restituisce (modello, pendenza, motivo).

    La calibrazione viene applicata solo quando la pendenza e' positiva:
    una calibrazione valida deve preservare l'ordinamento delle probabilita'.
    """
    if df_train is None or len(df_train) < 50:
        return None, np.nan, "campione insufficiente"
    y = df_train['y_over'].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None, np.nan, "un solo esito presente nel training"
    p = df_train['p_raw'].astype(float).to_numpy()
    x = np.array([_v13_logit(v) for v in p], dtype=float).reshape(-1, 1)
    model = LogisticRegression(solver='lbfgs', C=1e6, max_iter=1000)
    model.fit(x, y)
    coef = float(model.coef_[0, 0])
    if not np.isfinite(coef) or coef <= 0.0:
        return None, coef, "fit non monotono: pendenza non positiva"
    return model, coef, "applicabile"


def _v13_predict_platt_ou(p_raw_over, calibratore):
    if calibratore is None:
        return float(np.clip(p_raw_over, 0.001, 0.999))
    x = [[_v13_logit(p_raw_over)]]
    return float(np.clip(calibratore.predict_proba(x)[0, 1], 0.001, 0.999))


def _v13_apply_platt_ou(modello, calibratore):
    """Applica Platt esclusivamente alla probabilita O/U 2.5 del match."""
    if modello is None or calibratore is None:
        return modello
    p_raw_over = 1.0 - float(modello['prob_under'][2.5]) / 100.0
    p_platt_over = _v13_predict_platt_ou(p_raw_over, calibratore)
    modello['prob_under'][2.5] = (1.0 - p_platt_over) * 100.0
    return modello

def _path_calibratore(id_fd, chiave="1x2"):
    return f"calibratore_combo_{chiave}_{id_fd}.pkl"


def _salva_calibratore(id_fd, chiave, calibratore, n_oss):
    try:
        with open(_path_calibratore(id_fd, chiave), "wb") as f:
            pickle.dump({"calibratore": calibratore, "n_osservazioni": n_oss,
                         "timestamp": datetime.now().isoformat(timespec="minutes")}, f)
    except Exception:
        pass


def carica_calibratore(id_fd, chiave="1x2"):
    path = _path_calibratore(id_fd, chiave)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _allena_isotonic(osservazioni):
    if len(osservazioni) < 100:
        return None
    x = np.array([p for p, _ in osservazioni])
    y = np.array([1.0 if av else 0.0 for _, av in osservazioni])
    calibratore = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    calibratore.fit(x, y)
    return calibratore


def applica_calibrazione_1x2(modello, calibratore):
    if calibratore is None or modello is None:
        return modello
    p1 = float(calibratore.predict([modello['prob_1']/100])[0])
    px = float(calibratore.predict([modello['prob_X']/100])[0])
    p2 = float(calibratore.predict([modello['prob_2']/100])[0])
    tot = p1 + px + p2
    if tot <= 0:
        return modello
    m = dict(modello)
    m['prob_1'], m['prob_X'], m['prob_2'] = p1/tot*100, px/tot*100, p2/tot*100
    return m


def allena_tutte_le_calibrazioni(dati_completi, rho, ewma_span, emivita, id_fd):
    """Un'unica passata sui dati storici: per ogni partita calcola il modello
    UNA volta, poi costruisce le osservazioni (predetto, avverato) per 1X2 e
    per tutte e 12 le combo insieme — efficiente, una sola passata."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    oss_1x2 = []
    oss_combo = {c: [] for c in LE_12_COMBO}

    for i in range(15, len(tutte)):
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue

        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        oss_1x2.append((m['prob_1']/100, esito == '1'))
        oss_1x2.append((m['prob_X']/100, esito == 'X'))
        oss_1x2.append((m['prob_2']/100, esito == '2'))

        tot_gol_reale = partita['FTHG'] + partita['FTAG']
        entrambe_reale = partita['FTHG'] > 0 and partita['FTAG'] > 0
        for (segno_c, gg_c, tipo_c) in LE_12_COMBO:
            prob_c = calcola_combo_libera(m['griglia'], segno=segno_c, soglia_gol=2.5, tipo_soglia=tipo_c, gol_nogol=gg_c)
            avverato_c = (segno_c == esito) and \
                         ((gg_c == "Goal") == entrambe_reale) and \
                         ((tipo_c == "Over") == (tot_gol_reale > 2.5))
            oss_combo[(segno_c, gg_c, tipo_c)].append((prob_c/100, avverato_c))

    risultati = {"1x2": {"n": len(oss_1x2), "ok": False}}
    cal_1x2 = _allena_isotonic(oss_1x2)
    if cal_1x2:
        _salva_calibratore(id_fd, "1x2", cal_1x2, len(oss_1x2))
        risultati["1x2"]["ok"] = True

    for chiave_c, oss_c in oss_combo.items():
        nome_chiave = f"{chiave_c[0]}_{chiave_c[1]}_{chiave_c[2]}"
        cal_c = _allena_isotonic(oss_c)
        risultati[nome_chiave] = {"n": len(oss_c), "ok": cal_c is not None}
        if cal_c:
            _salva_calibratore(id_fd, nome_chiave, cal_c, len(oss_c))

    return risultati


# =====================================================================
# 🔧 BACKTEST COMBO — stessa logica del backtest 1X2 senza filtro, applicata
# alla combo automatica più probabile di ogni partita (tra le 12 possibili).
# Usa la calibrazione per-combo se disponibile e attiva, e la stessa verità
# (risultato reale) già usata per allenarle — nessuna fonte dati nuova.
# =====================================================================
def esegui_backtest_combo(dati_completi, rho, ewma_span, emivita, usa_oos, id_fd=None,
                           usa_calibrazione=False, richiedi_accordo_mercato=False, richiedi_accordo_ou=False):
    """richiedi_accordo_mercato: valuta solo le combo il cui SEGNO è d'accordo
    col favorito del mercato (idea #1, già validata su 1X2: +5/+10 punti).
    richiedi_accordo_ou (idea #1b, DA VERIFICARE): in aggiunta, richiede che
    anche la componente OVER/UNDER sia d'accordo con la quota di mercato
    Over/Under 2.5 — copre una seconda delle tre dimensioni di una combo
    (il filtro sul solo segno ne copre una su tre, la terza — Gol/No Gol —
    resta comunque non filtrabile: non esiste una quota di mercato gratuita
    per quel mercato nei file che usiamo)."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    ha_stagione = 'Stagione' in tutte.columns

    if usa_oos and ha_stagione:
        indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
    else:
        indici = list(range(15, len(tutte)))

    if not indici:
        return None

    colonne_h, colonne_d, colonne_a = classifica_colonne_quote(tutte.columns)
    colonne_over, colonne_under = classifica_colonne_over_under(tutte.columns, "2.5")
    n_partite, n_corrette = 0, 0
    n_scartate_disaccordo, n_scartate_disaccordo_ou, n_scartate_no_quote = 0, 0, 0
    per_combo = {f"{s}_{g}_{t}": {"previste": 0, "corrette": 0} for (s, g, t) in LE_12_COMBO}

    for i in indici:
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue

        probabilita_combo = {}
        for (s, g, t) in LE_12_COMBO:
            nome_chiave = f"{s}_{g}_{t}"
            prob_grezza = calcola_combo_libera(m['griglia'], segno=s, soglia_gol=2.5, tipo_soglia=t, gol_nogol=g)
            prob_finale = prob_grezza
            if usa_calibrazione and id_fd:
                calib_c = carica_calibratore(id_fd, nome_chiave)
                if calib_c:
                    prob_finale = float(calib_c["calibratore"].predict([prob_grezza/100])[0]) * 100
            probabilita_combo[nome_chiave] = prob_finale

        combo_prevista = max(probabilita_combo, key=probabilita_combo.get)
        s_p, g_p, t_p = combo_prevista.split("_")

        if richiedi_accordo_mercato or richiedi_accordo_ou:
            quote = quote_mercato_normalizzate(partita, colonne_h, colonne_d, colonne_a)
            if quote is None:
                n_scartate_no_quote += 1
                continue

        if richiedi_accordo_mercato:
            quote_per_segno = {"1": quote["q_casa_equa"], "X": quote["q_x_equa"], "2": quote["q_trasf_equa"]}
            favorito_mercato = min(quote_per_segno, key=quote_per_segno.get)
            if s_p != favorito_mercato:
                n_scartate_disaccordo += 1
                continue

        if richiedi_accordo_ou:
            quote_ou = quote_over_under_normalizzate(partita, colonne_over, colonne_under)
            if quote_ou is None:
                n_scartate_no_quote += 1
                continue
            favorito_ou_mercato = "Over" if quote_ou["q_over_equa"] < quote_ou["q_under_equa"] else "Under"
            if t_p != favorito_ou_mercato:
                n_scartate_disaccordo_ou += 1
                continue

        esito = '1' if partita['FTHG'] > partita['FTAG'] else ('2' if partita['FTHG'] < partita['FTAG'] else 'X')
        tot_gol_reale = partita['FTHG'] + partita['FTAG']
        entrambe_reale = partita['FTHG'] > 0 and partita['FTAG'] > 0
        avverata = (s_p == esito) and ((g_p == "Goal") == entrambe_reale) and ((t_p == "Over") == (tot_gol_reale > 2.5))

        n_partite += 1
        per_combo[combo_prevista]["previste"] += 1
        if avverata:
            n_corrette += 1
            per_combo[combo_prevista]["corrette"] += 1

    return {"n_partite": n_partite, "n_corrette": n_corrette, "per_combo": per_combo,
            "n_scartate_disaccordo": n_scartate_disaccordo, "n_scartate_disaccordo_ou": n_scartate_disaccordo_ou,
            "n_scartate_no_quote": n_scartate_no_quote}


# =====================================================================
# =====================================================================
# 🧊 V15 — OPERATIVO: PLATT TOP1 COMBO CON REGOLA CONGELATA
# Freeze validato nei test V14:
#   - soglia di attivazione sulla Top1 RAW: >= 35%
#   - training PLATT: 100 osservazioni precedenti
#   - stesso campionato
#   - walk-forward, nessuna informazione futura
#   - salvaguardia monotona: pendenza > 0; altrimenti si mantiene RAW
# Importante: PLATT calibra la probabilità dell'evento Top1, non ricostruisce
# le altre 11 probabilità e non modifica il ranking delle combo.
# =====================================================================
V15_TOP1_THRESHOLD = 0.35
V15_TOP1_TRAIN = 100
V15_TOP1_MIN_TRAIN = 50
V15_TOP1_MAX_TRAIN = 100


@st.cache_data(show_spinner=False, ttl=3600)
def _v15_build_top1_history(id_fd, rho, ewma_span, emivita):
    """Costruisce la storia OOS della Top1 Combo RAW per un singolo campionato."""
    try:
        dati_hist = carica_dati_campionato(id_fd)
        if dati_hist is None or dati_hist.empty:
            return pd.DataFrame()
        tutte = dati_hist[dati_hist['FTHG'].notna() & dati_hist['FTAG'].notna()].reset_index(drop=True)
        rows = []
        for i in range(15, len(tutte)):
            partita = tutte.iloc[i]
            try:
                m = calcola_modello_completo(
                    tutte.iloc[:i],
                    partita['HomeTeam'], partita['AwayTeam'],
                    rho, ewma_span, emivita, pd.DataFrame(),
                    data_riferimento=partita.get('Date_parsed')
                )
                if m is None or 'griglia' not in m:
                    continue
                probs = {}
                for segno in ['1', 'X', '2']:
                    for gg in ['Goal', 'NoGoal']:
                        for tipo in ['Over', 'Under']:
                            key = f'{segno}_{gg}_{tipo}'
                            probs[key] = float(calcola_combo_libera(
                                m['griglia'], segno=segno, soglia_gol=2.5,
                                tipo_soglia=tipo, gol_nogol=gg
                            )) / 100.0
                if not probs:
                    continue
                top1_key, top1_prob = max(probs.items(), key=lambda kv: kv[1])
                esito = ('1' if float(partita['FTHG']) > float(partita['FTAG'])
                         else ('2' if float(partita['FTHG']) < float(partita['FTAG']) else 'X'))
                gg_true = 'Goal' if float(partita['FTHG']) > 0 and float(partita['FTAG']) > 0 else 'NoGoal'
                tt_true = 'Over' if float(partita['FTHG']) + float(partita['FTAG']) > 2.5 else 'Under'
                true_key = f'{esito}_{gg_true}_{tt_true}'
                rows.append({
                    'data': partita.get('Date_parsed'),
                    'top1_prob': float(np.clip(top1_prob, 1e-6, 1.0 - 1e-6)),
                    'top1_hit': int(top1_key == true_key),
                })
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        h = pd.DataFrame(rows)
        h['data'] = pd.to_datetime(h['data'], errors='coerce')
        h['top1_prob'] = pd.to_numeric(h['top1_prob'], errors='coerce')
        h['top1_hit'] = pd.to_numeric(h['top1_hit'], errors='coerce')
        h = h.dropna(subset=['top1_prob', 'top1_hit'])
        return h.sort_values('data', kind='mergesort').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _v15_logit(p):
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _v15_fit_platt_top1(hist_df, n_train=V15_TOP1_TRAIN):
    """Fit Platt sulla Top1 con sole osservazioni precedenti e guardia monotona."""
    if hist_df is None or hist_df.empty:
        return None, np.nan, 'nessun training', 0
    x = hist_df.copy()
    required = ['top1_prob', 'top1_hit']
    if any(c not in x.columns for c in required):
        return None, np.nan, 'colonne mancanti', 0
    x['top1_prob'] = pd.to_numeric(x['top1_prob'], errors='coerce')
    x['top1_hit'] = pd.to_numeric(x['top1_hit'], errors='coerce')
    x = x.dropna(subset=required).copy()
    x = x[x['top1_prob'].between(1e-6, 1.0 - 1e-6)].copy()
    x['top1_hit'] = x['top1_hit'].astype(int)
    limit = int(min(max(n_train, V15_TOP1_MIN_TRAIN), V15_TOP1_MAX_TRAIN))
    x = x.tail(limit).reset_index(drop=True)
    n = len(x)
    if n < V15_TOP1_MIN_TRAIN:
        return None, np.nan, 'campione insufficiente', n
    y = x['top1_hit'].to_numpy(dtype=int).reshape(-1)
    if np.unique(y).size < 2:
        return None, np.nan, 'un solo esito presente nel training', n
    z = np.asarray([_v15_logit(v) for v in x['top1_prob'].to_numpy(dtype=float)], dtype=float).reshape(-1, 1)
    model = LogisticRegression(solver='lbfgs', C=1e6, max_iter=1000)
    model.fit(z, y)
    coef_arr = np.asarray(model.coef_, dtype=float).reshape(-1)
    coef = float(coef_arr[0]) if coef_arr.size else np.nan
    if not np.isfinite(coef) or coef <= 0.0:
        return None, coef, 'fit non monotono: pendenza non positiva', n
    return model, coef, 'applicabile', n


def _v15_predict_platt_top1(p_raw, model):
    p_raw = float(np.clip(p_raw, 1e-6, 1.0 - 1e-6))
    if model is None:
        return p_raw
    z = np.asarray([[_v15_logit(p_raw)]], dtype=float)
    proba = np.asarray(model.predict_proba(z), dtype=float).reshape(-1)
    if proba.size < 2:
        return p_raw
    return float(np.clip(proba[1], 1e-6, 1.0 - 1e-6))


def _v15_operational_top1_platt(id_fd, rho, ewma_span, emivita, modello, data_riferimento):
    """Applica il freeze V15 alla sola probabilità della Top1 Combo."""
    probs = {}
    if modello is None or 'griglia' not in modello:
        return None
    for segno in ['1', 'X', '2']:
        for gg in ['Goal', 'NoGoal']:
            for tipo in ['Over', 'Under']:
                key = f'{segno}_{gg}_{tipo}'
                try:
                    probs[key] = float(calcola_combo_libera(
                        modello['griglia'], segno=segno, soglia_gol=2.5,
                        tipo_soglia=tipo, gol_nogol=gg
                    )) / 100.0
                except Exception:
                    continue
    if not probs:
        return None
    top1_key, p_raw = max(probs.items(), key=lambda kv: kv[1])
    info = {
        'top1_key': top1_key,
        'top1_raw': float(np.clip(p_raw, 1e-6, 1.0 - 1e-6)),
        'threshold': V15_TOP1_THRESHOLD,
        'selected': bool(p_raw >= V15_TOP1_THRESHOLD),
        'platt_applied': False,
        'platt_prob': float(np.clip(p_raw, 1e-6, 1.0 - 1e-6)),
        'coef': np.nan,
        'train_n': 0,
        'reason': 'Top1 RAW < 35%'
    }
    if p_raw < V15_TOP1_THRESHOLD:
        return info

    hist = _v15_build_top1_history(id_fd, rho, ewma_span, emivita)
    if data_riferimento is not None and pd.notna(data_riferimento) and not hist.empty:
        hist = hist[hist['data'] < pd.to_datetime(data_riferimento)].copy()

    model, coef, reason, n_eff = _v15_fit_platt_top1(hist[['top1_prob','top1_hit']] if not hist.empty else hist, V15_TOP1_TRAIN)
    info['train_n'] = int(n_eff)
    info['coef'] = coef
    info['reason'] = reason
    if model is not None:
        info['platt_prob'] = _v15_predict_platt_top1(p_raw, model)
        info['platt_applied'] = True
    return info


st.caption("Versione operativa V15.2 — modello V11 completo + PLATT O/U 2.5 + PLATT Top1 Combo (freeze: 35% / 100 osservazioni)")

st.info("Uso operativo: il modello completo resta invariato. V13 calibra O/U 2.5; V15 calibra la probabilità della Top1 Combo con PLATT walk-forward (100 osservazioni precedenti dello stesso campionato) solo quando la Top1 RAW è ≥35%. Nessuna informazione futura.")

# Parametri del modello fissati ai valori di default validati — non più
# esposti nell'interfaccia: erano controlli tecnici che richiedevano di
# sapere cosa fanno per essere usati bene, e nell'uso normale non si toccano.
rho_val = -0.10        # correzione Dixon-Coles (valore tipico da letteratura)
ewma_span_val = 6      # finestra della forma recente
emivita_val = 180      # decadimento temporale in giorni

with st.sidebar:
    st.header("⚙️ Configurazione & API")
    api_key_input = st.text_input("Chiave API football-data.org (per Coppe)", type="password",
                                   help="Gratuita: football-data.org/client/register")
    st.divider()
    usa_calibrazione = st.checkbox(
        "🎯 Applica calibrazione 1X2 (se allenata)",
        value=False,
        help="Calibrazione separata del mercato 1X2. Disattivata di default."
    )
    usa_platt_v13 = st.checkbox(
        "🎯 V13 — PLATT solo O/U 2.5",
        value=True,
        help="Corregge esclusivamente la probabilità Over/Under 2.5 usando un fit walk-forward sulle partite precedenti."
    )


# =====================================================================
# 🚀 V15.1 — PULSANTE DI AVVIO OPERATIVO
# Il modello e la regola V15 restano invariati. Questo blocco aggiunge
# esclusivamente un comando esplicito per avviare l'interfaccia operativa.
# =====================================================================
if "v15_operativo_avviato" not in st.session_state:
    st.session_state.v15_operativo_avviato = False

st.divider()
st.markdown("## 🚀 V15 — AVVIO OPERATIVO")
st.caption(
    "Premi il pulsante per avviare l'analisi delle partite. Dopo l'avvio "
    "puoi scegliere campionato e partita; la regola V15 viene applicata "
    "automaticamente alla Top1 Combo quando le condizioni del freeze sono soddisfatte."
)

col_start, col_status = st.columns([1, 2])
with col_start:
    if st.button("▶️ AVVIA V15 OPERATIVO", key="v15_start", type="primary"):
        st.session_state.v15_operativo_avviato = True
with col_status:
    if st.session_state.v15_operativo_avviato:
        st.success("✅ V15 operativo avviato")
    else:
        st.info("⏸️ V15 non ancora avviato")

if not st.session_state.v15_operativo_avviato:
    st.stop()

# =====================================================================
# V15.3 — PANORAMICA UNIFICATA DELLE PROSSIME PARTITE
# Mostra in un'unica tabella tutte le fixture future reperibili dalle fonti
# gratuite nazionali; i filtri non modificano il modello di previsione.
# =====================================================================
st.markdown("## 📅 Panoramica prossime partite — tutti i campionati")
st.caption("Elenco aggregato delle fixture future disponibili. Seleziona qui periodo e campionato per trovare gli incontri; l'analisi dettagliata si avvia nella sezione sottostante.")

with st.spinner("Raccolta calendario dei campionati nazionali disponibili..."):
    _fixture_rows = []
    for _nome_lega, _info_lega in CAMPIONATI_DOMESTICI.items():
        _df_fx = carica_fixture_future(_info_lega["id_fd"])
        if _df_fx is not None and not _df_fx.empty:
            for _, _r_fx in _df_fx.iterrows():
                _fixture_rows.append({
                    "Data": _r_fx.get("Date_parsed"),
                    "Campionato": _nome_lega,
                    "Casa": _r_fx.get("HomeTeam", ""),
                    "Trasferta": _r_fx.get("AwayTeam", ""),
                    "_id_fd": _info_lega["id_fd"],
                })

_df_prossime_tutte = pd.DataFrame(_fixture_rows)
if _df_prossime_tutte.empty:
    st.info("La fonte gratuita non restituisce in questo momento fixture future per i campionati nazionali. Puoi comunque usare la selezione di campionato più sotto o riprovare più tardi.")
else:
    _df_prossime_tutte["Data"] = pd.to_datetime(_df_prossime_tutte["Data"], errors="coerce")
    _oggi_filtro = pd.Timestamp(date.today()).normalize()
    _periodo = st.radio("🗓️ Periodo", ["Oggi", "Domani", "Prossimi 7 giorni", "Prossimi 30 giorni", "Tutte le date disponibili"], index=2, horizontal=True, key="filtro_periodo_prossime_tutte")
    _lega_options = ["Tutti i campionati"] + sorted(_df_prossime_tutte["Campionato"].dropna().unique().tolist())
    _lega_filtro = st.selectbox("🏆 Filtra campionato", _lega_options, key="filtro_lega_prossime_tutte")
    if _periodo == "Oggi":
        _fine_periodo = _oggi_filtro
    elif _periodo == "Domani":
        _oggi_filtro = _oggi_filtro + pd.Timedelta(days=1)
        _fine_periodo = _oggi_filtro
    elif _periodo == "Prossimi 7 giorni":
        _fine_periodo = _oggi_filtro + pd.Timedelta(days=6)
    elif _periodo == "Prossimi 30 giorni":
        _fine_periodo = _oggi_filtro + pd.Timedelta(days=29)
    else:
        _fine_periodo = _df_prossime_tutte["Data"].max().normalize()
    _df_vis = _df_prossime_tutte[(_df_prossime_tutte["Data"].dt.normalize() >= _oggi_filtro) & (_df_prossime_tutte["Data"].dt.normalize() <= _fine_periodo)].copy()
    if _lega_filtro != "Tutti i campionati":
        _df_vis = _df_vis[_df_vis["Campionato"] == _lega_filtro]
    _df_vis = _df_vis.sort_values(["Data", "Campionato", "Casa"], na_position="last")
    if _df_vis.empty:
        st.warning("Nessuna partita trovata con questi filtri. Prova un periodo più ampio o tutti i campionati.")
    else:
        _df_vis["Data e ora/data"] = _df_vis["Data"].dt.strftime("%d/%m/%Y")
        _tab_prossime = _df_vis[["Data e ora/data", "Campionato", "Casa", "Trasferta"]].reset_index(drop=True)
        st.metric("Partite trovate", len(_tab_prossime))
        st.dataframe(_tab_prossime, use_container_width=True, hide_index=True)
        st.download_button("📥 Scarica calendario filtrato (CSV)", _tab_prossime.to_csv(index=False).encode("utf-8-sig"), file_name="combo_prossime_partite.csv", mime="text/csv", key="download_prossime_tutte")

# =====================================================================
# V15.6 — INDICATORE PRELIMINARE DI AFFIDABILITÀ PRONOSTICI
# Richiama il motore esistente in modalità walk-forward: per ogni partita
# usa esclusivamente risultati antecedenti alla data della fixture.
# Non modifica V11/V13/V15 né salva/modifica calibratori.
# =====================================================================
st.markdown("## 🤖 Pronostici automatici — prossime partite")
st.caption("Calcola 1/X/2 e doppia chance per le fixture filtrate. Ogni stima usa risultati precedenti alla partita; le percentuali 1X2 sono RAW, salvo calibrazione 1X2 opzionale già selezionata nella barra laterale.")
st.info("🧭 L'indicatore di affidabilità è una prima stima prudenziale basata sul numero di partite disponibili per entrambe le squadre e sulla distanza tra la probabilità del segno più probabile e quella del secondo segno. Non è una garanzia di esito e non è ancora una misura validata sui risultati storici.")

if not _df_prossime_tutte.empty:
    _df_batch_base = _df_prossime_tutte.copy()
    _df_batch_base["Data"] = pd.to_datetime(_df_batch_base["Data"], errors="coerce")
    _oggi_batch = pd.Timestamp(date.today()).normalize()
    _periodo_batch = st.radio("Periodo per i pronostici", ["Oggi", "Domani", "Prossimi 7 giorni", "Prossimi 30 giorni", "Tutte le date disponibili"], index=2, horizontal=True, key="periodo_pronostici_batch")
    if _periodo_batch == "Oggi":
        _inizio_batch = _oggi_batch; _fine_batch = _oggi_batch
    elif _periodo_batch == "Domani":
        _inizio_batch = _oggi_batch + pd.Timedelta(days=1); _fine_batch = _inizio_batch
    elif _periodo_batch == "Prossimi 7 giorni":
        _inizio_batch = _oggi_batch; _fine_batch = _oggi_batch + pd.Timedelta(days=6)
    elif _periodo_batch == "Prossimi 30 giorni":
        _inizio_batch = _oggi_batch; _fine_batch = _oggi_batch + pd.Timedelta(days=29)
    else:
        _inizio_batch = _oggi_batch; _fine_batch = _df_batch_base["Data"].max().normalize()
    _df_batch_base = _df_batch_base[
        (_df_batch_base["Data"].dt.normalize() >= _inizio_batch) &
        (_df_batch_base["Data"].dt.normalize() <= _fine_batch)
    ].copy()
    _leghe_batch = ["Tutti i campionati"] + sorted(_df_batch_base["Campionato"].dropna().unique().tolist())
    _lega_batch_sel = st.selectbox("Campionato per i pronostici", _leghe_batch, key="lega_pronostici_batch")
    if _lega_batch_sel != "Tutti i campionati":
        _df_batch_base = _df_batch_base[_df_batch_base["Campionato"] == _lega_batch_sel]
    _limite_batch = st.selectbox("Numero massimo di partite da calcolare", [10, 20, 30, 50, 100], index=2, key="limite_pronostici_batch")
    _df_batch_base = _df_batch_base.sort_values(["Data", "Campionato", "Casa"]).head(_limite_batch)
    st.caption(f"Partite selezionate per il calcolo: {len(_df_batch_base)}. Il calcolo parte solo quando premi il pulsante.")
    if st.button("⚽ CALCOLA PRONOSTICI DELLE PARTITE VISUALIZZATE", type="primary", key="calcola_batch_pronostici"):
        _righe_pronostici = []
        _cache_dati_lega = {}
        _globale_batch = carica_tutti_i_campionati()
        _progress_batch = st.progress(0, text="Preparazione dei dati storici...")
        _errori_batch = 0
        _tot_batch = max(1, len(_df_batch_base))
        for _idx_batch, (_idx_riga_batch, _fx_batch) in enumerate(_df_batch_base.iterrows(), start=1):
            try:
                _nome_lega_batch = _fx_batch["Campionato"]
                _id_lega_batch = str(_fx_batch["_id_fd"])
                if _id_lega_batch not in _cache_dati_lega:
                    _cache_dati_lega[_id_lega_batch] = carica_dati_campionato(_id_lega_batch)
                _dati_lega_batch = _cache_dati_lega[_id_lega_batch]
                _data_fx_batch = pd.to_datetime(_fx_batch["Data"], errors="coerce")
                if _dati_lega_batch is None or _dati_lega_batch.empty or pd.isna(_data_fx_batch):
                    raise ValueError("Dati storici o data partita non disponibili")
                _storico_batch = _dati_lega_batch[
                    _dati_lega_batch["Date_parsed"] < _data_fx_batch
                ].copy()
                _globale_storico_batch = _globale_batch.copy()
                if not _globale_storico_batch.empty:
                    _globale_storico_batch = _globale_storico_batch[
                        _globale_storico_batch["Date_parsed"] < _data_fx_batch
                    ].copy()
                _mod_batch = calcola_modello_completo(
                    _storico_batch, str(_fx_batch["Casa"]), str(_fx_batch["Trasferta"]),
                    rho_val, ewma_span_val, emivita_val, _globale_storico_batch,
                    data_riferimento=_data_fx_batch
                )
                if _mod_batch is None:
                    raise ValueError("Il modello non ha prodotto una stima")
                _p1b = float(_mod_batch["prob_1"]); _pxb = float(_mod_batch["prob_X"]); _p2b = float(_mod_batch["prob_2"])
                _info_cal_batch = None
                if usa_calibrazione:
                    try:
                        _info_cal_batch = carica_calibratore(_id_lega_batch, "1x2")
                        if _info_cal_batch:
                            _mod_batch = applica_calibrazione_1x2(_mod_batch, _info_cal_batch["calibratore"])
                            _p1b = float(_mod_batch["prob_1"]); _pxb = float(_mod_batch["prob_X"]); _p2b = float(_mod_batch["prob_2"])
                    except Exception:
                        _info_cal_batch = None
                _prob_segni_batch = {"1": _p1b, "X": _pxb, "2": _p2b}
                _segno_batch = max(_prob_segni_batch, key=_prob_segni_batch.get)
                _dc_batch = {"1X": _p1b + _pxb, "X2": _pxb + _p2b, "12": _p1b + _p2b}
                _dc_top_batch = max(_dc_batch, key=_dc_batch.get)
                _n_casa_batch = int(_mod_batch.get("n_casa", 0)); _n_trasf_batch = int(_mod_batch.get("n_trasf", 0))
                _min_storico_batch = min(_n_casa_batch, _n_trasf_batch)
                _qualita_batch = "Dati limitati" if _min_storico_batch < 5 else "Dati squadra ≥5"
                _prob_ordinate_batch = sorted(_prob_segni_batch.values(), reverse=True)
                _distacco_batch = (_prob_ordinate_batch[0] - _prob_ordinate_batch[1]) * 100
                _punti_campione_batch = 0 if _min_storico_batch < 5 else (1 if _min_storico_batch < 8 else 2)
                _punti_segnale_batch = 0 if _distacco_batch < 5 else (1 if _distacco_batch < 10 else 2)
                _punteggio_affidabilita_batch = _punti_campione_batch + _punti_segnale_batch
                if _punteggio_affidabilita_batch <= 1:
                    _affidabilita_batch = "Bassa"
                elif _punteggio_affidabilita_batch <= 3:
                    _affidabilita_batch = "Media"
                else:
                    _affidabilita_batch = "Alta"
                _righe_pronostici.append({
                    "Data": _data_fx_batch.strftime("%d/%m/%Y"), "Campionato": _nome_lega_batch,
                    "Partita": f'{_fx_batch["Casa"]} - {_fx_batch["Trasferta"]}',
                    "1 (%)": round(_p1b, 1), "X (%)": round(_pxb, 1), "2 (%)": round(_p2b, 1),
                    "Segno più probabile": _segno_batch, "Prob. segno (%)": round(_prob_segni_batch[_segno_batch], 1),
                    "Doppia chance": _dc_top_batch, "Prob. DC (%)": round(_dc_batch[_dc_top_batch], 1),
                    "Partite casa": _n_casa_batch, "Partite trasferta": _n_trasf_batch,
                    "Indicatore dati": _qualita_batch,
                    "Distacco 1°-2° segno (pp)": round(_distacco_batch, 1),
                    "Affidabilità preliminare": _affidabilita_batch,
                    "Calibrazione 1X2": "Applicata" if _info_cal_batch else "RAW",
                })
            except Exception as _err_batch:
                _errori_batch += 1
                _righe_pronostici.append({
                    "Data": pd.to_datetime(_fx_batch["Data"]).strftime("%d/%m/%Y") if pd.notna(_fx_batch["Data"]) else "",
                    "Campionato": _fx_batch.get("Campionato", ""),
                    "Partita": f'{_fx_batch.get("Casa", "")} - {_fx_batch.get("Trasferta", "")}',
                    "1 (%)": None, "X (%)": None, "2 (%)": None,
                    "Segno più probabile": "N/D", "Prob. segno (%)": None,
                    "Doppia chance": "N/D", "Prob. DC (%)": None,
                    "Partite casa": None, "Partite trasferta": None,
                    "Indicatore dati": f"Non calcolato: {type(_err_batch).__name__}",
                    "Distacco 1°-2° segno (pp)": None,
                    "Affidabilità preliminare": "N/D",
                    "Calibrazione 1X2": "N/D",
                })
            _progress_batch.progress(_idx_batch / _tot_batch, text=f"Analisi partite: {_idx_batch}/{_tot_batch}")
        _progress_batch.empty()
        _df_risultati_batch = pd.DataFrame(_righe_pronostici)
        st.session_state["combo_pronostici_batch_v154"] = _df_risultati_batch
        st.session_state["combo_pronostici_batch_v154_data"] = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.success(f"Calcolo terminato: {len(_righe_pronostici) - _errori_batch} pronostici calcolati; {_errori_batch} non disponibili.")

    if "combo_pronostici_batch_v154" in st.session_state:
        _df_risultati_batch = st.session_state["combo_pronostici_batch_v154"]
        st.caption("Ultimo calcolo: " + st.session_state.get("combo_pronostici_batch_v154_data", "n/d") + ". Le righe N/D indicano dati insufficienti o un errore di elaborazione: non sono pronostici.")
        st.dataframe(_df_risultati_batch, use_container_width=True, hide_index=True)
        st.download_button("📥 Scarica pronostici in CSV", _df_risultati_batch.to_csv(index=False).encode("utf-8-sig"), file_name="combo_pronostici_v15_6.csv", mime="text/csv", key="download_pronostici_batch_v154")
else:
    st.info("Non ci sono fixture future disponibili da analizzare automaticamente in questo momento.")

st.divider()
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

    # Le prossime partite sono la scelta predefinita; le storiche restano
    # disponibili solo come modalità separata per controlli e confronti.
    modalita_partite = st.radio(
        "📅 Quali partite vuoi analizzare?",
        ["Prossime partite", "Ultime partite giocate (controllo)"],
        horizontal=True,
        key="modalita_partite_domestiche",
    )
    opzioni_partite, mappa_partite = [], []
    if modalita_partite == "Prossime partite":
        if not fixture_future.empty:
            for _, r in fixture_future.iterrows():
                data_label = r['Date_parsed'].strftime('%d/%m/%Y') if pd.notna(r.get('Date_parsed')) else str(r.get('Date', '?'))
                opzioni_partite.append(f"📅 {data_label} — {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}")
                mappa_partite.append(r.to_dict())
        else:
            st.warning(
                "Non risultano fixture future per questo campionato nella fonte gratuita. "
                "Puoi riprovare più tardi oppure scegliere le ultime partite giocate per un controllo."
            )
    else:
        storiche = dati[dati['FTHG'].notna()].tail(15).sort_values('Date_parsed', ascending=False)
        for _, r in storiche.iterrows():
            data_label = r['Date_parsed'].strftime('%d/%m/%Y') if pd.notna(r.get('Date_parsed')) else str(r.get('Date', '?'))
            opzioni_partite.append(f"🕘 {data_label} — {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}")
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
    if not info.get("gratis_confermato", False):
        st.info("ℹ️ Questa competizione non risultava nell'elenco confermato del piano gratuito "
                "di football-data.org — se ricevi 'Accesso Negato' è un limite del piano, non un bug.")

    with st.spinner("Connessione alle API delle Coppe e caricamento dati di supporto..."):
        risultato_api = carica_dati_api_europee(code_api, api_key_input)
        df_globale = carica_tutti_i_campionati()

    if isinstance(risultato_api, str) and risultato_api == "ERRORE_403":
        st.error("❌ **Accesso Negato (Errore 403)**: la tua chiave API gratuita non ha accesso a questa competizione.")
        st.stop()
    elif isinstance(risultato_api, str):
        st.error(f"Errore di connessione API: {risultato_api}")
        st.stop()

    dati = risultato_api
    if dati is None or len(dati) == 0:
        st.error("Nessun dato trovato per questa competizione.")
        st.stop()

    dati_storico = dati[dati['Status'] == 'FINISHED'].copy()
    oggi_ts = pd.Timestamp(date.today())
    dati_future = dati[(dati['Status'].isin(['SCHEDULED', 'TIMED'])) &
                       (dati['Date_parsed'] >= oggi_ts)].copy()
    modalita_partite_coppe = st.radio(
        "📅 Quali partite vuoi analizzare?",
        ["Prossime partite", "Ultime partite giocate (controllo)"],
        horizontal=True,
        key="modalita_partite_coppe",
    )
    opzioni_partite, mappa_partite = [], []
    if modalita_partite_coppe == "Prossime partite":
        for _, r in dati_future.sort_values('Date_parsed').iterrows():
            data_label = r['Date_parsed'].strftime('%d/%m/%Y') if pd.notna(r.get('Date_parsed')) else str(r.get('Date', '?'))
            opzioni_partite.append(f"📅 {data_label} — {r['HomeTeam']} vs {r['AwayTeam']}")
            mappa_partite.append(r.to_dict())
        if not opzioni_partite:
            st.warning("Nessuna partita futura programmata restituita dall'API per questa coppa.")
    else:
        for _, r in dati_storico.sort_values('Date_parsed', ascending=False).head(10).iterrows():
            data_label = r['Date_parsed'].strftime('%d/%m/%Y') if pd.notna(r.get('Date_parsed')) else str(r.get('Date', '?'))
            opzioni_partite.append(f"🕘 {data_label} — {r['HomeTeam']} vs {r['AwayTeam']}")
            mappa_partite.append(r.to_dict())
    is_coppa = True

if not opzioni_partite:
    st.warning("Nessuna partita disponibile nella modalità selezionata. Prova a cambiare campionato o modalità.")
else:
    scelta = st.selectbox("⚽ Seleziona la partita da analizzare", opzioni_partite)
    idx_sel = opzioni_partite.index(scelta)
    partita_sel = mappa_partite[idx_sel]

    # =====================================================================
    # 🔧 FIX #2 — FILTRO NO-LOOK-AHEAD
    # Prima: il modello riceveva TUTTO il dataset, comprese le giornate
    # successive alla partita selezionata (per le partite "GIOCATA"/
    # "RECENTE" già nel passato) — calcolava quindi le statistiche anche
    # con risultati che, a quella data, non erano ancora accaduti.
    # Ora: filtriamo sempre a "solo partite precedenti alla data selezionata".
    # =====================================================================
    data_rif_sel = partita_sel.get('Date_parsed')
    if pd.notna(data_rif_sel):
        dati_filtrati = dati[(dati['FTHG'].notna()) & (dati['Date_parsed'] < data_rif_sel)].copy() if 'Date_parsed' in dati.columns else dati
        df_globale_filtrato = df_globale[(df_globale['FTHG'].notna()) & (df_globale['Date_parsed'] < data_rif_sel)].copy() \
            if (df_globale is not None and not df_globale.empty and 'Date_parsed' in df_globale.columns) else df_globale
    else:
        dati_filtrati = dati[dati['FTHG'].notna()].copy() if 'FTHG' in dati.columns else dati
        df_globale_filtrato = df_globale

    modello = calcola_modello_completo(dati_filtrati, partita_sel['HomeTeam'], partita_sel['AwayTeam'],
                                        rho_val, ewma_span_val, emivita_val, df_globale_filtrato,
                                        data_riferimento=data_rif_sel if pd.notna(data_rif_sel) else None)

    calib_1x2_info = None
    if modello is not None and not is_coppa and usa_calibrazione:
        calib_1x2_info = carica_calibratore(id_fd, "1x2")
        if calib_1x2_info:
            modello = applica_calibrazione_1x2(modello, calib_1x2_info["calibratore"])

    platt_v13_info = None
    if modello is not None and not is_coppa and usa_platt_v13:
        try:
            train_platt = _v13_training_ou_platt(
                dati, rho_val, ewma_span_val, emivita_val,
                data_rif_sel if pd.notna(data_rif_sel) else None,
                n_train=100
            )
            platt_v13, platt_coef, platt_status = _v13_fit_platt_ou(train_platt)
            p_raw = 1.0 - float(modello['prob_under'][2.5]) / 100.0
            if platt_v13 is not None:
                p_new = _v13_predict_platt_ou(p_raw, platt_v13)
                modello = _v13_apply_platt_ou(modello, platt_v13)
                platt_v13_info = {
                    'stato': 'applicato',
                    'n_train': len(train_platt),
                    'coef': platt_coef,
                    'raw_over': p_raw * 100.0,
                    'platt_over': p_new * 100.0,
                    'raw_under': (1.0 - p_raw) * 100.0,
                    'platt_under': (1.0 - p_new) * 100.0,
                }
            else:
                platt_v13_info = {
                    'stato': 'escluso',
                    'n_train': len(train_platt),
                    'coef': platt_coef,
                    'raw_over': p_raw * 100.0,
                    'raw_under': (1.0 - p_raw) * 100.0,
                    'motivo': platt_status,
                }
        except Exception as _platt_err:
            platt_v13_info = {'errore': f'{type(_platt_err).__name__}: {_platt_err}'}

    if modello is None:
        st.warning(f"⚠️ Impossibile elaborare il match per **{partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}**.")
    else:
        st.subheader(f"📊 Analisi Match: {partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}")
        if calib_1x2_info:
            st.caption(f"🎯 Probabilità 1X2 corrette con calibrazione (allenata su "
                       f"{calib_1x2_info['n_osservazioni']} osservazioni, {calib_1x2_info['timestamp']}).")

        # V15 operativo: PLATT Top1 Combo con regola congelata.
        v15_top1_info = None
        if not is_coppa:
            try:
                v15_top1_info = _v15_operational_top1_platt(
                    id_fd, rho_val, ewma_span_val, emivita_val, modello,
                    data_rif_sel if pd.notna(data_rif_sel) else None
                )
                if v15_top1_info is not None and v15_top1_info.get('selected'):
                    p_raw_pct = 100.0 * v15_top1_info['top1_raw']
                    p_final_pct = 100.0 * v15_top1_info['platt_prob']
                    if v15_top1_info.get('platt_applied'):
                        st.success(
                            f"🧊 **V15 TOP1 COMBO ATTIVA** — {v15_top1_info['top1_key']} · "
                            f"RAW {p_raw_pct:.1f}% → PLATT {p_final_pct:.1f}% · "
                            f"training: {v15_top1_info['train_n']} osservazioni · "
                            f"pendenza: {v15_top1_info['coef']:.4f}."
                        )
                    else:
                        st.warning(
                            f"⚠️ **V15 TOP1 ≥35%, ma PLATT non applicato** — {v15_top1_info['top1_key']} · "
                            f"RAW {p_raw_pct:.1f}% · training: {v15_top1_info['train_n']} · "
                            f"motivo: {v15_top1_info['reason']}. Resta la probabilità RAW."
                        )
                elif v15_top1_info is not None:
                    st.info(
                        f"ℹ️ **V15 TOP1 COMBO non attiva** — {v15_top1_info['top1_key']} · "
                        f"probabilità RAW {100.0*v15_top1_info['top1_raw']:.1f}% < soglia 35%."
                    )
            except Exception as _v15_err:
                st.warning(f"⚠️ V15 PLATT Top1 Combo non applicato per errore tecnico: {type(_v15_err).__name__}: {_v15_err}")

        if platt_v13_info and platt_v13_info.get('stato') == 'applicato':
            st.info(
                f"🎯 **V13 PLATT O/U 2.5 applicato** — training walk-forward: {platt_v13_info['n_train']} partite · "
                f"pendenza: {platt_v13_info['coef']:.4f}. "
                f"Over 2.5: {platt_v13_info['raw_over']:.1f}% → {platt_v13_info['platt_over']:.1f}% · "
                f"Under 2.5: {platt_v13_info['raw_under']:.1f}% → {platt_v13_info['platt_under']:.1f}%."
            )
        elif platt_v13_info and platt_v13_info.get('stato') == 'escluso':
            st.info(
                f"ℹ️ **V13 PLATT O/U 2.5 escluso automaticamente** — training: {platt_v13_info['n_train']} partite · "
                f"pendenza: {platt_v13_info['coef']:.4f}. "
                f"Il fit non e' monotono; resta la probabilita RAW: "
                f"Over 2.5 {platt_v13_info['raw_over']:.1f}% · Under 2.5 {platt_v13_info['raw_under']:.1f}%."
            )
        elif platt_v13_info and 'errore' in platt_v13_info:
            st.warning(f"⚠️ PLATT O/U 2.5 non applicato per errore tecnico: {platt_v13_info['errore']}")

        # Avviso trasparenza dati (sostituisce il vecchio generatore silenzioso di dati finti)
        SOGLIA_AVVISO = 5
        avvisi = []
        if modello["n_casa"] < SOGLIA_AVVISO:
            avvisi.append(f"{partita_sel['HomeTeam']} (solo {modello['n_casa']} partite trovate)")
        if modello["n_trasf"] < SOGLIA_AVVISO:
            avvisi.append(f"{partita_sel['AwayTeam']} (solo {modello['n_trasf']} partite trovate)")
        if avvisi:
            st.warning("⚠️ Stima poco affidabile per: " + " e ".join(avvisi) +
                       " — il modello converge verso la media di lega/competizione per compensare "
                       "la scarsità di dati specifici, ma la previsione resta meno precisa del solito.")

        def crea_tabella(dati_dict, col_nome="Mercato"):
            df = pd.DataFrame(list(dati_dict.items()), columns=[col_nome, "Probabilità (%)"])
            df["Probabilità (%)"] = df["Probabilità (%)"].round(1)
            df = df.sort_values(by="Probabilità (%)", ascending=False).reset_index(drop=True)
            df["Probabilità (%)"] = df["Probabilità (%)"].astype(str) + "%"
            return df

        # =====================================================================
        # 🎯 RIEPILOGO OPERATIVO — 1X2 + DOPPIA CHANCE
        # Solo presentazione: non modifica nessuna probabilità o logica del modello.
        # La doppia chance deriva direttamente dalle probabilità 1X2 correnti.
        # =====================================================================
        p1 = float(modello['prob_1'])
        px = float(modello['prob_X'])
        p2 = float(modello['prob_2'])
        probabilita_1x2 = {"1": p1, "X": px, "2": p2}
        segno_top = max(probabilita_1x2, key=probabilita_1x2.get)

        doppia_chance = {
            "1X": p1 + px,
            "X2": px + p2,
            "12": p1 + p2,
        }
        doppia_chance_top = max(doppia_chance, key=doppia_chance.get)

        st.markdown("### 🎯 PRONOSTICO OPERATIVO")
        st.caption("Il segno singolo è scelto confrontando le tre probabilità del modello; la X è selezionabile esattamente come 1 e 2.")
        c_prob1, c_probX, c_prob2 = st.columns(3)
        c_prob1.metric("SEGNO 1", f"{p1:.1f}%")
        c_probX.metric("SEGNO X · PAREGGIO", f"{px:.1f}%")
        c_prob2.metric("SEGNO 2", f"{p2:.1f}%")
        col_ris, col_dc = st.columns(2)
        with col_ris:
            st.markdown("#### 🏆 RISULTATO 1X2")
            st.markdown(
                f'<div style="padding:16px 12px;border-radius:12px;border:2px solid #2e7d32;">'
                f'<div style="font-size:14px;opacity:.8;">Segno più probabile</div>'
                f'<div style="font-size:42px;font-weight:800;line-height:1.05;">{segno_top}</div>'
                f'<div style="font-size:18px;font-weight:600;">{probabilita_1x2[segno_top]:.1f}%</div>'
                f"</div>", unsafe_allow_html=True
            )
        with col_dc:
            st.markdown("#### 🛡️ DOPPIA CHANCE")
            st.markdown(
                f'<div style="padding:16px 12px;border-radius:12px;border:2px solid #1565c0;">'
                f'<div style="font-size:14px;opacity:.8;">Doppia chance più probabile</div>'
                f'<div style="font-size:42px;font-weight:800;line-height:1.05;">{doppia_chance_top}</div>'
                f'<div style="font-size:18px;font-weight:600;">{doppia_chance[doppia_chance_top]:.1f}%</div>'
                f"</div>", unsafe_allow_html=True
            )

        df_1x2 = crea_tabella({"1 (Casa)": p1, "X (Pareggio)": px, "2 (Trasferta)": p2}, "Segno")
        st.dataframe(df_1x2, use_container_width=True, hide_index=True)

        df_doppia_chance = crea_tabella(doppia_chance, "Doppia Chance")
        st.dataframe(df_doppia_chance, use_container_width=True, hide_index=True)

        quote, quote_ou_25 = None, None  # usate più sotto per la stima combo approssimata (Punto C)

        if not is_coppa:
            # =====================================================================
            # ✅ INDICATORE DI CONCORDANZA MODELLO/MERCATO — come scegliere le partite
            # Il backtest ha confermato: quando la previsione principale del modello
            # coincide col favorito del mercato, l'accuratezza sale (idea #1: segno,
            # +5/+10 punti; idea #1b: anche Over/Under 2.5, ulteriore miglioramento
            # in La Liga, Bundesliga, Ligue 1 — più debole ma comunque presente su
            # Serie A e Premier League). Due badge separati, stesso criterio già
            # validato nel backtest, applicato partita per partita.
            # =====================================================================
            colonne_h_pre, colonne_d_pre, colonne_a_pre = classifica_colonne_quote(dati.columns)
            quote_pre = quote_mercato_normalizzate(partita_sel, colonne_h_pre, colonne_d_pre, colonne_a_pre)
            if quote_pre:
                probabilita_1x2 = {"1": modello['prob_1'], "X": modello['prob_X'], "2": modello['prob_2']}
                previsione_modello = max(probabilita_1x2, key=probabilita_1x2.get)
                quote_per_segno_pre = {"1": quote_pre["q_casa_equa"], "X": quote_pre["q_x_equa"], "2": quote_pre["q_trasf_equa"]}
                favorito_mercato_pre = min(quote_per_segno_pre, key=quote_per_segno_pre.get)
                if previsione_modello == favorito_mercato_pre:
                    st.success(f"✅ **Segno — modello e mercato d'accordo** (entrambi favoriscono '{previsione_modello}') — "
                              f"nel backtest, su questo tipo di partite l'accuratezza è stata sensibilmente più alta.")
                else:
                    st.warning(f"⚠️ **Segno — disaccordo**: il modello preferisce '{previsione_modello}', il mercato "
                              f"favorisce '{favorito_mercato_pre}' — nel backtest, questo tipo di partite ha "
                              f"un'accuratezza più bassa. Trattala con più cautela.")

            colonne_over_pre, colonne_under_pre = classifica_colonne_over_under(dati.columns, "2.5")
            quote_ou_pre = quote_over_under_normalizzate(partita_sel, colonne_over_pre, colonne_under_pre)
            if quote_ou_pre:
                p_under_modello = modello['prob_under'][2.5]
                previsione_ou_modello = "Under" if p_under_modello >= 50 else "Over"
                favorito_ou_mercato_pre = "Over" if quote_ou_pre["q_over_equa"] < quote_ou_pre["q_under_equa"] else "Under"
                if previsione_ou_modello == favorito_ou_mercato_pre:
                    st.success(f"✅ **Over/Under 2.5 — modello e mercato d'accordo** (entrambi favoriscono "
                              f"'{previsione_ou_modello}') — segnale aggiuntivo utile soprattutto per le combo.")
                else:
                    st.warning(f"⚠️ **Over/Under 2.5 — disaccordo**: il modello preferisce '{previsione_ou_modello}', "
                              f"il mercato favorisce '{favorito_ou_mercato_pre}'.")

        if not is_coppa:
            st.markdown("### 💰 Controllo Value Bet (media multi-bookmaker, quote depurate dal margine)")
            colonne_h, colonne_d, colonne_a = classifica_colonne_quote(dati.columns)
            quote = quote_mercato_normalizzate(partita_sel, colonne_h, colonne_d, colonne_a)
            colonne_over, colonne_under = classifica_colonne_over_under(dati.columns, "2.5")
            quote_ou_25 = quote_over_under_normalizzate(partita_sel, colonne_over, colonne_under)
            if quote:
                ev_1 = (modello['prob_1'] / 100.0) * quote['q_casa_equa']
                ev_x = (modello['prob_X'] / 100.0) * quote['q_x_equa']
                ev_2 = (modello['prob_2'] / 100.0) * quote['q_trasf_equa']
                dati_ev = [
                    {"Segno": "1 (Casa)", "Quota Equa": f"{quote['q_casa_equa']:.2f}", "Valutazione": "🔥 ALTO VALORE" if ev_1 > 1.05 else ("📈 Leggero Valore" if ev_1 > 1.0 else "Nessun Valore")},
                    {"Segno": "X (Pareggio)", "Quota Equa": f"{quote['q_x_equa']:.2f}", "Valutazione": "🔥 ALTO VALORE" if ev_x > 1.05 else ("📈 Leggero Valore" if ev_x > 1.0 else "Nessun Valore")},
                    {"Segno": "2 (Trasferta)", "Quota Equa": f"{quote['q_trasf_equa']:.2f}", "Valutazione": "🔥 ALTO VALORE" if ev_2 > 1.05 else ("📈 Leggero Valore" if ev_2 > 1.0 else "Nessun Valore")},
                ]
                st.caption(f"Media su {quote['n_bookmakers']} bookmaker, margine rimosso: {(quote['overround']-1)*100:.1f}%")
                st.dataframe(pd.DataFrame(dati_ev), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Quote dei bookmaker non disponibili per questa specifica partita.")
        else:
            st.markdown("### 🎯 Quota Equa Statistica (Coppe Europee)")
            st.caption("Nessuna quota di mercato disponibile per le coppe: il modello mostra la quota equa "
                       "basata sulla probabilità pura. Cerca sui bookmaker quote superiori a questi valori.")
            q_fair_1 = 100.0 / modello['prob_1'] if modello['prob_1'] > 0 else 0.0
            q_fair_x = 100.0 / modello['prob_X'] if modello['prob_X'] > 0 else 0.0
            q_fair_2 = 100.0 / modello['prob_2'] if modello['prob_2'] > 0 else 0.0
            dati_fair = [
                {"Segno": "1 (Casa)", "Probabilità": f"{modello['prob_1']:.1f}%", "Quota Equa Minima": f"{q_fair_1:.2f}"},
                {"Segno": "X (Pareggio)", "Probabilità": f"{modello['prob_X']:.1f}%", "Quota Equa Minima": f"{q_fair_x:.2f}"},
                {"Segno": "2 (Trasferta)", "Probabilità": f"{modello['prob_2']:.1f}%", "Quota Equa Minima": f"{q_fair_2:.2f}"},
            ]
            st.dataframe(pd.DataFrame(dati_fair), use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ⚽ Goal / No Goal")
            st.dataframe(crea_tabella({"Goal": modello['prob_goal'], "No Goal": modello['prob_nogoal']}, "Opzione"),
                        use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("### 📉 Under / Over")
            oo_dict = {}
            for soglia, prob_u in modello['prob_under'].items():
                oo_dict[f"Under {soglia}"] = prob_u
                oo_dict[f"Over {soglia}"] = 100 - prob_u
            st.dataframe(crea_tabella(oo_dict, "Linea"), use_container_width=True, hide_index=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("### 🏠 Multigol Casa")
            st.dataframe(crea_tabella(modello['multigol_casa'], "Intervallo"), use_container_width=True, hide_index=True)
        with col_d:
            st.markdown("### ✈️ Multigol Ospite")
            st.dataframe(crea_tabella(modello['multigol_trasf'], "Intervallo"), use_container_width=True, hide_index=True)

        st.markdown("### 🔥 Combo Rapide")
        st.dataframe(crea_tabella(modello['combo'], "Combinazione"), use_container_width=True, hide_index=True)

        # =====================================================================
        # 🏆 TOP COMBO AUTOMATICHE
        # Scopre da sola le combinazioni più probabili tra tutte le 12
        # possibili (segno × Gol/No Gol × Over/Under 2.5), riusando la stessa
        # funzione calcola_combo_libera del motore libero qui sotto — nessuna
        # logica duplicata, un solo posto dove le combo vengono calcolate.
        # =====================================================================
        st.markdown("### 🏆 Top 4 Combo Automatiche (su Over/Under 2.5)")
        st.caption("Tutte le 12 combinazioni possibili (1/X/2 × Gol/No Gol × Over/Under 2.5), calibrate "
                   "se disponibile, filtrate per affidabilità (esclude combo troppo rare o su dati scarsi) "
                   "e ordinate per probabilità.")

        # Punto A — filtro di affidabilità: escludiamo le combo troppo rare
        # (rumore, non segnale) o calcolate su squadre con pochi dati specifici.
        SOGLIA_MIN_PROB_COMBO = 8.0  # sotto questa probabilità, troppo raro per essere utile
        dati_scarsi = bool(avvisi)

        tutte_le_combo = {}
        for segno_auto in ["1", "X", "2"]:
            for gg_auto in ["Goal", "NoGoal"]:
                for tipo_auto in ["Over", "Under"]:
                    prob_grezza = calcola_combo_libera(modello['griglia'], segno=segno_auto, soglia_gol=2.5,
                                                        tipo_soglia=tipo_auto, gol_nogol=gg_auto)
                    nome_chiave = f"{segno_auto}_{gg_auto}_{tipo_auto}"
                    prob_finale = prob_grezza
                    calib_combo_info = None
                    if not is_coppa and usa_calibrazione:
                        calib_combo_info = carica_calibratore(id_fd, nome_chiave)
                        if calib_combo_info:
                            prob_finale = float(calib_combo_info["calibratore"].predict([prob_grezza/100])[0]) * 100
                    chiave_vis = f"{segno_auto} + {'Gol' if gg_auto=='Goal' else 'No Gol'} + {tipo_auto} 2.5"
                    tutte_le_combo[chiave_vis] = {
                        "prob": prob_finale, "segno": segno_auto, "tipo": tipo_auto,
                        "calibrata": calib_combo_info is not None,
                    }

        combo_affidabili = {k: v for k, v in tutte_le_combo.items() if v["prob"] >= SOGLIA_MIN_PROB_COMBO}
        top4_combo = dict(sorted(combo_affidabili.items(), key=lambda x: x[1]["prob"], reverse=True)[:4])

        # Etichetta esplicita sulla Top1 RAW usata dal freeze V15; il ranking delle
        # 4 combo rimane quello già presente nell'interfaccia.
        if v15_top1_info is not None and v15_top1_info.get('selected'):
            st.caption(
                f"🧊 Freeze V15: Top1 RAW {100.0*v15_top1_info['top1_raw']:.1f}% ≥ 35% · "
                f"probabilità operativa Top1 = {100.0*v15_top1_info['platt_prob']:.1f}%"
                if v15_top1_info.get('platt_applied') else
                f"🧊 Freeze V15: Top1 RAW {100.0*v15_top1_info['top1_raw']:.1f}% ≥ 35% · PLATT non applicato ({v15_top1_info.get('reason','')})"
            )

        if dati_scarsi:
            st.warning("⚠️ Combo calcolate su dati limitati per una delle due squadre (vedi avviso sopra) "
                      "— trattale con più cautela del solito.")
        if not top4_combo:
            st.info(f"Nessuna combo sopra la soglia minima di affidabilità ({SOGLIA_MIN_PROB_COMBO}%) "
                   "per questa partita.")
        else:
            righe_top4 = []
            for chiave, info_c in top4_combo.items():
                q_stimata, non_prezzate = stima_quota_combo_approssimata(info_c["segno"], 2.5, info_c["tipo"], quote, quote_ou_25)
                righe_top4.append({
                    "Combo": chiave + (" 🎯" if info_c["calibrata"] else ""),
                    "Probabilità (%)": f"{info_c['prob']:.1f}%",
                    "Quota stimata*": f"~{q_stimata:.2f}" if q_stimata else "n/d",
                })
            st.dataframe(pd.DataFrame(righe_top4), use_container_width=True, hide_index=True)
            st.caption("🎯 = probabilità corretta con calibrazione specifica per questa combo. "
                       "*Quota STIMATA per approssimazione (1X2 × Over/Under come se fossero indipendenti — "
                       "non lo sono del tutto). Il componente Gol/No Gol non ha una quota nel file, quindi "
                       "non entra nella stima: il numero reale del bookmaker sarà diverso.")

        st.divider()
        st.markdown("### ⚔️ Ultimi Scontri Diretti (H2H)")
        df_h2h = estrai_scontri_diretti(partita_sel['HomeTeam'], partita_sel['AwayTeam'], dati_filtrati, df_globale_filtrato)
        if not df_h2h.empty:
            cols_mostra = [c for c in ['Date', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam'] if c in df_h2h.columns]
            st.dataframe(df_h2h[cols_mostra], use_container_width=True, hide_index=True)
        else:
            st.info("Nessun precedente recente trovato negli archivi disponibili tra queste due squadre.")

        st.divider()
        st.markdown("### 🎯 Statistiche Match (Stimate)")
        nota_stima = ""
        if avvisi:
            nota_stima = " ⚠️ (basate parzialmente sulla media di lega per scarsità di dati specifici)"
        st.info(f"🚩 Angoli Totali Stimati: **{modello['angoli_stimati']}** | "
                f"🎯 Tiri in Porta Totali Stimati: **{modello['tiri_stimati']}**{nota_stima}")
# =====================================================================
# 🧊 V15 — OPERATIVE FREEZE SUMMARY
# Solo riepilogo della regola attiva; non riottimizza e non altera i parametri.
# =====================================================================
def mostra_v15_operational_freeze_summary():
    st.divider()
    st.markdown('## 🧊 V15 — REGOLA OPERATIVA CONGELATA')
    st.info(
        'Top1 Combo RAW ≥ 35% → PLATT walk-forward = 100 osservazioni precedenti '
        'dello stesso campionato, con salvaguardia monotona (pendenza > 0). '
        'La calibrazione modifica la probabilità della Top1, non il ranking delle altre combo. '
        'Nessuna informazione futura.'
    )


mostra_v15_operational_freeze_summary()
