import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import time
import os
import pickle
import re
import unicodedata
from functools import lru_cache
from datetime import datetime, date
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

st.set_page_config(page_title="COMBO - Advanced Betting Model", page_icon="⚽", layout="centered")

# "code_api" = codice della stessa competizione su football-data.org, usato
# solo come fonte di RISERVA quando football-data.co.uk non risponde.
CAMPIONATI_DOMESTICI = {
    "Italia - Serie A": {"id_fd": "I1", "code_api": "SA"},
    "Inghilterra - Premier League": {"id_fd": "E0", "code_api": "PL"},
    "Spagna - La Liga": {"id_fd": "SP1", "code_api": "PD"},
    "Germania - Bundesliga": {"id_fd": "D1", "code_api": "BL1"},
    "Francia - Ligue 1": {"id_fd": "F1", "code_api": "FL1"},
}
CODICE_API_PER_ID_FD = {v["id_fd"]: v["code_api"] for v in CAMPIONATI_DOMESTICI.values()}

CAMPIONATI_COPPE = {
    "🌍 UEFA Champions League": {"code": "CL", "gratis_confermato": True},
    "🌍 UEFA Europa League": {"code": "EL", "gratis_confermato": False},
    "🌍 UEFA Conference League": {"code": "UECL", "gratis_confermato": False},
}

HEADERS_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

K_SHRINKAGE = 10  # stesso principio già validato nell'App Risultati Fissi


# =====================================================================
# 🔧 MATCHING SQUADRE — versione 2 (sostituisce il vecchio FIX #4)
# Il confronto precedente ("una stringa contenuta nell'altra") falliva sui
# club con nomi molto diversi tra le due fonti ("Manchester United FC" contro
# "Man United", "Paris Saint-Germain FC" contro "Paris SG") e confondeva
# club diversi ("Milan" dentro "Internazionale Milano", "Paris" dentro
# "Paris SG"). Ora:
#  1. i nomi vengono normalizzati (accenti, punteggiatura, sigle tipo FC/AC);
#  2. i club più comuni stanno in una tabella di alias: due nomi corrispondono
#     solo se puntano allo STESSO club, mai per sottostringa;
#  3. per i nomi fuori tabella si accetta solo l'inclusione di PAROLE intere
#     ("Newcastle" dentro "Newcastle United"), mai di pezzi di parola, e mai
#     tra un club in tabella e uno fuori tabella.
# Se una squadra non viene riconosciuta e non ha storico, l'app lo dice
# (vedi avviso "nessuna partita trovata" più sotto).
# Per aggiungere un club: una riga in ALIAS_SQUADRE, con TUTTE le grafie che
# compaiono nelle due fonti (football-data.co.uk e football-data.org).
# =====================================================================
_PAROLE_IGNORATE = {"fc", "cf", "afc", "ac", "acf", "sc", "cfc", "ssc", "as", "us", "ss", "rc", "rcd",
                    "ud", "cd", "ca", "sd", "sv", "vfb", "vfl", "tsg", "fsv", "bsc", "fk", "sk", "nk",
                    "bc", "calcio", "de", "di", "del", "the"}

ALIAS_SQUADRE = {
    # --- Inghilterra
    "arsenal": ["Arsenal"], "aston villa": ["Aston Villa"], "bournemouth": ["Bournemouth", "AFC Bournemouth"],
    "brentford": ["Brentford"], "brighton": ["Brighton", "Brighton & Hove Albion"], "burnley": ["Burnley"],
    "chelsea": ["Chelsea"], "crystal palace": ["Crystal Palace"], "everton": ["Everton"], "fulham": ["Fulham"],
    "leeds": ["Leeds", "Leeds United"], "liverpool": ["Liverpool"],
    "man city": ["Man City", "Manchester City"],
    "man united": ["Man United", "Man Utd", "Manchester United", "Manchester Utd"],
    "newcastle": ["Newcastle", "Newcastle United", "Newcastle Utd"],
    "nottm forest": ["Nott'm Forest", "Nottm Forest", "Nottingham Forest"],
    "sunderland": ["Sunderland"], "tottenham": ["Tottenham", "Tottenham Hotspur", "Spurs"],
    "west ham": ["West Ham", "West Ham United"], "wolves": ["Wolves", "Wolverhampton Wanderers", "Wolverhampton"],
    "leicester": ["Leicester", "Leicester City"], "southampton": ["Southampton"],
    "ipswich": ["Ipswich", "Ipswich Town"], "west brom": ["West Brom", "West Bromwich Albion"],
    "sheffield united": ["Sheffield United", "Sheffield Utd", "Sheff Utd"],
    "sheffield wednesday": ["Sheffield Weds", "Sheffield Wednesday", "Sheff Wed"],
    "luton": ["Luton", "Luton Town"], "norwich": ["Norwich", "Norwich City"],
    # --- Spagna
    "alaves": ["Alaves", "Deportivo Alaves"],
    "ath bilbao": ["Ath Bilbao", "Athletic Bilbao", "Athletic Club", "Athletic Club Bilbao"],
    "ath madrid": ["Ath Madrid", "Atletico Madrid", "Atletico de Madrid", "Club Atletico de Madrid", "Atl Madrid"],
    "barcelona": ["Barcelona", "FC Barcelona"], "betis": ["Betis", "Real Betis", "Real Betis Balompie"],
    "celta": ["Celta", "Celta Vigo", "RC Celta de Vigo"], "elche": ["Elche"],
    "espanol": ["Espanol", "Espanyol", "RCD Espanyol de Barcelona", "Espanyol Barcelona"],
    "getafe": ["Getafe"], "girona": ["Girona"], "levante": ["Levante", "Levante UD"],
    "mallorca": ["Mallorca", "RCD Mallorca"], "osasuna": ["Osasuna", "CA Osasuna"],
    "oviedo": ["Oviedo", "Real Oviedo"], "real madrid": ["Real Madrid"], "sevilla": ["Sevilla"],
    "sociedad": ["Sociedad", "Real Sociedad", "Real Sociedad de Futbol"], "valencia": ["Valencia"],
    "vallecano": ["Vallecano", "Rayo Vallecano", "Rayo Vallecano de Madrid"], "villarreal": ["Villarreal"],
    "las palmas": ["Las Palmas", "UD Las Palmas"], "leganes": ["Leganes", "CD Leganes"],
    "valladolid": ["Valladolid", "Real Valladolid"],
    # --- Italia
    "atalanta": ["Atalanta", "Atalanta BC"], "bologna": ["Bologna", "Bologna FC 1909"],
    "cagliari": ["Cagliari", "Cagliari Calcio"], "como": ["Como", "Como 1907"],
    "cremonese": ["Cremonese", "US Cremonese"], "fiorentina": ["Fiorentina", "ACF Fiorentina"],
    "genoa": ["Genoa", "Genoa CFC"],
    "inter": ["Inter", "Internazionale", "FC Internazionale Milano", "Inter Milan"],
    "juventus": ["Juventus", "Juventus FC"], "lazio": ["Lazio", "SS Lazio"], "lecce": ["Lecce", "US Lecce"],
    "milan": ["Milan", "AC Milan"], "napoli": ["Napoli", "SSC Napoli"],
    "parma": ["Parma", "Parma Calcio 1913"], "pisa": ["Pisa", "AC Pisa 1909"], "roma": ["Roma", "AS Roma"],
    "sassuolo": ["Sassuolo", "US Sassuolo Calcio"], "torino": ["Torino", "Torino FC"],
    "udinese": ["Udinese", "Udinese Calcio"], "verona": ["Verona", "Hellas Verona"],
    # --- Francia
    "angers": ["Angers", "Angers SCO"], "auxerre": ["Auxerre", "AJ Auxerre"],
    "brest": ["Brest", "Stade Brestois", "Stade Brestois 29"], "le havre": ["Le Havre", "Le Havre AC"],
    "lens": ["Lens", "RC Lens", "Racing Club de Lens"], "lille": ["Lille", "Lille OSC", "LOSC Lille"],
    "lorient": ["Lorient", "FC Lorient"], "lyon": ["Lyon", "Olympique Lyonnais", "Olympique Lyon"],
    "marseille": ["Marseille", "Olympique de Marseille", "Olympique Marseille"],
    "metz": ["Metz", "FC Metz"], "monaco": ["Monaco", "AS Monaco", "AS Monaco FC"],
    "nantes": ["Nantes", "FC Nantes"], "nice": ["Nice", "OGC Nice"],
    "paris fc": ["Paris FC"],   # attenzione: diverso dal PSG ("Paris" da solo = Paris FC)
    "paris sg": ["Paris SG", "Paris Saint-Germain", "Paris Saint Germain", "PSG"],
    "rennes": ["Rennes", "Stade Rennais", "Stade Rennais FC 1901"],
    "strasbourg": ["Strasbourg", "RC Strasbourg", "RC Strasbourg Alsace"],
    "toulouse": ["Toulouse", "Toulouse FC"],
    "st etienne": ["St Etienne", "Saint-Etienne", "AS Saint-Etienne"],
    # --- Germania
    "augsburg": ["Augsburg", "FC Augsburg"],
    "bayern munich": ["Bayern Munich", "Bayern Munchen", "FC Bayern Munchen", "Bayern"],
    "dortmund": ["Dortmund", "Borussia Dortmund"],
    "ein frankfurt": ["Ein Frankfurt", "Eintracht Frankfurt", "Frankfurt"],
    "freiburg": ["Freiburg", "SC Freiburg"], "hamburg": ["Hamburg", "Hamburger SV"],
    "heidenheim": ["Heidenheim", "1. FC Heidenheim 1846"], "hoffenheim": ["Hoffenheim", "TSG 1899 Hoffenheim"],
    "koln": ["Koln", "FC Koln", "1. FC Köln"],
    "leverkusen": ["Leverkusen", "Bayer Leverkusen", "Bayer 04 Leverkusen"],
    "mgladbach": ["M'gladbach", "Monchengladbach", "Borussia Mönchengladbach", "Gladbach"],
    "mainz": ["Mainz", "Mainz 05", "1. FSV Mainz 05"], "rb leipzig": ["RB Leipzig", "Leipzig"],
    "st pauli": ["St Pauli", "FC St. Pauli", "FC St. Pauli 1910"], "stuttgart": ["Stuttgart", "VfB Stuttgart"],
    "union berlin": ["Union Berlin", "1. FC Union Berlin"],
    "werder bremen": ["Werder Bremen", "SV Werder Bremen", "Bremen"],
    "wolfsburg": ["Wolfsburg", "VfL Wolfsburg"], "bochum": ["Bochum", "VfL Bochum", "VfL Bochum 1848"],
    "holstein kiel": ["Holstein Kiel", "KSV Holstein Kiel"], "schalke": ["Schalke 04", "FC Schalke 04"],
    "hertha": ["Hertha", "Hertha Berlin", "Hertha BSC"], "darmstadt": ["Darmstadt", "SV Darmstadt 98"],
}


@lru_cache(maxsize=None)
def _token_nome(nome):
    """Nome -> tupla di parole normalizzate: senza accenti, minuscole, senza punteggiatura,
    senza sigle societarie (FC, AC, SSC...) e senza numeri (anni di fondazione)."""
    t = unicodedata.normalize("NFKD", str(nome))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return tuple(p for p in t.split() if p and not p.isdigit() and p not in _PAROLE_IGNORATE)


def _costruisci_alias_inverso():
    inverso = {}
    for canonico, varianti in ALIAS_SQUADRE.items():
        for v in [canonico] + list(varianti):
            chiave = _token_nome(v)
            if chiave and chiave not in inverso:
                inverso[chiave] = canonico
    return inverso


_ALIAS_INVERSO = _costruisci_alias_inverso()


@lru_cache(maxsize=None)
def chiave_squadra(nome):
    """Nome canonico del club se è in tabella, altrimenti None."""
    return _ALIAS_INVERSO.get(_token_nome(nome))


@lru_cache(maxsize=None)
def nomi_corrispondono(a, b):
    ka, kb = chiave_squadra(a), chiave_squadra(b)
    if ka is not None and kb is not None:
        return ka == kb                      # entrambi noti: stesso club o niente
    ta, tb = _token_nome(a), _token_nome(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    if ka is not None or kb is not None:
        return False                         # uno noto e uno no: solo uguaglianza esatta
    sa, sb = set(ta), set(tb)
    return sa <= sb or sb <= sa              # entrambi ignoti: parole intere, mai pezzi di parola


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
#
# 🔧 CATENA DI RISERVA (nuova). Ordine di tentativi per i campionati:
#   1. football-data.co.uk online  → dati completi (quote, tiri, corner)
#   2. ultima copia valida salvata su disco → dati completi ma datati
#   3. football-data.org (serve la chiave API) → MODALITÀ RIDOTTA: solo
#      risultati della stagione corrente, senza quote/tiri/corner
# L'app dice sempre quale fonte sta usando (colonna "FonteDati").
# Nota: su Streamlit Community Cloud il disco si azzera a ogni riavvio/
# redeploy, quindi la copia (punto 2) può non esserci: per questo esiste
# anche il punto 3.
# =====================================================================
CARTELLA_COPIE = "copie_dati_combo"   # stessa logica "best effort" dei calibratori (.pkl)
FONTE_RIDOTTA = "football-data.org (modalità ridotta)"


def _path_copia(url):
    nome = re.sub(r"[^A-Za-z0-9]+", "_", url.replace("://www.", "://").split("://", 1)[-1]).strip("_")
    return os.path.join(CARTELLA_COPIE, nome + ".csv")


def _salva_copia(url, testo):
    """Salva l'ultimo CSV scaricato correttamente. Mai bloccante: se il disco non
    è scrivibile, semplicemente non ci sarà copia."""
    try:
        os.makedirs(CARTELLA_COPIE, exist_ok=True)
        path = _path_copia(url)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(testo)
        os.replace(tmp, path)   # scrittura atomica: mai una copia a metà
    except Exception:
        pass


def _leggi_copia(url):
    path = _path_copia(url)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            testo = f.read()
        df = pd.read_csv(io.StringIO(testo))
        df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
        return df, datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None, None


def scarica_csv_robusto(url, tentativi=3, attesa_secondi=2, colonne_attese=("HomeTeam", "AwayTeam")):
    """FIX #11 — BOM (Byte Order Mark): fixtures.csv inizia con un carattere
    invisibile Unicode che, se non gestito, si attacca al nome della prima
    colonna ("Div" diventa "\\ufeffDiv"), facendo fallire in silenzio ogni
    controllo tipo "if 'Div' in colonne" — nessuna fixture futura veniva mai
    trovata, su nessun campionato, per questo motivo esatto. 'utf-8-sig' lo
    rimuove in fase di decodifica; non fa danno se il file non ha il BOM.

    Ritorna (df, errore). Se il download online riesce, df.attrs["fonte"] = "online" e
    l'ultimo file valido viene salvato su disco. Se fallisce ma esiste una copia
    salvata, ritorna la copia (df.attrs["fonte"] = "copia salvata il ...") e
    l'errore online. Solo se non c'è nemmeno la copia ritorna (None, errore).
    `colonne_attese` evita di scambiare per dati una pagina HTML d'errore."""
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
                if colonne_attese and not set(colonne_attese).issubset(df.columns):
                    raise ValueError("file scaricato non valido (colonne attese mancanti)")
                _salva_copia(url, testo)
                df.attrs["fonte"] = "online"
                return df, None
            except Exception as e:
                ultimo_errore = str(e)
        if tentativo < tentativi - 1:
            time.sleep(attesa_secondi)

    copia, quando = _leggi_copia(url)
    if copia is not None and (not colonne_attese or set(colonne_attese).issubset(copia.columns)):
        copia.attrs["fonte"] = f"copia salvata il {quando:%d/%m/%Y %H:%M}"
        return copia, ultimo_errore
    return None, ultimo_errore


def _da_api_come_stagione_corrente(id_fd, api_key):
    """Riserva football-data.org per un campionato domestico: DataFrame con lo stesso
    schema minimo del CSV (Date, HomeTeam, AwayTeam, FTHG, FTAG) o None."""
    codice = CODICE_API_PER_ID_FD.get(id_fd)
    if not api_key or not codice:
        return None
    da_api = carica_dati_api_europee(codice, api_key)
    if not isinstance(da_api, pd.DataFrame) or len(da_api) == 0:
        return None
    da_api = da_api.copy()
    # stesso formato data del CSV (gg/mm/aaaa): se mescolassimo ISO e gg/mm/aaaa nella
    # stessa colonna, pandas scarterebbe in silenzio le righe del formato "minoritario"
    da_api['Date'] = da_api['Date_parsed'].dt.strftime('%d/%m/%Y')
    return da_api


@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_campionato(id_fd, api_key=None):
    """FIX B — aggiunto il tag 'Stagione' (precedente/corrente), necessario
    per poter validare il value bet SOLO sui risultati reali più recenti,
    non su quelli già usati per calibrare le medie di lega.
    Colonna 'FonteDati': "online", "copia salvata il ..." o FONTE_RIDOTTA."""
    codice_corrente, codice_precedente = codici_stagione()
    frames = []
    ha_corrente = False
    for codice, label in [(codice_precedente, 'precedente'), (codice_corrente, 'corrente')]:
        url = f"https://football-data.co.uk/mmz4281/{codice}/{id_fd}.csv"
        df, _ = scarica_csv_robusto(url)
        if df is not None:
            df.columns = df.columns.str.strip()
            df['Stagione'] = label
            df['FonteDati'] = df.attrs.get("fonte", "online")
            frames.append(df)
            ha_corrente = ha_corrente or label == 'corrente'

    # Riserva: manca la stagione corrente (né online né in copia) → football-data.org.
    # Si usa solo se ha almeno una partita già giocata (a inizio stagione, prima della
    # prima giornata, il file .co.uk può non esistere ancora: non è un guasto).
    if not ha_corrente:
        da_api = _da_api_come_stagione_corrente(id_fd, api_key)
        if da_api is not None and (da_api['Status'] == 'FINISHED').any():
            da_api['Stagione'] = 'corrente'
            da_api['FonteDati'] = FONTE_RIDOTTA
            frames.append(da_api)

    if frames:
        dati = pd.concat(frames, ignore_index=True, sort=False)
        dati['Date_parsed'] = pd.to_datetime(dati['Date'], errors='coerce', dayfirst=True)
        return dati.dropna(subset=['Date_parsed']).sort_values('Date_parsed').reset_index(drop=True)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def carica_tutti_i_campionati(api_key=None):
    tutti_dati = []
    for c_info in CAMPIONATI_DOMESTICI.values():
        df = carica_dati_campionato(c_info["id_fd"], api_key)
        if df is not None:
            tutti_dati.append(df)
    if tutti_dati:
        return pd.concat(tutti_dati, ignore_index=True, sort=False)
    return pd.DataFrame()


def fonti_non_principali(*dataframes):
    """Insieme delle fonti diverse da "online" presenti nei DataFrame (copia salvata,
    modalità ridotta). Vuoto = tutto arriva dalla fonte principale."""
    fonti = set()
    for d in dataframes:
        if d is not None and hasattr(d, "columns") and 'FonteDati' in d.columns and len(d) > 0:
            fonti.update(str(x) for x in d['FonteDati'].dropna().unique())
    fonti.discard("online")
    return fonti


def avviso_fonte_dati(*dataframes):
    """Mostra a schermo un avviso per ogni fonte non principale; ritorna l'insieme di fonti."""
    fonti = fonti_non_principali(*dataframes)
    for f in sorted(fonti):
        if f == FONTE_RIDOTTA:
            st.warning("⚠️ **Modalità ridotta**: football-data.co.uk non è raggiungibile e non c'è una copia salvata, "
                       "quindi uso football-data.org (solo stagione corrente). Mancano quote, tiri e corner: "
                       "value bet e statistiche angoli/tiri non sono disponibili, e lo storico è più corto.")
        else:
            st.warning(f"⚠️ football-data.co.uk non risponde: sto usando una **{f}**. "
                       "Le partite più recenti potrebbero mancare. Prova «Riprova a scaricare i dati» nella barra laterale.")
    return fonti


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
    for gc in range(8):
        for gt in range(8):
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
        if segno and r["segno"] not in segno:      # segno = "1", "X", "2" oppure doppia chance "1X", "X2", "12"
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
def carica_fixture_future(id_fd, api_key=None):
    df, _ = scarica_csv_robusto("https://football-data.co.uk/fixtures.csv")
    if df is not None:
        fonte = df.attrs.get("fonte", "online")
        fx = df.copy()
        fx.columns = fx.columns.str.strip()
        if 'Div' in fx.columns:
            fx = fx[fx['Div'] == id_fd].copy()
            fx['Date_parsed'] = pd.to_datetime(fx['Date'], errors='coerce', dayfirst=True)
            oggi = pd.Timestamp(date.today())
            fx = fx[fx['Date_parsed'] >= oggi].sort_values('Date_parsed').reset_index(drop=True)
            fx['FonteDati'] = fonte
            return fx
        return pd.DataFrame()

    # Né online né copia salvata: calendario dalla riserva football-data.org (solo squadre e date)
    da_api = _da_api_come_stagione_corrente(id_fd, api_key)
    if da_api is not None:
        fx = da_api[da_api['Status'].isin(['SCHEDULED', 'TIMED'])].copy()
        fx = fx[fx['Date_parsed'] >= pd.Timestamp(date.today())]
        fx['FonteDati'] = FONTE_RIDOTTA
        return fx.sort_values('Date_parsed').reset_index(drop=True)
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
            # doppia chance (es. "1X"): quota equa = 1 / (somma delle probabilità eque dei due esiti)
            fattori.append(1.0 / sum(1.0 / mappa[sg] for sg in segno))
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
# 🔧 PUNTO B — CALIBRAZIONE POST-HOC (1X2 + le 36 combo automatiche a DUE gambe)
# Stessa tecnica isotonic regression già validata nell'App Risultati Fissi.
# Un correttore per 1X2 (corregge "Esito Finale" e il Value Bet), più uno
# separato per ciascuna delle 36 combo: un ESITO (1, X, 2 oppure la doppia
# chance 1X, X2, 12) abbinato a UNA condizione tra Gol, No Gol, Over 2.5,
# Under 2.5, Over 1.5, Under 3.5 (le ultime due, meno 50/50, hanno probabilità
# più alte: servono a non avere combo che sbagliano quasi sempre per natura).
# Le combo a tre gambe usate in precedenza sono state tolte: i
# loro vecchi file .pkl non vengono più letti. La verità per allenarle è già
# nei risultati reali che scarichiamo (nessuna fonte dati nuova serve). Le
# combo libere con soglie diverse da 2.5 restano SENZA calibrazione.
# =====================================================================
ESITI_SINGOLI = ["1", "X", "2"]
ESITI_DOPPIA = ["1X", "X2", "12"]          # 12 = "non pareggia"
GAMBE_GOL = ["Goal", "NoGoal", "Over", "Under", "Over15", "Under35"]   # Over/Under = linea 2.5; Over15 = Over 1.5; Under35 = Under 3.5
LINEE_GOL = {"Over": (2.5, "Over"), "Under": (2.5, "Under"), "Over15": (1.5, "Over"), "Under35": (3.5, "Under")}
CATEGORIE_COMBO = {"singolo": ESITI_SINGOLI, "doppia": ESITI_DOPPIA}
ETICHETTE_CATEGORIA = {"singolo": "Esito singolo (1/X/2)", "doppia": "Doppia chance (1X/X2/12)"}
LE_COMBO = [(e, g) for e in ESITI_SINGOLI + ESITI_DOPPIA for g in GAMBE_GOL]


def chiave_combo(esito, gamba):
    return f"{esito}_{gamba}"


def etichetta_combo(esito, gamba):
    nome = {"Goal": "Gol", "NoGoal": "No Gol", "Over": "Over 2.5", "Under": "Under 2.5",
            "Over15": "Over 1.5", "Under35": "Under 3.5"}[gamba]
    return f"{esito} + {nome}"


def calcola_combo_2_gambe(griglia, esito, gamba):
    """Probabilità (%) di: esito (anche doppia chance, es. "1X") + una condizione sui gol."""
    if gamba in ("Goal", "NoGoal"):
        return calcola_combo_libera(griglia, segno=esito, gol_nogol=gamba)
    soglia, tipo = LINEE_GOL[gamba]
    return calcola_combo_libera(griglia, segno=esito, soglia_gol=soglia, tipo_soglia=tipo)


def combo_avverata(esito, gamba, fthg, ftag):
    """True se la combo si è avverata con il risultato reale fthg-ftag."""
    reale = '1' if fthg > ftag else ('2' if fthg < ftag else 'X')
    if reale not in esito:
        return False
    if gamba == "Goal":
        return fthg > 0 and ftag > 0
    if gamba == "NoGoal":
        return not (fthg > 0 and ftag > 0)
    soglia, tipo = LINEE_GOL[gamba]
    return (fthg + ftag) > soglia if tipo == "Over" else (fthg + ftag) < soglia


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


def carica_calibratori_combo(id_fd):
    """Carica una sola volta i calibratori delle 36 combo: {chiave: info o None}."""
    return {chiave_combo(e, g): carica_calibratore(id_fd, chiave_combo(e, g)) for (e, g) in LE_COMBO}


def probabilita_tutte_le_combo(griglia, calibratori=None):
    """{(esito, gamba): (probabilità %, calibrata sì/no)} per le 36 combo."""
    out = {}
    for (e, g) in LE_COMBO:
        prob = calcola_combo_2_gambe(griglia, e, g)
        calibrata = False
        info = calibratori.get(chiave_combo(e, g)) if calibratori else None
        if info:
            prob = float(info["calibratore"].predict([prob / 100])[0]) * 100
            calibrata = True
        out[(e, g)] = (prob, calibrata)
    return out


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
    per tutte e 36 le combo insieme — efficiente, una sola passata."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    oss_1x2 = []
    oss_combo = {c: [] for c in LE_COMBO}

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

        for (e_c, g_c) in LE_COMBO:
            prob_c = calcola_combo_2_gambe(m['griglia'], e_c, g_c)
            oss_combo[(e_c, g_c)].append((prob_c/100, combo_avverata(e_c, g_c, partita['FTHG'], partita['FTAG'])))

    risultati = {"1x2": {"n": len(oss_1x2), "ok": False}}
    cal_1x2 = _allena_isotonic(oss_1x2)
    if cal_1x2:
        _salva_calibratore(id_fd, "1x2", cal_1x2, len(oss_1x2))
        risultati["1x2"]["ok"] = True

    for (e_c, g_c), oss_c in oss_combo.items():
        nome_chiave = chiave_combo(e_c, g_c)
        cal_c = _allena_isotonic(oss_c)
        risultati[nome_chiave] = {"n": len(oss_c), "ok": cal_c is not None}
        if cal_c:
            _salva_calibratore(id_fd, nome_chiave, cal_c, len(oss_c))

    return risultati


# =====================================================================
# 🔧 BACKTEST COMBO — stessa logica del backtest 1X2 senza filtro, applicata
# alle combo a due gambe. Per ogni partita si valutano DUE previsioni nella
# stessa passata (il modello viene calcolato una volta sola): la combo più
# probabile tra quelle con esito singolo (1/X/2) e quella tra le doppie
# chance (1X/X2/12). Usa la calibrazione per-combo se disponibile e attiva,
# e la stessa verità (risultato reale) già usata per allenarle.
# =====================================================================
def verdetto_calibrazione(n, corrette, prob_media):
    """Confronta la frequenza reale con la probabilità media prevista (prob_media tra 0 e 1).
    Ritorna (scarto in punti %, margine statistico in punti %, testo) oppure None.
    Il margine è 2 errori standard: sotto quella soglia lo scarto può essere solo fortuna."""
    if not n:
        return None
    freq = corrette / n
    errore_standard = (prob_media * (1 - prob_media) / n) ** 0.5
    scarto = (freq - prob_media) * 100
    margine = 2 * errore_standard * 100
    if abs(scarto) <= margine:
        testo = "in linea con le probabilità previste (entro l'errore statistico)"
    elif scarto < 0:
        testo = "il modello è TROPPO SICURO: queste combo escono meno di quanto previsto"
    else:
        testo = "il modello è PRUDENTE: queste combo escono più di quanto previsto"
    if n < 100:
        testo += " — campione piccolo, indicativo"
    return scarto, margine, testo


def esegui_backtest_combo(dati_completi, rho, ewma_span, emivita, usa_oos, id_fd=None,
                           usa_calibrazione=False, richiedi_accordo_mercato=False, richiedi_accordo_ou=False):
    """Ritorna None, oppure {"singolo": r, "doppia": r} con r = n_partite, n_corrette,
    per_combo e i conteggi delle partite scartate dai filtri.

    richiedi_accordo_mercato: valuta solo le combo il cui ESITO è compatibile col
    favorito del mercato (per la doppia chance basta che il favorito sia uno dei due
    esiti coperti; idea #1, già validata su 1X2).
    richiedi_accordo_ou (idea #1b, DA VERIFICARE): in aggiunta, quando la combo
    prevista contiene Over/Under 2.5, richiede l'accordo con la quota di mercato
    Over/Under 2.5. Le combo con Gol/No Gol, Over 1.5 o Under 3.5 non hanno una
    quota di mercato nei file che usiamo: su di esse questo filtro non può agire e
    la partita passa.

    Diagnostica di calibrazione: per ogni categoria e per ogni combo si accumula
    "somma_prob", cioè la somma delle probabilità previste per le combo scelte. Divisa
    per le volte in cui la combo è stata scelta, dà la probabilità media prevista, da
    confrontare con la frequenza reale (vedi verdetto_calibrazione)."""
    tutte = dati_completi[dati_completi['FTHG'].notna()].reset_index(drop=True)
    ha_stagione = 'Stagione' in tutte.columns

    if usa_oos and ha_stagione:
        indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
    else:
        indici = list(range(15, len(tutte)))

    if not indici:
        return None

    # I 36 calibratori si leggono UNA volta, non a ogni partita.
    calibratori = carica_calibratori_combo(id_fd) if (usa_calibrazione and id_fd) else None
    colonne_h, colonne_d, colonne_a = classifica_colonne_quote(tutte.columns)
    colonne_over, colonne_under = classifica_colonne_over_under(tutte.columns, "2.5")
    servono_quote = richiedi_accordo_mercato or richiedi_accordo_ou

    risultati = {
        cat: {"n_partite": 0, "n_corrette": 0, "somma_prob": 0.0, "n_scartate_disaccordo": 0,
              "n_scartate_disaccordo_ou": 0, "n_scartate_no_quote": 0,
              "per_combo": {chiave_combo(e, g): {"previste": 0, "corrette": 0, "somma_prob": 0.0} for e in esiti for g in GAMBE_GOL}}
        for cat, esiti in CATEGORIE_COMBO.items()
    }

    for i in indici:
        partita = tutte.iloc[i]
        prec = tutte.iloc[:i]
        m = calcola_modello_completo(prec, partita['HomeTeam'], partita['AwayTeam'], rho, ewma_span,
                                      emivita, pd.DataFrame(), data_riferimento=partita.get('Date_parsed'))
        if m is None: continue

        probs = probabilita_tutte_le_combo(m['griglia'], calibratori)
        quote = quote_mercato_normalizzate(partita, colonne_h, colonne_d, colonne_a) if servono_quote else None
        quote_ou = quote_over_under_normalizzate(partita, colonne_over, colonne_under) if richiedi_accordo_ou else None

        for cat, esiti in CATEGORIE_COMBO.items():
            r = risultati[cat]
            candidate = {(e, g): probs[(e, g)][0] for e in esiti for g in GAMBE_GOL}
            e_p, g_p = max(candidate, key=candidate.get)

            if servono_quote:
                if quote is None:
                    r["n_scartate_no_quote"] += 1
                    continue
                if richiedi_accordo_mercato:
                    quote_per_segno = {"1": quote["q_casa_equa"], "X": quote["q_x_equa"], "2": quote["q_trasf_equa"]}
                    favorito_mercato = min(quote_per_segno, key=quote_per_segno.get)
                    if favorito_mercato not in e_p:
                        r["n_scartate_disaccordo"] += 1
                        continue
                if richiedi_accordo_ou and g_p in ("Over", "Under"):
                    if quote_ou is None:
                        r["n_scartate_no_quote"] += 1
                        continue
                    favorito_ou_mercato = "Over" if quote_ou["q_over_equa"] < quote_ou["q_under_equa"] else "Under"
                    if g_p != favorito_ou_mercato:
                        r["n_scartate_disaccordo_ou"] += 1
                        continue

            avverata = combo_avverata(e_p, g_p, partita['FTHG'], partita['FTAG'])
            prob_prevista = candidate[(e_p, g_p)] / 100
            r["n_partite"] += 1
            r["somma_prob"] += prob_prevista
            r["per_combo"][chiave_combo(e_p, g_p)]["previste"] += 1
            r["per_combo"][chiave_combo(e_p, g_p)]["somma_prob"] += prob_prevista
            if avverata:
                r["n_corrette"] += 1
                r["per_combo"][chiave_combo(e_p, g_p)]["corrette"] += 1

    return risultati


# =====================================================================
# 🖥️ INTERFACCIA
# =====================================================================
st.title("⚽ COMBO — Advanced Betting Model")
st.caption("Modello statistico Dixon-Coles + EWMA + shrinkage, con combo a due gambe (anche doppia chance) e value bet")

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
        "🎯 Applica calibrazione (se allenata per questo campionato)",
        value=True,
        help="Corregge le probabilità sulla base della verifica storica. Disattiva per confrontare prima/dopo."
    )
    st.divider()
    # Il clic rilancia lo script: svuotando la cache, i dati vengono riscaricati da capo.
    # Utile se è partita una fonte di riserva e football-data.co.uk nel frattempo è tornato.
    if st.button("🔄 Riprova a scaricare i dati"):
        st.cache_data.clear()

modalita_ridotta = False   # True solo se i dati del campionato arrivano dalla fonte di riserva (senza tiri/corner)

scelta_categoria = st.radio("Categoria Torneo", ["Campionati Nazionali (Gratuiti)", "Coppe Europee (Richiede API Key)", "📅 Schedina del giorno (multi-campionato)"], horizontal=True)

if scelta_categoria == "📅 Schedina del giorno (multi-campionato)":
    # =====================================================================
    # 📅 SCHEDINA DEL GIORNO — le migliori occasioni su tutti i campionati
    # domestici insieme, per una data specifica. Filtro obbligatorio: accordo
    # modello/mercato sul segno (idea #1, validata: +5/+10 punti in tutti i
    # campionati testati). L'accordo su Over/Under (idea #1b) è un BONUS
    # nell'ordinamento, non un'esclusione — su Serie A/Premier aiuta poco,
    # escluderlo del tutto avrebbe buttato via occasioni buone lì.
    # Solo la giornata scelta: nessun riempimento con giorni successivi se
    # ne trova meno di 5 — mostra quello che c'è davvero quel giorno.
    # =====================================================================
    st.markdown("## 📅 Schedina del giorno")
    st.caption("Le migliori occasioni su tutti i campionati domestici insieme, per una data specifica. "
               "Filtro: accordo modello/mercato sul segno (obbligatorio). L'accordo anche su Over/Under "
               "fa salire in classifica, ma non esclude — su alcuni campionati aiuta poco.")

    data_scelta = st.date_input("Data delle partite", value=date.today())

    if st.button("📅 Trova le migliori occasioni di questo giorno"):
        candidate = []
        avvisi_leghe = []

        with st.spinner("Scarico e analizzo tutti i campionati (può richiedere qualche secondo)..."):
            for nome_campionato, info_lega in CAMPIONATI_DOMESTICI.items():
                id_fd_lega = info_lega["id_fd"]
                dati_lega = carica_dati_campionato(id_fd_lega, api_key_input)
                fixture_lega = carica_fixture_future(id_fd_lega, api_key_input)

                if dati_lega is None:
                    avvisi_leghe.append(f"{nome_campionato}: dati storici non disponibili in questo momento.")
                    continue
                fonti_lega = fonti_non_principali(dati_lega, fixture_lega)
                if fonti_lega:
                    avvisi_leghe.append(f"{nome_campionato}: fonte di riserva in uso ({'; '.join(sorted(fonti_lega))}).")

                # FIX — ricerca retroattiva: fixtures.csv contiene SOLO partite
                # non ancora giocate (una volta giocate spariscono da lì). Se la
                # data è nel passato, o non è tra le fixture future, cerchiamo
                # nello storico già scaricato — stessa logica, fonte diversa.
                partite_del_giorno = pd.DataFrame()
                if not fixture_lega.empty:
                    partite_del_giorno = fixture_lega[fixture_lega['Date_parsed'].dt.date == data_scelta]
                if partite_del_giorno.empty and 'Date_parsed' in dati_lega.columns:
                    partite_del_giorno = dati_lega[dati_lega['Date_parsed'].dt.date == data_scelta]
                if partite_del_giorno.empty:
                    continue

                colonne_h_s, colonne_d_s, colonne_a_s = classifica_colonne_quote(dati_lega.columns)
                colonne_over_s, colonne_under_s = classifica_colonne_over_under(dati_lega.columns, "2.5")
                calib_1x2_lega = carica_calibratore(id_fd_lega, "1x2") if usa_calibrazione else None

                for _, partita_g in partite_del_giorno.iterrows():
                    data_rif_g = partita_g.get('Date_parsed')
                    dati_prec_g = dati_lega[dati_lega['Date_parsed'] < data_rif_g] if pd.notna(data_rif_g) else dati_lega
                    m_g = calcola_modello_completo(dati_prec_g, partita_g['HomeTeam'], partita_g['AwayTeam'],
                                                    rho_val, ewma_span_val, emivita_val, pd.DataFrame(),
                                                    data_riferimento=data_rif_g)
                    if calib_1x2_lega:
                        m_g = applica_calibrazione_1x2(m_g, calib_1x2_lega["calibratore"])

                    quote_g = quote_mercato_normalizzate(partita_g, colonne_h_s, colonne_d_s, colonne_a_s)
                    if quote_g is None:
                        continue  # senza quote non possiamo verificare l'accordo, scartiamo

                    probabilita_g = {"1": m_g['prob_1'], "X": m_g['prob_X'], "2": m_g['prob_2']}
                    previsione_g = max(probabilita_g, key=probabilita_g.get)
                    quote_per_segno_g = {"1": quote_g["q_casa_equa"], "X": quote_g["q_x_equa"], "2": quote_g["q_trasf_equa"]}
                    favorito_mercato_g = min(quote_per_segno_g, key=quote_per_segno_g.get)

                    if previsione_g != favorito_mercato_g:
                        continue  # filtro obbligatorio: niente accordo sul segno, fuori

                    accordo_ou_g = False
                    quote_ou_g = quote_over_under_normalizzate(partita_g, colonne_over_s, colonne_under_s)
                    if quote_ou_g:
                        prev_ou_g = "Under" if m_g['prob_under'][2.5] >= 50 else "Over"
                        favorito_ou_g = "Over" if quote_ou_g["q_over_equa"] < quote_ou_g["q_under_equa"] else "Under"
                        accordo_ou_g = (prev_ou_g == favorito_ou_g)

                    punteggio_g = probabilita_g[previsione_g] + (5 if accordo_ou_g else 0)
                    candidate.append({
                        "Campionato": nome_campionato, "Partita": f"{partita_g['HomeTeam']} vs {partita_g['AwayTeam']}",
                        "Segno consigliato": previsione_g, "Probabilità": probabilita_g[previsione_g],
                        "Quota mercato": quote_per_segno_g[previsione_g],
                        "Accordo O/U": "✅" if accordo_ou_g else "—", "_punteggio": punteggio_g,
                    })

        if avvisi_leghe:
            for a in avvisi_leghe:
                st.caption(f"⚠️ {a}")

        if not candidate:
            st.warning(f"Nessuna partita con accordo modello/mercato trovata per il {data_scelta.strftime('%d/%m/%Y')}. "
                      "Prova un'altra data, o verifica che le fixture siano già pubblicate per quel giorno.")
        else:
            candidate.sort(key=lambda x: x["_punteggio"], reverse=True)
            top5 = candidate[:5]
            st.success(f"Trovate {len(candidate)} partite con accordo sul segno — mostro le migliori {len(top5)}.")
            df_schedina = pd.DataFrame(top5).drop(columns=["_punteggio"])
            df_schedina["Probabilità"] = df_schedina["Probabilità"].map(lambda x: f"{x:.1f}%")
            df_schedina["Quota mercato"] = df_schedina["Quota mercato"].map(lambda x: f"{x:.2f}")
            st.dataframe(df_schedina, use_container_width=True, hide_index=True)
            st.caption("Ordinate per probabilità del segno (+ bonus se c'è accordo anche su Over/Under). "
                      "Tutte hanno già superato il filtro obbligatorio sul segno.")

    st.stop()  # FIX — senza questo, l'esecuzione proseguiva nel codice sotto
               # (pensato per gli altri due rami) usando variabili mai definite qui.

elif scelta_categoria == "Campionati Nazionali (Gratuiti)":
    campionato = st.selectbox("Seleziona Campionato", list(CAMPIONATI_DOMESTICI.keys()))
    info = CAMPIONATI_DOMESTICI[campionato]
    id_fd = info["id_fd"]

    with st.spinner("Caricamento dataset campionato e quote automatiche..."):
        dati = carica_dati_campionato(id_fd, api_key_input)
        fixture_future = carica_fixture_future(id_fd, api_key_input)

    if dati is None or len(dati) == 0:
        st.error("Impossibile scaricare i dati (né da football-data.co.uk, né da una copia salvata"
                 + (")." if api_key_input else ", e senza chiave API non posso usare la fonte di riserva football-data.org).")
                 + " Riprova tra poco.")
        st.stop()
    modalita_ridotta = FONTE_RIDOTTA in avviso_fonte_dati(dati, fixture_future)

    opzioni_partite, mappa_partite = [], []
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

    with st.expander("📈 Verifica storica (backtest)"):
        st.caption("Quante volte la previsione principale del modello (il segno più probabile) "
                   "ha indovinato il risultato vero — su TUTTE le partite disponibili, senza "
                   "nessun filtro. Confronta stagione corrente e storico completo per vedere "
                   "se il modello regge o se il risultato dipende dal periodo.")

        richiedi_accordo = st.checkbox(
            "💡 Idea #1: valuta solo quando modello e mercato sono d'accordo sul favorito",
            value=False,
            help="Filtro di fiducia, non una correzione della stima: scarta le partite dove il "
                 "modello si discosta dal favorito del mercato. Riduce il campione."
        )

        col_bt_a, col_bt_b = st.columns(2)
        with col_bt_a:
            cliccato_corrente = st.button("🎯 Backtest — solo stagione corrente")
        with col_bt_b:
            cliccato_tutto = st.button("📚 Backtest — tutto lo storico")

        def mostra_risultato_backtest(risultato, etichetta):
            if risultato is None:
                st.warning(f"⚠️ {etichetta}: nessuna partita disponibile per questa modalità.")
                return
            if risultato["n_partite"] == 0:
                st.warning(f"⚠️ {etichetta}: nessuna partita valutabile (o tutte scartate dal filtro).")
                return
            win_rate_tot = risultato["n_corrette"] / risultato["n_partite"] * 100
            st.write(f"**{etichetta}**")
            c1, c2 = st.columns(2)
            c1.metric("Partite valutate", risultato["n_partite"])
            c2.metric("Accuratezza", f"{win_rate_tot:.1f}%")
            if risultato.get("n_scartate_disaccordo", 0) > 0 or risultato.get("n_scartate_no_quote", 0) > 0:
                st.caption(f"Scartate per disaccordo modello/mercato: {risultato['n_scartate_disaccordo']} — "
                          f"scartate per quote mancanti: {risultato['n_scartate_no_quote']}.")
            righe_segno = []
            for s, d in risultato["per_segno"].items():
                wr_s = (d["corrette"]/d["previste"]*100) if d["previste"] > 0 else None
                righe_segno.append({
                    "Segno previsto": s, "Volte previsto": d["previste"],
                    "Corrette": d["corrette"],
                    "Accuratezza": f"{wr_s:.1f}%" if wr_s is not None else "—",
                })
            st.table(pd.DataFrame(righe_segno))

        if cliccato_corrente:
            with st.spinner("Backtest su stagione corrente..."):
                ris = esegui_backtest_senza_filtro(dati, rho_val, ewma_span_val, emivita_val,
                                                    True, id_fd=id_fd, usa_calibrazione=usa_calibrazione,
                                                    richiedi_accordo_mercato=richiedi_accordo)
            mostra_risultato_backtest(ris, "Solo stagione corrente (out-of-sample)")

        if cliccato_tutto:
            with st.spinner("Backtest su tutto lo storico (può richiedere più tempo)..."):
                ris = esegui_backtest_senza_filtro(dati, rho_val, ewma_span_val, emivita_val,
                                                    False, id_fd=id_fd, usa_calibrazione=usa_calibrazione,
                                                    richiedi_accordo_mercato=richiedi_accordo)
            mostra_risultato_backtest(ris, "Tutto lo storico disponibile")

        if cliccato_corrente or cliccato_tutto:
            st.caption("Se il modello prevede quasi sempre '1' e quasi mai 'X', è normale: il pareggio "
                       "è statisticamente l'esito più difficile da prevedere, per qualunque modello. "
                       "Un'accuratezza intorno al 45-55% su 1X2 è nella norma per modelli di questo tipo — "
                       "il mercato stesso, con molte più informazioni, non fa enormemente meglio.")

        st.divider()
        st.write("**🔥 Backtest COMBO — accuratezza della combo più probabile**")
        st.caption("Stessa logica del backtest 1X2, applicata alle combo a due gambe. Per ogni partita si "
                   "valutano DUE previsioni: la combo più probabile con esito singolo (1/X/2) e quella con "
                   "doppia chance (1X/X2/12), ciascuna abbinata a Gol/No Gol, Over/Under 2.5, Over 1.5 o "
                   "Under 3.5 (usa la calibrazione per-combo se allenata e attiva). Le due categorie non sono confrontabili tra "
                   "loro — la doppia chance copre due esiti su tre, quindi ha un'accuratezza molto più alta per "
                   "costruzione — né con i risultati delle vecchie combo a tre gambe.")
        st.caption("Il checkbox 'Idea #1' qui sopra filtra sull'esito (per la doppia chance basta che il favorito "
                   "del mercato sia uno dei due esiti coperti). Quello qui sotto (idea #1b, da verificare) filtra "
                   "ANCHE sull'Over/Under 2.5 quando la combo prevista lo contiene; sulle combo con Gol/No Gol, Over 1.5 "
                   "o Under 3.5, che non hanno una quota di mercato, non può filtrare e la partita passa. Prova le "
                   "combinazioni: nessuno dei due, solo #1, solo #1b, entrambi.")
        st.caption("📏 **Come leggere i risultati**: oltre a quante combo escono (accuratezza) c'è la "
                   "**probabilità media prevista** dal modello per le combo scelte. Se le due cifre coincidono, "
                   "il modello fa quello che dice: le combo escono poco perché sono poco probabili. Se escono "
                   "molto meno di quanto previsto, il modello è troppo sicuro e va corretto.")
        richiedi_accordo_ou = st.checkbox(
            "💡 Idea #1b: richiedi accordo anche su Over/Under 2.5 col mercato",
            value=False,
        )

        col_cb_a, col_cb_b = st.columns(2)
        with col_cb_a:
            cliccato_combo_corrente = st.button("🎯 Backtest combo — solo stagione corrente")
        with col_cb_b:
            cliccato_combo_tutto = st.button("📚 Backtest combo — tutto lo storico")

        def mostra_risultato_backtest_combo(risultato, etichetta):
            if risultato is None:
                st.warning(f"⚠️ {etichetta}: nessuna partita disponibile per questa modalità.")
                return
            st.write(f"**{etichetta}**")
            if usa_calibrazione:
                st.warning("⚠️ Calibrazione attiva: i correttori sono allenati sulle stesse partite che stai "
                           "valutando, quindi probabilità previste e frequenze reali tendono a combaciare per "
                           "costruzione. Per giudicare il modello grezzo disattiva la calibrazione nella barra laterale.")
            for categoria in CATEGORIE_COMBO:
                r = risultato[categoria]
                st.write(f"*{ETICHETTE_CATEGORIA[categoria]}*")
                if r["n_partite"] == 0:
                    st.warning("⚠️ Nessuna partita valutabile (o tutte scartate dai filtri).")
                    continue
                win_rate_combo = r["n_corrette"] / r["n_partite"] * 100
                prob_media = r["somma_prob"] / r["n_partite"]
                scarto, margine, testo_verdetto = verdetto_calibrazione(r["n_partite"], r["n_corrette"], prob_media)
                c1, c2, c3 = st.columns(3)
                c1.metric("Partite valutate", r["n_partite"])
                c2.metric("Accuratezza combo", f"{win_rate_combo:.1f}%")
                c3.metric("Probabilità media prevista", f"{prob_media*100:.1f}%", delta=f"{scarto:+.1f} punti (reale − previsto)",
                          delta_color="off")
                st.caption(f"📏 Scarto {scarto:+.1f} punti (margine statistico ±{margine:.1f}): {testo_verdetto}.")
                if r["n_scartate_disaccordo"] > 0 or r["n_scartate_disaccordo_ou"] > 0 or r["n_scartate_no_quote"] > 0:
                    st.caption(f"Scartate per disaccordo sull'esito: {r['n_scartate_disaccordo']} — "
                              f"scartate per disaccordo su Over/Under: {r['n_scartate_disaccordo_ou']} — "
                              f"scartate per quote mancanti: {r['n_scartate_no_quote']}.")
                righe_combo_bt = []
                for chiave, d in sorted(r["per_combo"].items(), key=lambda x: x[1]["previste"], reverse=True):
                    if d["previste"] == 0: continue
                    wr_c = d["corrette"]/d["previste"]*100
                    pm_c = d["somma_prob"]/d["previste"]*100
                    e_k, g_k = chiave.split("_")
                    righe_combo_bt.append({
                        "Combo prevista": etichetta_combo(e_k, g_k), "Volte prevista": d["previste"],
                        "Corrette": d["corrette"], "Accuratezza": f"{wr_c:.1f}%",
                        "Prob. media prevista": f"{pm_c:.1f}%", "Scarto (punti)": f"{wr_c - pm_c:+.1f}",
                    })
                if righe_combo_bt:
                    st.table(pd.DataFrame(righe_combo_bt))

        if cliccato_combo_corrente:
            with st.spinner("Backtest combo su stagione corrente..."):
                ris_c = esegui_backtest_combo(dati, rho_val, ewma_span_val, emivita_val,
                                              True, id_fd=id_fd, usa_calibrazione=usa_calibrazione,
                                              richiedi_accordo_mercato=richiedi_accordo,
                                              richiedi_accordo_ou=richiedi_accordo_ou)
            mostra_risultato_backtest_combo(ris_c, "Solo stagione corrente (out-of-sample)")

        if cliccato_combo_tutto:
            with st.spinner("Backtest combo su tutto lo storico (può richiedere più tempo)..."):
                ris_c = esegui_backtest_combo(dati, rho_val, ewma_span_val, emivita_val,
                                              False, id_fd=id_fd, usa_calibrazione=usa_calibrazione,
                                              richiedi_accordo_mercato=richiedi_accordo,
                                              richiedi_accordo_ou=richiedi_accordo_ou)
            mostra_risultato_backtest_combo(ris_c, "Tutto lo storico disponibile")

        if cliccato_combo_corrente or cliccato_combo_tutto:
            st.caption("Confronta questa accuratezza con quella 1X2 qui sopra: se è più bassa, è normale "
                       "(una combo richiede che due condizioni si avverino insieme, non una sola), non "
                       "necessariamente un problema — ma dà la misura reale di quanto ci si può fidare delle combo.")

        st.divider()
        st.write("**⚡ Test completo automatico (le 8 combinazioni insieme)**")
        st.caption("Invece di lanciare i due bottoni sopra 4 volte a mano (con/senza ciascun filtro) e "
                   "copiare ogni tabella, questo le fa tutte in un colpo solo (per entrambe le categorie, "
                   "esito singolo e doppia chance) e ti dà un file da scaricare e mandarmi direttamente — "
                   "niente più copia-incolla su Word.")

        if st.button("⚡ Esegui tutte le 8 combinazioni per questo campionato"):
            righe_test_completo = []
            combinazioni = [
                ("Solo stagione corrente", True, False, False),
                ("Solo stagione corrente", True, True, False),
                ("Solo stagione corrente", True, False, True),
                ("Solo stagione corrente", True, True, True),
                ("Tutto lo storico", False, False, False),
                ("Tutto lo storico", False, True, False),
                ("Tutto lo storico", False, False, True),
                ("Tutto lo storico", False, True, True),
            ]
            barra_avanzamento = st.progress(0.0, text="Avvio...")
            for idx, (etichetta_periodo, oos, filtro_segno, filtro_ou) in enumerate(combinazioni):
                barra_avanzamento.progress((idx)/8, text=f"{idx+1}/8 — {etichetta_periodo}, "
                                           f"esito={'sì' if filtro_segno else 'no'}, O/U={'sì' if filtro_ou else 'no'}...")
                ris = esegui_backtest_combo(dati, rho_val, ewma_span_val, emivita_val, oos, id_fd=id_fd,
                                            usa_calibrazione=usa_calibrazione,
                                            richiedi_accordo_mercato=filtro_segno, richiedi_accordo_ou=filtro_ou)
                for categoria in CATEGORIE_COMBO:
                    r = ris[categoria] if ris is not None else None
                    riga = {
                        "Campionato": campionato, "Periodo": etichetta_periodo,
                        "Categoria": ETICHETTE_CATEGORIA[categoria],
                        "Filtro esito": "sì" if filtro_segno else "no", "Filtro O/U": "sì" if filtro_ou else "no",
                    }
                    if r is None or r["n_partite"] == 0:
                        riga.update({"Partite valutate": 0, "Corrette": 0, "Accuratezza %": None,
                                     "Prob. media prevista %": None, "Scarto (punti)": None})
                    else:
                        acc = r["n_corrette"] / r["n_partite"] * 100
                        pm = r["somma_prob"] / r["n_partite"] * 100
                        riga.update({"Partite valutate": r["n_partite"], "Corrette": r["n_corrette"],
                                     "Accuratezza %": round(acc, 1), "Prob. media prevista %": round(pm, 1),
                                     "Scarto (punti)": round(acc - pm, 1)})
                    righe_test_completo.append(riga)
            barra_avanzamento.progress(1.0, text="Completato!")

            df_test_completo = pd.DataFrame(righe_test_completo)
            st.dataframe(df_test_completo, use_container_width=True, hide_index=True)

            csv_bytes = df_test_completo.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Scarica questi risultati come CSV",
                data=csv_bytes,
                file_name=f"backtest_combo_{id_fd}.csv",
                mime="text/csv",
            )
            st.caption("Scarica il CSV per ogni campionato che testi, poi mandameli tutti insieme — "
                       "posso leggerli direttamente, non serve più copiarli su Word.")


        st.divider()
        st.write("**🎯 Calibrazione (1X2 + le 36 combo automatiche)**")
        st.caption("Allena i correttori su tutto lo storico disponibile. Una sola passata sui dati "
                   "calcola il modello una volta per partita e allena tutti i 37 correttori insieme "
                   "(1X2 + le 36 combinazioni esito, anche doppia chance, × Gol/No Gol/Over-Under 2.5/Over 1.5/Under 3.5). "
                   "I vecchi correttori delle combo a tre gambe non vengono più usati: va riallenato.")
        if st.button("🎯 Allena tutte le calibrazioni per questo campionato"):
            with st.spinner("Allenamento in corso (può richiedere qualche secondo)..."):
                risultati_calib = allena_tutte_le_calibrazioni(dati, rho_val, ewma_span_val, emivita_val, id_fd)
            ok_1x2 = risultati_calib["1x2"]["ok"]
            n_combo_ok = sum(1 for k, v in risultati_calib.items() if k != "1x2" and v["ok"])
            if ok_1x2:
                st.success(f"✅ 1X2: calibrato su {risultati_calib['1x2']['n']} osservazioni.")
            else:
                st.warning(f"⚠️ 1X2: campione insufficiente ({risultati_calib['1x2']['n']} osservazioni, "
                          f"ne servono almeno 100).")
            st.write(f"**Combo calibrate con successo: {n_combo_ok} su {len(LE_COMBO)}.**")
            righe_combo_calib = []
            for chiave, v in risultati_calib.items():
                if chiave == "1x2": continue
                e_k, g_k = chiave.split("_")
                righe_combo_calib.append({"Combo": etichetta_combo(e_k, g_k), "Osservazioni": v["n"],
                                          "Calibrata": "✅" if v["ok"] else "⚠️ campione scarso"})
            st.dataframe(pd.DataFrame(righe_combo_calib), use_container_width=True, hide_index=True)
            st.caption("Le combo più rare (es. 'X + Over 2.5') hanno naturalmente meno osservazioni "
                       "delle più comuni — è normale, non un errore.")

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
        df_globale = carica_tutti_i_campionati(api_key_input)

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

    avviso_fonte_dati(df_globale)
    dati_storico = dati[dati['Status'] == 'FINISHED'].copy()
    dati_future = dati[dati['Status'] != 'FINISHED'].copy()

    opzioni_partite, mappa_partite = [], []
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

    if modello is None:
        st.warning(f"⚠️ Impossibile elaborare il match per **{partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}**.")
    else:
        st.subheader(f"📊 Analisi Match: {partita_sel['HomeTeam']} vs {partita_sel['AwayTeam']}")
        if calib_1x2_info:
            st.caption(f"🎯 Probabilità 1X2 corrette con calibrazione (allenata su "
                       f"{calib_1x2_info['n_osservazioni']} osservazioni, {calib_1x2_info['timestamp']}).")

        # Avviso trasparenza dati (sostituisce il vecchio generatore silenzioso di dati finti)
        SOGLIA_AVVISO = 5
        avvisi = []
        for nome_sq, n_sq in [(partita_sel['HomeTeam'], modello["n_casa"]), (partita_sel['AwayTeam'], modello["n_trasf"])]:
            if n_sq == 0:
                avvisi.append(f"{nome_sq} (nessuna partita trovata: nome non riconosciuto o squadra senza storico)")
            elif n_sq < SOGLIA_AVVISO:
                avvisi.append(f"{nome_sq} (solo {n_sq} partite trovate)")
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

        st.markdown("### 🏆 Esito Finale (1X2)")
        df_1x2 = crea_tabella({"1 (Casa)": modello['prob_1'], "X (Pareggio)": modello['prob_X'], "2 (Trasferta)": modello['prob_2']}, "Segno")
        st.dataframe(df_1x2, use_container_width=True, hide_index=True)

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
        # 🏆 TOP COMBO AUTOMATICHE — a DUE gambe: un esito (1/X/2 oppure doppia
        # chance 1X/X2/12) + una condizione tra Gol, No Gol, Over 2.5, Under 2.5,
        # Over 1.5, Under 3.5.
        # Due liste separate: la doppia chance copre due esiti su tre e
        # occuperebbe quasi sempre i primi posti di una classifica unica.
        # Riusa calcola_combo_libera (un solo posto dove le combo vengono calcolate).
        # =====================================================================
        st.markdown("### 🏆 Top 3 Combo Automatiche")
        st.caption("Combo a due gambe: un esito (1/X/2, oppure doppia chance 1X/X2/12) abbinato a Gol, No Gol, "
                   "Over/Under 2.5, Over 1.5 o Under 3.5. Probabilità calibrate se disponibile (🎯). Per ognuna delle "
                   "due categorie (la doppia chance, coprendo due esiti su tre, avrebbe quasi sempre le "
                   "probabilità più alte) ci sono due liste, entrambe ordinate per probabilità: le 3 più "
                   "probabili in assoluto e le 3 più probabili con la linea 2.5. Over 1.5 e Under 3.5 sono "
                   "linee meno 50/50: escono più spesso, ma pagano meno, e di solito dominano la prima lista.")

        dati_scarsi = bool(avvisi)
        calibratori_combo = carica_calibratori_combo(id_fd) if (not is_coppa and usa_calibrazione) else None
        prob_combo = probabilita_tutte_le_combo(modello['griglia'], calibratori_combo)

        if dati_scarsi:
            st.warning("⚠️ Combo calcolate su dati limitati per una delle due squadre (vedi avviso sopra) "
                      "— trattale con più cautela del solito.")

        def tabella_top3(combo_ordinate):
            """Le prime 3 combo (già ordinate per probabilità) come tabella con quota stimata."""
            righe_top3 = []
            for (e_c, g_c) in combo_ordinate[:3]:
                prob_c, calibrata_c = prob_combo[(e_c, g_c)]
                con_ou = g_c in ("Over", "Under")
                q_stimata, non_prezzate = stima_quota_combo_approssimata(
                    e_c, 2.5 if con_ou else None, g_c if con_ou else None, quote, quote_ou_25)
                quota_ok = bool(q_stimata) and con_ou and not non_prezzate
                righe_top3.append({
                    "Combo": etichetta_combo(e_c, g_c) + (" 🎯" if calibrata_c else ""),
                    "Probabilità (%)": f"{prob_c:.1f}%",
                    "Quota stimata*": f"~{q_stimata:.2f}" if quota_ok else "n/d",
                })
            st.dataframe(pd.DataFrame(righe_top3), use_container_width=True, hide_index=True)

        for categoria, esiti_cat in CATEGORIE_COMBO.items():
            st.markdown(f"**{ETICHETTE_CATEGORIA[categoria]}**")
            tutte_cat = sorted(((e, g) for e in esiti_cat for g in GAMBE_GOL),
                               key=lambda c: prob_combo[c][0], reverse=True)
            st.caption("Le 3 più probabili (qualsiasi gamba)")
            tabella_top3(tutte_cat)
            # Seconda lista: solo la linea 2.5, sempre ordinata per probabilità. Sono le uniche
            # con una quota di mercato (Over/Under 2.5) da cui stimare la quota della combo.
            st.caption("Le 3 più probabili con la linea 2.5 (Over/Under 2.5)")
            tabella_top3([c for c in tutte_cat if c[1] in ("Over", "Under")])
        st.caption("🎯 = probabilità corretta con calibrazione specifica per questa combo. "
                   "*Quota STIMATA solo per le combo con Over/Under 2.5: esito dalle quote 1X2 (la doppia chance dalle "
                   "probabilità eque 1X2) × Over/Under 2.5, come se fossero indipendenti — non lo sono del tutto. "
                   "Sono quote EQUE, cioè senza il margine del bookmaker: quelle reali sono più basse. Gol/No Gol, "
                   "Over 1.5 e Under 3.5 non hanno una quota nei file, quindi per quelle combo il valore è n/d.")

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
        if modalita_ridotta:
            st.info("🚩 Angoli e 🎯 tiri in porta: non disponibili in modalità ridotta "
                    "(la fonte di riserva non li include; mostrare i valori medi di default sarebbe fuorviante).")
        else:
            st.info(f"🚩 Angoli Totali Stimati: **{modello['angoli_stimati']}** | "
                    f"🎯 Tiri in Porta Totali Stimati: **{modello['tiri_stimati']}**{nota_stima}")
