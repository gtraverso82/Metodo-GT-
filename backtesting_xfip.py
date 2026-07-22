import requests, json, os
from datetime import datetime
from supabase import create_client
from motor import (
    gamelog_pitcher, parse_ip, shrink_era, factor_kbb, calcular_xfip,
    calibrar_platt, PRIOR_IP_ABRIDOR, LIGA_ERA_PROMEDIO, TEAM_IDS
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

DATASET_URL = "https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/download/dataset/mlb_odds.json"

def obtener_url_dataset():
    api_url = "https://api.github.com/repos/ArnavSaraogi/mlb-odds-scraper/releases/tags/dataset"
    r = requests.get(api_url, timeout=30)
    data = r.json()
    assets = data.get("assets", [])
    if not assets:
        raise Exception("No se encontraron assets en el release")
    print(f"Asset encontrado: {assets[0]['name']}")
    return assets[0]["browser_download_url"]

def descargar_y_filtrar_dataset(anio_inicio=2022, anio_fin=2023):
    url_real = obtener_url_dataset()
    print(f"Descargando desde: {url_real}")
    r = requests.get(url_real, timeout=300)
    print(f"Status code: {r.status_code}, tamano: {len(r.content)} bytes")
    data_completa = r.json()
    print(f"Dataset descargado: {len(data_completa)} fechas totales")

    juegos_filtrados = {}
    for fecha, juegos in data_completa.items():
        anio = int(fecha[:4])
        if anio < anio_inicio or anio > anio_fin:
            continue
        juegos_regulares = [j for j in juegos if j.get("gameView", {}).get("gameType") == "R"]
        if juegos_regulares:
            juegos_filtrados[fecha] = juegos_regulares

    total_juegos = sum(len(v) for v in juegos_filtrados.values())
    print(f"Juegos filtrados ({anio_inicio}-{anio_fin}, temporada regular): {total_juegos}")
    return juegos_filtrados

def obtener_abridor_real(team_abbrev, fecha):
    team_id = TEAM_IDS.get(team_abbrev)
    if not team_id:
        return None
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}&date={fecha}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        juego = data["dates"][0]["games"][0]
        game_pk = juego["gamePk"]
    except (KeyError, IndexError):
        return None

    try:
        r2 = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=10)
        box = r2.json()
        for lado in ["home", "away"]:
            equipo_box = box["teams"][lado]
            if equipo_box["team"]["abbreviation"] == team_abbrev:
                pitchers = equipo_box.get("pitchers", [])
                return pitchers[0] if pitchers else None
    except (KeyError, IndexError):
        return None
    return None
EV_LIGA = 88.5
BARREL_LIGA = 0.075
PARK_FACTOR_NEUTRAL = 1.00

def runs_esperados_simplificado(era_rival, ip_rival, k_rival, bb_rival):
    era_adj = shrink_era(era_rival, ip_rival, PRIOR_IP_ABRIDOR)
    f_era = era_adj / LIGA_ERA_PROMEDIO
    f_kbb = factor_kbb(k_rival, bb_rival, ip_rival)
    f_abridor = (f_era * 0.70) + (f_kbb * 0.30)
    f_ofensiva = (EV_LIGA/88.5 * 0.6) + (BARREL_LIGA/0.075 * 0.4)
    return 4.35 * f_ofensiva * f_abridor * PARK_FACTOR_NEUTRAL

def correr_backtesting():
    juegos = descargar_y_filtrar_dataset(2022, 2023)
    cache_gamelogs = {}
    resultados = []
    contador = 0

    for fecha, lista_juegos in juegos.items():
        for juego in lista_juegos:
            gv = juego.get("gameView", {})
            local = gv.get("homeTeam", {}).get("shortName")
            visitante = gv.get("awayTeam", {}).get("shortName")
            score_local = gv.get("homeTeamScore")
            score_visitante = gv.get("awayTeamScore")

            if local not in TEAM_IDS or visitante not in TEAM_IDS:
                continue
            if score_local is None or score_visitante is None:
                continue

            pid_local = obtener_abridor_real(local, fecha)
            pid_visitante = obtener_abridor_real(visitante, fecha)
            if not pid_local or not pid_visitante:
                continue

            year = int(fecha[:4])

            if pid_local not in cache_gamelogs:
                cache_gamelogs[pid_local] = gamelog_pitcher(pid_local, year)
            if pid_visitante not in cache_gamelogs:
                cache_gamelogs[pid_visitante] = gamelog_pitcher(pid_visitante, year)

            contador += 1
            if contador % 50 == 0:
                print(f"Procesados: {contador} juegos...")

    print(f"Total procesado (fase 1 - descarga gamelogs): {contador} juegos")
    return cache_gamelogs, juegos

if __name__ == "__main__":
    correr_backtesting()
