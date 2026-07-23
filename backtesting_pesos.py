import requests, json, os
from datetime import datetime
from supabase import create_client
from motor import (
    gamelog_pitcher, parse_ip, shrink_era, factor_kbb,
    PRIOR_IP_ABRIDOR, PRIOR_IP_BULLPEN, LIGA_ERA_PROMEDIO, TOPE_ERA_RELEVISTA, TEAM_IDS
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

DATASET_URL_API = "https://api.github.com/repos/ArnavSaraogi/mlb-odds-scraper/releases/tags/dataset"

EV_LIGA = 88.5
BARREL_LIGA = 0.075

def obtener_url_dataset():
    r = requests.get(DATASET_URL_API, timeout=30)
    data = r.json()
    assets = data.get("assets", [])
    return assets[0]["browser_download_url"]

def descargar_y_filtrar_dataset(anio_inicio=2022, anio_fin=2023):
    url_real = obtener_url_dataset()
    r = requests.get(url_real, timeout=300)
    data_completa = r.json()
    juegos_filtrados = {}
    for fecha, juegos in data_completa.items():
        anio = int(fecha[:4])
        if anio < anio_inicio or anio > anio_fin:
            continue
        juegos_regulares = [j for j in juegos if j.get("gameView", {}).get("gameType") == "R"]
        if juegos_regulares:
            juegos_filtrados[fecha] = juegos_regulares
    total = sum(len(v) for v in juegos_filtrados.values())
    print(f"Juegos filtrados: {total}")
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

def calcular_era_hasta(gamelog, fecha_corte):
    ip_t, k_t, bb_t, er_t = 0.0, 0, 0, 0
    for ap in gamelog:
        if ap["date"] >= fecha_corte:
            continue
        s = ap["stat"]
        ip_t += parse_ip(s.get("inningsPitched", "0.0"))
        k_t += s.get("strikeOuts", 0)
        bb_t += s.get("baseOnBalls", 0)
        er_t += s.get("earnedRuns", 0)
    if ip_t <= 0:
        return None
    era = er_t * 9 / ip_t
    return (era, ip_t, k_t, bb_t)

def componentes_pitcheo(era_rival, ip_rival, k_rival, bb_rival):
    era_adj = shrink_era(era_rival, ip_rival, PRIOR_IP_ABRIDOR)
    f_era = era_adj / LIGA_ERA_PROMEDIO
    f_kbb = factor_kbb(k_rival, bb_rival, ip_rival)
    return f_era, f_kbb

def correr_backtesting_pesos():
    juegos = descargar_y_filtrar_dataset(2022, 2023)
    cache_gamelogs = {}
    lote = []
    contador = 0
    guardados = 0

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

            era_l = calcular_era_hasta(cache_gamelogs[pid_local], fecha)
            era_v = calcular_era_hasta(cache_gamelogs[pid_visitante], fecha)
            if era_l is None or era_v is None:
                continue

            f_era_l, f_kbb_l = componentes_pitcheo(era_l[0], era_l[1], era_l[2], era_l[3])
            f_era_v, f_kbb_v = componentes_pitcheo(era_v[0], era_v[1], era_v[2], era_v[3])

            gano_local = score_local > score_visitante

            lote.append({
                "game_id": f"{visitante}@{local}_{fecha}",
                "fecha": fecha,
                "f_era_local": round(float(f_era_l), 4),
                "f_kbb_local": round(float(f_kbb_l), 4),
                "f_era_visitante": round(float(f_era_v), 4),
                "f_kbb_visitante": round(float(f_kbb_v), 4),
                "gano_local": bool(gano_local),
                "runs_reales_local": int(score_local),
                "runs_reales_visitante": int(score_visitante),
            })

            contador += 1
            if len(lote) >= 100:
                supabase.table("backtesting_pesos").insert(lote).execute()
                guardados += len(lote)
                print(f"Guardados: {guardados} / procesados: {contador}")
                lote = []

    if lote:
        supabase.table("backtesting_pesos").insert(lote).execute()
        guardados += len(lote)

    print(f"=== Backtesting de pesos completo: {guardados} juegos guardados ===")

if __name__ == "__main__":
    correr_backtesting_pesos()
