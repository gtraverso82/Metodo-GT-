import requests, pandas as pd, numpy as np
from io import StringIO
from datetime import datetime, timedelta

LIGA_ERA_PROMEDIO = 4.20
LIGA_RUNS_PROMEDIO = 4.35
PRIOR_IP_ABRIDOR = 45
PRIOR_IP_BULLPEN = 28
TOPE_ERA_RELEVISTA = 9.00
LIGA_KBB_POR_IP = 0.60
DISPERSION_RUNS = 1.2
N_SIMULACIONES = 20000
PLATT_A = 0.5795
PLATT_B = 0.1449
LIGA_HR_FB_RATE = 0.105
FIP_CONSTANT_AJUSTADO = 3.582589285714286

TEAM_IDS = {
    "LAA":108, "AZ":109, "BAL":110, "BOS":111, "CHC":112, "CIN":113, "CLE":114,
    "COL":115, "DET":116, "HOU":117, "KC":118, "LAD":119, "WSH":120, "NYM":121,
    "OAK":133, "ATH":133, "PIT":134, "SD":135, "SEA":136, "SF":137, "STL":138, "TB":139,
    "TEX":140, "TOR":141, "MIN":142, "PHI":143, "ATL":144, "CWS":145, "MIA":146,
    "NYY":147, "MIL":158,
}
COORDENADAS_ESTADIO = {
    "LAA": (33.8003, -117.8827), "AZ": (33.4455, -112.0667), "BAL": (39.2838, -76.6217),
    "BOS": (42.3467, -71.0972), "CHC": (41.9484, -87.6553), "CIN": (39.0975, -84.5066),
    "CLE": (41.4962, -81.6852), "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555), "KC": (39.0517, -94.4803), "LAD": (34.0739, -118.2400),
    "WSH": (38.8730, -77.0074), "NYM": (40.7571, -73.8458), "OAK": (37.7516, -122.2005),
    "ATH": (37.7516, -122.2005), "PIT": (40.4469, -80.0057), "SD": (32.7076, -117.1570),
    "SEA": (47.5914, -122.3325), "SF": (37.7786, -122.3893), "STL": (38.6226, -90.1928),
    "TB": (27.7683, -82.6534), "TEX": (32.7473, -97.0842), "TOR": (43.6414, -79.3894),
    "MIN": (44.9817, -93.2776), "PHI": (39.9061, -75.1665), "ATL": (33.8907, -84.4677),
    "CWS": (41.8299, -87.6338), "MIA": (25.7781, -80.2196), "NYY": (40.8296, -73.9262),
    "MIL": (43.0280, -87.9712),
}

PARK_FACTORS = {
    "COL":1.15,"CIN":1.08,"BOS":1.06,"TEX":1.05,"PHI":1.04,
    "BAL":1.03,"TOR":1.02,"MIL":1.02,"CWS":1.01,"HOU":1.01,
    "ATL":1.00,"AZ":1.00,"WSH":1.00,"CHC":0.99,"MIN":0.99,
    "KC":0.99,"LAA":0.98,"NYY":0.98,"TB":0.97,"STL":0.97,
    "CLE":0.97,"NYM":0.96,"DET":0.96,"SEA":0.95,"MIA":0.94,
    "SD":0.94,"OAK":0.94,"ATH":0.94,"LAD":0.93,"SF":0.92,
}

def shrink_era(era, ip, prior_ip, prior_era=LIGA_ERA_PROMEDIO):
    return (era * ip + prior_era * prior_ip) / (ip + prior_ip)

def factor_kbb(k, bb, ip, prior_ip=PRIOR_IP_ABRIDOR, prior_kbb=LIGA_KBB_POR_IP):
    if ip <= 0: return 1.0
    kbb = (k - bb) / ip
    kbb_adj = (kbb * ip + prior_kbb * prior_ip) / (ip + prior_ip)
    if kbb_adj <= 0: kbb_adj = 0.05
    return prior_kbb / kbb_adj

def calcular_xfip(k, bb, hbp, fly_balls, ip):
    if ip <= 0:
        return LIGA_ERA_PROMEDIO
    hr_esperados = fly_balls * LIGA_HR_FB_RATE
    xfip = ((13 * hr_esperados) + (3 * (bb + hbp)) - (2 * k)) / ip + FIP_CONSTANT_AJUSTADO
    return max(xfip, 0.5)

def sigma_era_muestral(era, ip):
    if ip <= 0:
        return 2.0
    return np.sqrt(9 * max(era, 0.1) / ip)
def gamelog_pitcher(pitcher_id, year):
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&group=pitching&season={year}"
    r = requests.get(url)
    try:
        return r.json()["stats"][0]["splits"]
    except (KeyError, IndexError):
        return []

def parse_ip(ip_str):
    partes = ip_str.split(".")
    return float(partes[0]) + (int(partes[1])/3 if len(partes) > 1 else 0)

def stats_abridor_hasta_hoy(pitcher_id, fecha_hoy, year):
    gamelog = gamelog_pitcher(pitcher_id, year)
    ip_t, k_t, bb_t, er_t = 0.0, 0, 0, 0
    for ap in gamelog:
        if ap["date"] >= fecha_hoy:
            continue
        s = ap["stat"]
        ip = parse_ip(s.get("inningsPitched", "0.0"))
        ip_t += ip
        k_t += s.get("strikeOuts", 0)
        bb_t += s.get("baseOnBalls", 0)
        er_t += s.get("earnedRuns", 0)
    era = (er_t * 9 / ip_t) if ip_t > 0 else 0.0
    return ip_t, k_t, bb_t, era

def stats_abridor_xfip_hasta_hoy(pitcher_id, fecha_hoy, year):
    gamelog = gamelog_pitcher(pitcher_id, year)
    ip_t, k_t, bb_t, hr_t, hbp_t, fb_t = 0.0, 0, 0, 0, 0, 0
    for ap in gamelog:
        if ap["date"] >= fecha_hoy:
            continue
        s = ap["stat"]
        ip_t += parse_ip(s.get("inningsPitched", "0.0"))
        k_t += s.get("strikeOuts", 0)
        bb_t += s.get("baseOnBalls", 0)
        hr_t += s.get("homeRuns", 0)
        hbp_t += s.get("hitByPitch", 0)
        fb_t += s.get("flyOuts", 0)
    return ip_t, k_t, bb_t, hr_t, hbp_t, fb_t

def stats_ofensiva_hasta_hoy(team, fecha_inicio, fecha_hoy, year):
    url = (f"https://baseballsavant.mlb.com/statcast_search/csv?"
           f"all=true&hfGT=R%7C&hfSea={year}%7C&player_type=batter&team={team}"
           f"&group_by=name&sort_col=pitches&player_event_sort=api_p_release_speed"
           f"&sort_order=desc&type=details&game_date_gt={fecha_inicio}&game_date_lt={fecha_hoy}")
    r = requests.get(url)
    df = pd.read_csv(StringIO(r.text))
    dfv = df[df["launch_speed"].notna() & df["events"].notna()]
    if len(dfv) == 0:
        return 88.5, 0.075
    ev = dfv["launch_speed"].mean()
    barrels = (dfv["launch_speed_angle"] == 6).sum()
    return round(ev, 1), round(barrels/len(dfv), 3)

def bullpen_reciente(team_abbrev, fecha_hoy, year, dias_atras=5):
    team_id = TEAM_IDS.get(team_abbrev)
    if not team_id:
        return []
    fecha_dt = datetime.strptime(fecha_hoy, "%Y-%m-%d")
    fecha_inicio_ventana = (fecha_dt - timedelta(days=dias_atras)).strftime("%Y-%m-%d")
    fecha_fin_ventana = (fecha_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}"
           f"&startDate={fecha_inicio_ventana}&endDate={fecha_fin_ventana}")
    r = requests.get(url)
    data = r.json()
    game_pks = []
    for fecha_obj in data.get("dates", []):
        for juego in fecha_obj.get("games", []):
            if juego.get("status", {}).get("abstractGameState") == "Final":
                game_pks.append(juego["gamePk"])
    relevistas_ids = set()
    for pk in game_pks:
        r2 = requests.get(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore")
        box = r2.json()
        for lado in ["home", "away"]:
            equipo_box = box["teams"][lado]
            if equipo_box["team"]["abbreviation"] != team_abbrev:
                continue
            for pid in equipo_box.get("pitchers", [])[1:]:
                relevistas_ids.add(pid)
    eras_bullpen = []
    for pid in relevistas_ids:
        gamelog = gamelog_pitcher(pid, year)
        if not gamelog:
            continue
        ultimas_5 = sorted(gamelog, key=lambda x: x["date"])[-5:]
        ip_t, er_t = 0.0, 0
        for ap in ultimas_5:
            s = ap["stat"]
            ip_t += parse_ip(s.get("inningsPitched", "0.0"))
            er_t += s.get("earnedRuns", 0)
        if ip_t > 0:
            eras_bullpen.append(round(er_t * 9 / ip_t, 2))
    return eras_bullpen

def winsorizar_bullpen(eras, tope=TOPE_ERA_RELEVISTA):
    return [min(e, tope) for e in eras]
