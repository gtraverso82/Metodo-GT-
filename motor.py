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
LIGA_OPS_CONTRA = 0.700

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

def runs_esperados_completo(era_rival, ip_rival, k_rival, bb_rival, bullpen_eras_rival,
                             ev_propio, barrel_propio, park_factor):
    era_adj = shrink_era(era_rival, ip_rival, PRIOR_IP_ABRIDOR)
    f_era = era_adj / LIGA_ERA_PROMEDIO
    f_kbb = factor_kbb(k_rival, bb_rival, ip_rival)
    f_abridor = (f_era * 0.70) + (f_kbb * 0.30)
    if bullpen_eras_rival:
        eras_topadas = winsorizar_bullpen(bullpen_eras_rival)
        era_bp = shrink_era(np.mean(eras_topadas), PRIOR_IP_BULLPEN, PRIOR_IP_BULLPEN)
        f_bullpen = era_bp / LIGA_ERA_PROMEDIO
    else:
        f_bullpen = 1.0
    f_pitcheo = (f_abridor * 0.58) + (f_bullpen * 0.42)
    f_ofensiva = (ev_propio/88.5 * 0.6) + (barrel_propio/0.075 * 0.4)
    return LIGA_RUNS_PROMEDIO * f_ofensiva * f_pitcheo * park_factor

def simular_negbinom(media, n_sim, dispersion=DISPERSION_RUNS):
    media = max(media, 0.1)
    var = media * dispersion
    p = media / var
    r = media * p / (1 - p)
    return np.random.negative_binomial(r, p, n_sim)

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def calibrar_platt(prob_cruda):
    z = logit(prob_cruda)
    return 1 / (1 + np.exp(-(PLATT_A * z + PLATT_B)))

def cuota_a_prob(cuota):
    return -cuota/(-cuota+100) if cuota < 0 else 100/(cuota+100)

def remover_vig(pa, pb):
    t = pa + pb
    return pa/t, pb/t

def calcular_confianza(ip_l, ip_v, n_bp_l, n_bp_v, bandera):
    conf_ip = (min(ip_l/PRIOR_IP_ABRIDOR,1) + min(ip_v/PRIOR_IP_ABRIDOR,1)) / 2
    conf_bp = min((n_bp_l+n_bp_v)/8, 1.0)
    conf = (conf_ip*0.6) + (conf_bp*0.4)
    if bandera == "extrema": conf *= 0.5
    elif bandera == "moderada": conf *= 0.9
    return round(conf*100, 1)

def recomendacion_final(bandera, confianza):
    if bandera == "extrema": return "NO JUGAR — divergencia sospechosa"
    if bandera == "alineado": return "NO JUGAR — sin edge, Confirmación"
    if confianza < 45: return "CAUTELA — edge moderado, baja confianza"
    return "CRUZAR CON ANÁLISIS CUALITATIVO"

def obtener_cuotas_espn(equipo_local, equipo_visitante, fecha):
    fecha_espn = fecha.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={fecha_espn}"
    r = requests.get(url)
    data = r.json()
    for evento in data.get("events", []):
        competencia = evento["competitions"][0]
        equipos = competencia["competitors"]
        home = next(e for e in equipos if e["homeAway"] == "home")
        away = next(e for e in equipos if e["homeAway"] == "away")
        if home["team"]["abbreviation"] == equipo_local and away["team"]["abbreviation"] == equipo_visitante:
            odds_list = competencia.get("odds", [])
            if not odds_list:
                return None, None
            moneyline = odds_list[0].get("moneyline", {})
            cuota_local = moneyline.get("home", {}).get("close", {}).get("odds")
            cuota_visitante = moneyline.get("away", {}).get("close", {}).get("odds")
            try:
                return int(cuota_local), int(cuota_visitante)
            except (TypeError, ValueError):
                return None, None
    return None, None

def obtener_handicap_espn(equipo_local, equipo_visitante, fecha):
    fecha_espn = fecha.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={fecha_espn}"
    r = requests.get(url)
    data = r.json()
    for evento in data.get("events", []):
        competencia = evento["competitions"][0]
        equipos = competencia["competitors"]
        home = next(e for e in equipos if e["homeAway"] == "home")
        away = next(e for e in equipos if e["homeAway"] == "away")
        if home["team"]["abbreviation"] == equipo_local and away["team"]["abbreviation"] == equipo_visitante:
            odds_list = competencia.get("odds", [])
            if not odds_list:
                return None
            spread_info = odds_list[0].get("pointSpread", {})
            home_odds = spread_info.get("home", {}).get("close", {}).get("odds")
            away_odds = spread_info.get("away", {}).get("close", {}).get("odds")
            home_line = spread_info.get("home", {}).get("close", {}).get("line")
            try:
                return {"home_line": float(home_line) if home_line else None,
                        "home_odds": int(home_odds) if home_odds else None,
                        "away_odds": int(away_odds) if away_odds else None}
            except (TypeError, ValueError):
                return None
    return None

def obtener_total_espn(equipo_local, equipo_visitante, fecha):
    fecha_espn = fecha.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={fecha_espn}"
    r = requests.get(url)
    data = r.json()
    for evento in data.get("events", []):
        competencia = evento["competitions"][0]
        equipos = competencia["competitors"]
        home = next(e for e in equipos if e["homeAway"] == "home")
        away = next(e for e in equipos if e["homeAway"] == "away")
        if home["team"]["abbreviation"] == equipo_local and away["team"]["abbreviation"] == equipo_visitante:
            odds_list = competencia.get("odds", [])
            if not odds_list:
                return None
            total_info = odds_list[0].get("total", {})
            over_odds = total_info.get("over", {}).get("close", {}).get("odds")
            under_odds = total_info.get("under", {}).get("close", {}).get("odds")
            over_line = total_info.get("over", {}).get("close", {}).get("line", "")
            try:
                linea = float(over_line.replace("o", ""))
                return {"linea": linea, "over_odds": int(over_odds), "under_odds": int(under_odds)}
            except (TypeError, ValueError, AttributeError):
                return None
    return None

def obtener_clima(team_abbrev, fecha):
    coords = COORDENADAS_ESTADIO.get(team_abbrev)
    if not coords:
        return None
    lat, lon = coords
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
           f"&temperature_unit=fahrenheit&windspeed_unit=mph"
           f"&start_date={fecha}&end_date={fecha}")
    r = requests.get(url, timeout=10)
    data = r.json()
    try:
        hourly = data["hourly"]
        idx = 23
        return {"temperatura_f": hourly["temperature_2m"][idx],
                "viento_mph": hourly["windspeed_10m"][idx],
                "direccion_viento": hourly["winddirection_10m"][idx]}
    except (KeyError, IndexError):
        return None

def analizar_partido_hoy(equipo_local, equipo_visitante, pitcher_id_local, pitcher_id_visitante,
                          park_factor, cuota_ml_local, cuota_ml_visitante,
                          fecha_hoy=None, year=2026, inicio_temporada="2026-03-15"):
    if fecha_hoy is None:
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    ip_l, k_l, bb_l, era_l = stats_abridor_hasta_hoy(pitcher_id_local, fecha_hoy, year)
    ip_v, k_v, bb_v, era_v = stats_abridor_hasta_hoy(pitcher_id_visitante, fecha_hoy, year)
    ev_l, barrel_l = stats_ofensiva_hasta_hoy(equipo_local, inicio_temporada, fecha_hoy, year)
    ev_v, barrel_v = stats_ofensiva_hasta_hoy(equipo_visitante, inicio_temporada, fecha_hoy, year)
    bp_l = bullpen_reciente(equipo_local, fecha_hoy, year)
    bp_v = bullpen_reciente(equipo_visitante, fecha_hoy, year)
    runs_l = runs_esperados_completo(era_v, ip_v, k_v, bb_v, bp_v, ev_l, barrel_l, park_factor)
    runs_v = runs_esperados_completo(era_l, ip_l, k_l, bb_l, bp_l, ev_v, barrel_v, park_factor)
    sim_l = simular_negbinom(runs_l, N_SIMULACIONES)
    sim_v = simular_negbinom(runs_v, N_SIMULACIONES)
    victorias_l = np.sum(sim_l > sim_v) + np.sum(sim_l == sim_v)//2
    prob_l_cruda = victorias_l / N_SIMULACIONES
    prob_l = calibrar_platt(prob_l_cruda)
    p_impl_l, p_impl_v = remover_vig(cuota_a_prob(cuota_ml_local), cuota_a_prob(cuota_ml_visitante))
    diff = prob_l - p_impl_l
    bandera = "extrema" if abs(diff) > 0.15 else ("moderada" if abs(diff) > 0.06 else "alineado")
    confianza = calcular_confianza(ip_l, ip_v, len(bp_l), len(bp_v), bandera)
    recomendacion = recomendacion_final(bandera, confianza)
    return {"prob_local": prob_l, "bandera": bandera, "confianza": confianza,
            "recomendacion": recomendacion, "runs_local": runs_l, "runs_visitante": runs_v}

def analizar_total(runs_local, runs_visitante, cuota_over=None, cuota_under=None, linea=None, n_sim=N_SIMULACIONES):
    sim_l = simular_negbinom(runs_local, n_sim)
    sim_v = simular_negbinom(runs_visitante, n_sim)
    total_sim = sim_l + sim_v
    if linea is None:
        linea = round((np.mean(total_sim)) * 2) / 2
    prob_over = np.mean(total_sim > linea)
    prob_under = np.mean(total_sim < linea)
    resultado = {"prob_over": prob_over, "prob_under": prob_under, "total_esperado": np.mean(total_sim), "linea": linea}
    if cuota_over is not None:
        p_impl_over, p_impl_under = remover_vig(cuota_a_prob(cuota_over), cuota_a_prob(cuota_under))
        resultado["diferencia"] = prob_over - p_impl_over
    return resultado

LIGA_RUNS_5IP = LIGA_RUNS_PROMEDIO * (5/9)

def runs_esperados_f5(era_rival, ip_rival, k_rival, bb_rival, ev_propio, barrel_propio, park_factor):
    era_adj = shrink_era(era_rival, ip_rival, PRIOR_IP_ABRIDOR)
    f_era = era_adj / LIGA_ERA_PROMEDIO
    f_kbb = factor_kbb(k_rival, bb_rival, ip_rival)
    f_abridor = (f_era * 0.70) + (f_kbb * 0.30)
    f_ofensiva = (ev_propio/88.5 * 0.6) + (barrel_propio/0.075 * 0.4)
    return LIGA_RUNS_5IP * f_ofensiva * f_abridor * park_factor

def analizar_f5_completo(equipo_local, equipo_visitante, pitcher_id_local, pitcher_id_visitante,
                          park_factor, fecha_hoy, cuota_f5_local=None, cuota_f5_visitante=None,
                          year=2026, inicio_temporada="2026-03-15"):
    ip_l, k_l, bb_l, era_l = stats_abridor_hasta_hoy(pitcher_id_local, fecha_hoy, year)
    ip_v, k_v, bb_v, era_v = stats_abridor_hasta_hoy(pitcher_id_visitante, fecha_hoy, year)
    ev_l, barrel_l = stats_ofensiva_hasta_hoy(equipo_local, inicio_temporada, fecha_hoy, year)
    ev_v, barrel_v = stats_ofensiva_hasta_hoy(equipo_visitante, inicio_temporada, fecha_hoy, year)
    runs_l = runs_esperados_f5(era_v, ip_v, k_v, bb_v, ev_l, barrel_l, park_factor)
    runs_v = runs_esperados_f5(era_l, ip_l, k_l, bb_l, ev_v, barrel_v, park_factor)
    sim_l = simular_negbinom(runs_l, N_SIMULACIONES)
    sim_v = simular_negbinom(runs_v, N_SIMULACIONES)
    prob_local = np.mean(sim_l > sim_v)
    prob_visitante = np.mean(sim_v > sim_l)
    prob_empate = np.mean(sim_l == sim_v)
    resultado = {"prob_local_f5": prob_local, "prob_visitante_f5": prob_visitante, "prob_empate_f5": prob_empate}
    if cuota_f5_local is not None:
        p_impl_l, p_impl_v = remover_vig(cuota_a_prob(cuota_f5_local), cuota_a_prob(cuota_f5_visitante))
        resultado["diferencia_f5"] = prob_local - p_impl_l
    return resultado

def obtener_cartelera_dia(fecha):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha}&hydrate=probablePitcher,team,venue&gameType=R"
    r = requests.get(url)
    data = r.json()
    partidos_hoy = []
    for fecha_obj in data.get("dates", []):
        for juego in fecha_obj.get("games", []):
            home = juego["teams"]["home"]
            away = juego["teams"]["away"]
            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})
            partidos_hoy.append({
                "local": home["team"]["abbreviation"], "visitante": away["team"]["abbreviation"],
                "venue": juego.get("venue", {}).get("name", "?"),
                "pitcher_local_id": home_pitcher.get("id"), "pitcher_local_nombre": home_pitcher.get("fullName", "No anunciado"),
                "pitcher_visitante_id": away_pitcher.get("id"), "pitcher_visitante_nombre": away_pitcher.get("fullName", "No anunciado"),
            })
    return partidos_hoy

def obtener_lineup_confirmado(equipo_abbrev, fecha):
    team_id = TEAM_IDS.get(equipo_abbrev)
    if not team_id:
        return None
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}"
           f"&date={fecha}&hydrate=lineups,team")
    r = requests.get(url)
    data = r.json()
    try:
        juego = data["dates"][0]["games"][0]
        home_abbr = juego["teams"]["home"]["team"]["abbreviation"]
        lineups = juego.get("lineups", {})
        jugadores = lineups.get("homePlayers", []) if home_abbr == equipo_abbrev else lineups.get("awayPlayers", [])
        if jugadores:
            return [{"id": j.get("id"), "nombre": j.get("fullName", "?")} for j in jugadores]
        return None
    except (KeyError, IndexError):
        return None

def obtener_splits_pitcher(pitcher_id, year):
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=statSplits&group=pitching&season={year}&sitCodes=vl,vr"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        splits = data["stats"][0]["splits"]
    except (KeyError, IndexError):
        return None, None
    ops_vl, ops_vr = None, None
    for s in splits:
        stat = s.get("stat", {})
        ops_str = stat.get("ops")
        bf = stat.get("battersFaced", 0)
        if ops_str is None or bf < 20:
            continue
        try:
            ops_val = float(ops_str)
        except (TypeError, ValueError):
            continue
        codigo = s.get("split", {}).get("code")
        if codigo == "vl":
            ops_vl = ops_val
        elif codigo == "vr":
            ops_vr = ops_val
    return ops_vl, ops_vr

def obtener_batside_lote(lista_ids):
    if not lista_ids:
        return {}
    ids_str = ",".join(str(i) for i in lista_ids if i)
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception:
        return {}
    resultado = {}
    for persona in data.get("people", []):
        lado = persona.get("batSide", {}).get("code")
        resultado[persona["id"]] = lado
    return resultado

def factor_matchup_lr(ops_vl, ops_vr, lista_batside):
    if ops_vl is None or ops_vr is None or not lista_batside:
        return 1.0
    n_l = sum(1 for b in lista_batside if b == "L")
    n_r = sum(1 for b in lista_batside if b == "R")
    total = n_l + n_r
    if total == 0:
        return 1.0
    ops_esperado = (ops_vl * n_l + ops_vr * n_r) / total
    return ops_esperado / LIGA_OPS_CONTRA

def imprimir_matchup_lr(p, fecha_hoy):
    lineup_local = obtener_lineup_confirmado(p['local'], fecha_hoy)
    lineup_visitante = obtener_lineup_confirmado(p['visitante'], fecha_hoy)
    if lineup_local:
        ids_local = [j['id'] for j in lineup_local]
        lados_local = list(obtener_batside_lote(ids_local).values())
        ops_vl_v, ops_vr_v = obtener_splits_pitcher(p['pitcher_visitante_id'], 2026)
        factor_v = factor_matchup_lr(ops_vl_v, ops_vr_v, lados_local)
        print(f"  Matchup {p['pitcher_visitante_nombre']} vs lineup {p['local']}: {factor_v:.3f}")
    if lineup_visitante:
        ids_visitante = [j['id'] for j in lineup_visitante]
        lados_visitante = list(obtener_batside_lote(ids_visitante).values())
        ops_vl_l, ops_vr_l = obtener_splits_pitcher(p['pitcher_local_id'], 2026)
        factor_l = factor_matchup_lr(ops_vl_l, ops_vr_l, lados_visitante)
        print(f"  Matchup {p['pitcher_local_nombre']} vs lineup {p['visitante']}: {factor_l:.3f}")

def obtener_espn_predictor_partido(equipo_local, equipo_visitante, fecha):
    fecha_espn = fecha.replace("-", "")
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={fecha_espn}"
    r = requests.get(url)
    data = r.json()
    for evento in data.get("events", []):
        competencia = evento["competitions"][0]
        equipos = competencia["competitors"]
        home = next(e for e in equipos if e["homeAway"] == "home")
        away = next(e for e in equipos if e["homeAway"] == "away")
        if home["team"]["abbreviation"] == equipo_local and away["team"]["abbreviation"] == equipo_visitante:
            pred = competencia.get("predictor")
            if pred:
                return {"prob_local": pred.get("homeTeam", {}).get("gameProjection"),
                        "prob_visitante": pred.get("awayTeam", {}).get("gameProjection")}
            return None
    return None

def obtener_lesiones_espn(equipo_abbrev):
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams/{equipo_abbrev.lower()}/injuries"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        lesiones = []
        for item in data.get("injuries", []):
            for atleta in item.get("injuries", []):
                nombre = atleta.get("athlete", {}).get("displayName", "?")
                estado = atleta.get("status", "?")
                lesiones.append(f"{nombre} ({estado})")
        return lesiones
    except Exception:
        return []

def contexto_cualitativo(equipo_local, equipo_visitante, fecha):
    resultado = {}
    resultado["clima"] = obtener_clima(equipo_local, fecha)
    resultado["espn_predictor"] = obtener_espn_predictor_partido(equipo_local, equipo_visitante, fecha)
    resultado["lesiones_local"] = obtener_lesiones_espn(equipo_local)
    resultado["lesiones_visitante"] = obtener_lesiones_espn(equipo_visitante)
    return resultado

def obtener_winpct_equipo(team_abbrev, fecha):
    team_id = TEAM_IDS.get(team_abbrev)
    if not team_id:
        return None
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2026&date={fecha}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        for record in data.get("records", []):
            for team_record in record.get("teamRecords", []):
                if team_record.get("team", {}).get("id") == team_id:
                    lr = team_record.get("leagueRecord", {})
                    wins = lr.get("wins", 0)
                    losses = lr.get("losses", 0)
                    total = wins + losses
                    return wins / total if total > 0 else 0.5
    except Exception:
        return None
    return None

def factor_winpct(equipo_local, equipo_visitante, fecha):
    wp_local = obtener_winpct_equipo(equipo_local, fecha)
    wp_visitante = obtener_winpct_equipo(equipo_visitante, fecha)
    if wp_local is None or wp_visitante is None:
        return None
    if wp_visitante == 0:
        return None
    return wp_local / wp_visitante

def imprimir_winpct(p, fecha_hoy):
    ratio = factor_winpct(p['local'], p['visitante'], fecha_hoy)
    if ratio is not None:
        print(f"  Win% ratio {p['local']}/{p['visitante']}: {ratio:.3f}")
def analizar_handicap_multiple(runs_local, runs_visitante, favorito="local",
                                 lineas=(1.5,), cuotas_handicap=None, n_sim=N_SIMULACIONES):
    sim_l = simular_negbinom(runs_local, n_sim)
    sim_v = simular_negbinom(runs_visitante, n_sim)
    margen = sim_l - sim_v if favorito == "local" else sim_v - sim_l
    resultados = {}
    for linea in lineas:
        prob_cubre = np.mean(margen > linea)
        resultado_linea = {"prob_cubre": prob_cubre}
        if cuotas_handicap and linea in cuotas_handicap:
            prob_mercado = cuota_a_prob(cuotas_handicap[linea])
            resultado_linea["prob_mercado"] = prob_mercado
            resultado_linea["diferencia"] = prob_cubre - prob_mercado
        resultados[linea] = resultado_linea
    return resultados
