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
