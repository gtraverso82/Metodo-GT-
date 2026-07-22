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

def descargar_y_filtrar_dataset(anio_inicio=2022, anio_fin=2023):
    print("Descargando dataset de cuotas (76 MB, puede tardar)...")
    r = requests.get(DATASET_URL, timeout=300)
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
