import os
from datetime import datetime
from supabase import create_client
from motor import (
    obtener_cartelera_dia, obtener_cuotas_espn, obtener_handicap_espn, obtener_total_espn,
    analizar_partido_hoy, analizar_total, analizar_f5_completo,
    PARK_FACTORS
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def guardar_snapshot(game_id, bookmaker_key, market_key, outcome_name, price,
                      point=None, modelo_prob=None, modelo_bandera=None,
                      tipo_snapshot='apertura'):
    supabase.table("odds_snapshots").insert({
        "game_id": game_id, "bookmaker_key": bookmaker_key, "market_key": market_key,
        "outcome_name": outcome_name, "point": point, "price": price,
        "modelo_prob_en_captura": modelo_prob, "modelo_bandera": modelo_bandera,
        "tipo_snapshot": tipo_snapshot
    }).execute()
    print(f"Snapshot: {game_id} - {market_key} - {outcome_name}")

def guardar_diagnostico_total(game_id, fecha, total_esperado, linea_mercado,
                                park_factor, era_local, era_visitante):
    diferencia = total_esperado - linea_mercado
    supabase.table("diagnostico_total").insert({
        "game_id": game_id, "fecha": fecha, "total_esperado_modelo": total_esperado,
        "linea_mercado": linea_mercado, "diferencia": diferencia,
        "park_factor": park_factor, "era_local": era_local, "era_visitante": era_visitante
    }).execute()
    print(f"Diagnostico: {game_id} (dif: {diferencia:+.2f})")
