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
def correr_jornada():
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Corrida diaria: {fecha_hoy} ===")
    partidos = obtener_cartelera_dia(fecha_hoy)
    print(f"Partidos encontrados: {len(partidos)}")

    for p in partidos:
        if p['pitcher_local_id'] is None or p['pitcher_visitante_id'] is None:
            print(f"Saltando: {p['visitante']} @ {p['local']} - abridor no confirmado")
            continue

        cuota_local, cuota_visitante = obtener_cuotas_espn(p['local'], p['visitante'], fecha_hoy)
        if cuota_local is None:
            print(f"Sin cuotas: {p['visitante']} @ {p['local']}")
            continue

        park_factor = PARK_FACTORS.get(p['local'], 1.00)

        try:
            resultado = analizar_partido_hoy(
                equipo_local=p['local'], equipo_visitante=p['visitante'],
                pitcher_id_local=p['pitcher_local_id'], pitcher_id_visitante=p['pitcher_visitante_id'],
                park_factor=park_factor, cuota_ml_local=cuota_local, cuota_ml_visitante=cuota_visitante,
                fecha_hoy=fecha_hoy
            )
            game_id = f"{p['visitante']}@{p['local']}_{fecha_hoy.replace('-','')}"

            guardar_snapshot(game_id, "espn", "moneyline", p['local'], cuota_local,
                              modelo_prob=resultado['prob_local'], modelo_bandera=resultado['bandera'])
            guardar_snapshot(game_id, "espn", "moneyline", p['visitante'], cuota_visitante,
                              modelo_prob=1-resultado['prob_local'], modelo_bandera=resultado['bandera'])

            total_info = obtener_total_espn(p['local'], p['visitante'], fecha_hoy)
            if total_info:
                total_resultado = analizar_total(resultado['runs_local'], resultado['runs_visitante'],
                                                    total_info['over_odds'], total_info['under_odds'], total_info['linea'])
                guardar_diagnostico_total(game_id, fecha_hoy, total_resultado['total_esperado'],
                                            total_info['linea'], park_factor,
                                            resultado.get('era_local', 0), resultado.get('era_visitante', 0))

            print(f"{p['visitante']} @ {p['local']}: {resultado['recomendacion']} (bandera: {resultado['bandera']})")

        except Exception as e:
            print(f"Error en {p['visitante']} @ {p['local']}: {e}")

    print("=== Corrida completa ===")

if __name__ == "__main__":
    correr_jornada()
