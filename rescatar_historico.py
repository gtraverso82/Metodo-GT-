import os
from supabase import create_client
from motor import (
    obtener_cartelera_dia, obtener_total_espn, analizar_partido_hoy, analizar_total, PARK_FACTORS
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

RESULTADOS_REALES = {
    ("2026-07-17", "SD", "KC"): (6, 7),
    ("2026-07-17", "CHC", "MIN"): (2, 5),
    ("2026-07-17", "TEX", "ATL"): (1, 15),
    ("2026-07-17", "ATH", "WSH"): (4, 23),
    ("2026-07-17", "NYY", "LAD"): (1, 2),
    ("2026-07-17", "MIA", "MIL"): (1, 2),
    ("2026-07-19", "TEX", "ATL"): (5, 8),
    ("2026-07-19", "TB", "BOS"): (3, 5),
    ("2026-07-19", "SD", "KC"): (6, 7),
    ("2026-07-19", "CHC", "MIN"): (2, 5),
    ("2026-07-20", "MIN", "CLE"): (4, 13),
    ("2026-07-20", "PIT", "NYY"): (5, 8),
    ("2026-07-20", "TOR", "TB"): (1, 7),
    ("2026-07-20", "BAL", "BOS"): (5, 1),
    ("2026-07-20", "LAD", "PHI"): (7, 10),
    ("2026-07-20", "SD", "ATL"): (2, 3),
    ("2026-07-20", "SF", "KC"): (3, 4),
    ("2026-07-20", "NYM", "MIL"): (3, 8),
    ("2026-07-20", "DET", "CHC"): (8, 6),
    ("2026-07-20", "MIA", "HOU"): (5, 8),
    ("2026-07-20", "WSH", "COL"): (7, 3),
    ("2026-07-20", "CIN", "SEA"): (0, 8),
    ("2026-07-20", "STL", "LAA"): (2, 3),
}

def correr_rescate():
    for fecha in ["2026-07-17", "2026-07-19", "2026-07-20"]:
        partidos = obtener_cartelera_dia(fecha)
        print(f"=== {fecha}: {len(partidos)} partidos encontrados ===")
        for p in partidos:
            key = (fecha, p['visitante'], p['local'])
            if key not in RESULTADOS_REALES:
                continue
            if p['pitcher_local_id'] is None or p['pitcher_visitante_id'] is None:
                print(f"Sin abridor: {key}")
                continue
            score_v, score_l = RESULTADOS_REALES[key]
            park_factor = PARK_FACTORS.get(p['local'], 1.00)
            try:
                resultado = analizar_partido_hoy(
                    equipo_local=p['local'], equipo_visitante=p['visitante'],
                    pitcher_id_local=p['pitcher_local_id'], pitcher_id_visitante=p['pitcher_visitante_id'],
                    park_factor=park_factor, cuota_ml_local=-110, cuota_ml_visitante=-110,
                    fecha_hoy=fecha, year=2026
                )
                total_info = obtener_total_espn(p['local'], p['visitante'], fecha)
                linea = total_info['linea'] if total_info else float(score_v + score_l)
                total_resultado = analizar_total(resultado['runs_local'], resultado['runs_visitante'], linea=linea)
                game_id = f"{p['visitante']}@{p['local']}_{fecha.replace('-','')}"
                supabase.table("diagnostico_total").insert({
                    "game_id": game_id, "fecha": fecha,
                    "total_esperado_modelo": total_resultado['total_esperado'],
                    "linea_mercado": linea,
                    "diferencia": total_resultado['total_esperado'] - linea,
                    "park_factor": park_factor,
                    "era_local": 0, "era_visitante": 0,
                    "total_real": score_v + score_l
                }).execute()
                print(f"OK {game_id}: modelo={total_resultado['total_esperado']:.2f} linea={linea} real={score_v+score_l}")
            except Exception as e:
                print(f"Error {key}: {e}")

if __name__ == "__main__":
    correr_rescate()
