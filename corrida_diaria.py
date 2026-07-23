import os
from datetime import datetime
from supabase import create_client
from motor import (
    obtener_cartelera_dia, obtener_cuotas_espn, obtener_handicap_espn, obtener_total_espn,
    analizar_partido_hoy, analizar_total, analizar_f5_completo,
    PARK_FACTORS, imprimir_matchup_lr, contexto_cualitativo, imprimir_winpct,
    proyectar_ponches
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
    print(f"Diagnostico: {game_id} | Linea mercado: {linea_mercado} | Proyectado modelo: {total_esperado:.2f} | Diferencia: {diferencia:+.2f}")

def correr_jornada():
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Corrida diaria: {fecha_hoy} ===")
    partidos = obtener_cartelera_dia(fecha_hoy)
    print(f"Partidos encontrados: {len(partidos)}")

    ranking_del_dia = []
    ranking_ponches_del_dia = []
    ranking_totales_del_dia = []

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

            favorito = p['local'] if resultado['prob_local'] >= 0.5 else p['visitante']
            prob_favorito = resultado['prob_local'] if resultado['prob_local'] >= 0.5 else 1 - resultado['prob_local']
            ranking_del_dia.append({
                "partido": f"{p['visitante']} @ {p['local']}",
                "favorito": favorito,
                "prob": prob_favorito,
                "bandera": resultado['bandera']
            })

            try:
                imprimir_matchup_lr(p, fecha_hoy)
            except Exception as e:
                print(f"  (matchup L/R no disponible: {e})")

            try:
                imprimir_winpct(p, fecha_hoy)
            except Exception as e:
                print(f"  (win% no disponible: {e})")

            try:
                ponches_l = proyectar_ponches(p['pitcher_local_id'], fecha_hoy, 2026)
                ponches_v = proyectar_ponches(p['pitcher_visitante_id'], fecha_hoy, 2026)
                if ponches_l is not None:
                    print(f"  Ponches proyectados {p['pitcher_local_nombre']}: {ponches_l}")
                    ranking_ponches_del_dia.append({
                        "pitcher": p['pitcher_local_nombre'], "equipo": p['local'], "ponches": ponches_l
                    })
                if ponches_v is not None:
                    print(f"  Ponches proyectados {p['pitcher_visitante_nombre']}: {ponches_v}")
                    ranking_ponches_del_dia.append({
                        "pitcher": p['pitcher_visitante_nombre'], "equipo": p['visitante'], "ponches": ponches_v
                    })
            except Exception as e:
                print(f"  (ponches no disponibles: {e})")

            try:
                ctx = contexto_cualitativo(p['local'], p['visitante'], fecha_hoy)
                if ctx['clima']:
                    print(f"  Clima: {ctx['clima']['temperatura_f']}F, viento {ctx['clima']['viento_mph']}mph")
                if ctx['espn_predictor']:
                    print(f"  ESPN Predictor: {p['local']} {ctx['espn_predictor']['prob_local']}% - {p['visitante']} {ctx['espn_predictor']['prob_visitante']}%")
                if ctx['lesiones_local']:
                    print(f"  Lesiones {p['local']}: {ctx['lesiones_local']}")
                if ctx['lesiones_visitante']:
                    print(f"  Lesiones {p['visitante']}: {ctx['lesiones_visitante']}")
            except Exception as e:
                print(f"  (contexto cualitativo no disponible: {e})")

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
                ranking_totales_del_dia.append({
                    "partido": f"{p['visitante']} @ {p['local']}",
                    "linea": total_info['linea'],
                    "proyectado": total_resultado['total_esperado'],
                    "diferencia": total_resultado['total_esperado'] - total_info['linea']
                })

            print(f"{p['visitante']} @ {p['local']}: {resultado['recomendacion']} (bandera: {resultado['bandera']})")

        except Exception as e:
            print(f"Error en {p['visitante']} @ {p['local']}: {e}")

    print("\n=== RANKING DEL DIA - FAVORITOS (SOLO SEGUIMIENTO Y APRENDIZAJE, NO ES RECOMENDACION DE APUESTA) ===")
    ranking_ordenado = sorted(ranking_del_dia, key=lambda x: x['prob'], reverse=True)
    for i, r in enumerate(ranking_ordenado, 1):
        print(f"{i}. {r['favorito']} favorito ({r['prob']:.1%}) - {r['partido']} [bandera: {r['bandera']}]")

    print("\n=== RANKING DEL DIA - PONCHES PROYECTADOS (SOLO SEGUIMIENTO Y APRENDIZAJE) ===")
    ranking_ponches_ordenado = sorted(ranking_ponches_del_dia, key=lambda x: x['ponches'], reverse=True)
    for i, r in enumerate(ranking_ponches_ordenado, 1):
        print(f"{i}. {r['pitcher']} ({r['equipo']}): {r['ponches']} ponches proyectados")

    print("\n=== RANKING DEL DIA - TOTALES (LINEA VS PROYECCION, SOLO SEGUIMIENTO) ===")
    ranking_totales_ordenado = sorted(ranking_totales_del_dia, key=lambda x: abs(x['diferencia']), reverse=True)
    for i, r in enumerate(ranking_totales_ordenado, 1):
        direccion = "Over" if r['diferencia'] > 0 else "Under"
        print(f"{i}. {r['partido']}: Linea {r['linea']} | Proyectado {r['proyectado']:.2f} | {direccion} ({r['diferencia']:+.2f})")

    print("\n=== Corrida completa ===")

if __name__ == "__main__":
    correr_jornada()
