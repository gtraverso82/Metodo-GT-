from datetime import datetime
from motor import (
    obtener_cartelera_dia, obtener_cuotas_espn, obtener_total_espn,
    analizar_partido_hoy, analizar_total, PARK_FACTORS,
    imprimir_matchup_lr, imprimir_winpct, proyectar_ponches
)

def analizar_partido_puntual():
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    partidos = obtener_cartelera_dia(fecha_hoy)
    juego = next((p for p in partidos if p['local'] == 'MIL' and p['visitante'] == 'COL'), None)

    if not juego:
        print("Partido no encontrado en la cartelera de hoy.")
        return

    print(f"Partido: {juego['visitante']} @ {juego['local']}")
    print(f"Abridores: {juego['pitcher_visitante_nombre']} vs {juego['pitcher_local_nombre']}")

    cuota_local, cuota_visitante = obtener_cuotas_espn('MIL', 'COL', fecha_hoy)
    if cuota_local is None:
        print("Sin cuotas disponibles todavia.")
        return
    print(f"Cuotas: MIL {cuota_local:+d} | COL {cuota_visitante:+d}")

    park_factor = PARK_FACTORS.get('MIL', 1.00)
    resultado = analizar_partido_hoy(
        equipo_local='MIL', equipo_visitante='COL',
        pitcher_id_local=juego['pitcher_local_id'], pitcher_id_visitante=juego['pitcher_visitante_id'],
        park_factor=park_factor, cuota_ml_local=cuota_local, cuota_ml_visitante=cuota_visitante,
        fecha_hoy=fecha_hoy
    )
    print(f"\nProb. modelo: MIL {resultado['prob_local']:.1%} | COL {1-resultado['prob_local']:.1%}")
    print(f"Bandera: {resultado['bandera']} | Recomendacion: {resultado['recomendacion']}")

    try:
        imprimir_matchup_lr(juego, fecha_hoy)
    except Exception as e:
        print(f"(matchup no disponible: {e})")

    try:
        imprimir_winpct(juego, fecha_hoy)
    except Exception as e:
        print(f"(win% no disponible: {e})")

    try:
        k_local = proyectar_ponches(juego['pitcher_local_id'], fecha_hoy, 2026)
        k_visitante = proyectar_ponches(juego['pitcher_visitante_id'], fecha_hoy, 2026)
        print(f"Ponches proyectados: {juego['pitcher_local_nombre']} {k_local} | {juego['pitcher_visitante_nombre']} {k_visitante}")
    except Exception as e:
        print(f"(ponches no disponibles: {e})")

    total_info = obtener_total_espn('MIL', 'COL', fecha_hoy)
    if total_info:
        total_resultado = analizar_total(resultado['runs_local'], resultado['runs_visitante'],
                                            total_info['over_odds'], total_info['under_odds'], total_info['linea'])
        print(f"\nTotal: Linea {total_info['linea']} | Proyectado {total_resultado['total_esperado']:.2f}")
    else:
        print("\nTotal: no disponible")

if __name__ == "__main__":
    analizar_partido_puntual()
