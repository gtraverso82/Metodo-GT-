from datetime import datetime
from motor import (
    obtener_cartelera_dia, analizar_partido_hoy, analizar_total, PARK_FACTORS,
    imprimir_matchup_lr, imprimir_winpct, proyectar_ponches
)

def analizar_puntual(local, visitante, cuota_local, cuota_visitante, linea_total=None, cuota_over=None, cuota_under=None):
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    partidos = obtener_cartelera_dia(fecha_hoy)
    juego = next((p for p in partidos if p['local'] == local and p['visitante'] == visitante), None)

    if not juego:
        print(f"Partido {visitante} @ {local} no encontrado.")
        return

    print(f"\n=== {visitante} @ {local} ===")
    print(f"Abridores: {juego['pitcher_visitante_nombre']} vs {juego['pitcher_local_nombre']}")

    park_factor = PARK_FACTORS.get(local, 1.00)
    resultado = analizar_partido_hoy(
        equipo_local=local, equipo_visitante=visitante,
        pitcher_id_local=juego['pitcher_local_id'], pitcher_id_visitante=juego['pitcher_visitante_id'],
        park_factor=park_factor, cuota_ml_local=cuota_local, cuota_ml_visitante=cuota_visitante,
        fecha_hoy=fecha_hoy
    )
    print(f"Prob. modelo: {local} {resultado['prob_local']:.1%} | {visitante} {1-resultado['prob_local']:.1%}")
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

    if linea_total:
        total_resultado = analizar_total(resultado['runs_local'], resultado['runs_visitante'],
                                            cuota_over, cuota_under, linea_total)
        print(f"Total: Linea {linea_total} | Proyectado {total_resultado['total_esperado']:.2f}")

if __name__ == "__main__":
    analizar_puntual('WSH', 'AZ', cuota_local=106, cuota_visitante=-119, linea_total=9.5, cuota_over=-111, cuota_under=-106)
    analizar_puntual('CWS', 'HOU', cuota_local=-130, cuota_visitante=115, linea_total=9.0, cuota_over=-102, cuota_under=-115)
