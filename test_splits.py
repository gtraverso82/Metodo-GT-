from datetime import datetime
from motor import (
    obtener_cartelera_dia, analizar_partido_hoy, analizar_total, PARK_FACTORS,
    imprimir_matchup_lr, imprimir_winpct, proyectar_ponches
)

def analizar_doubleheader():
    fecha_hoy = "2026-07-28"
    partidos = obtener_cartelera_dia(fecha_hoy)

    juegos_cle_cin = [p for p in partidos if p['local'] == 'CIN' and p['visitante'] == 'CLE']

    if not juegos_cle_cin:
        print("No se encontraron juegos CLE @ CIN para hoy.")
        print("Partidos disponibles hoy:")
        for p in partidos:
            print(f"  {p['visitante']} @ {p['local']}")
        return

    for idx, juego in enumerate(juegos_cle_cin, 1):
        print(f"\n=== CLE @ CIN (Juego {idx} de doble cartelera) ===")
        print(f"Abridores: {juego['pitcher_visitante_nombre']} vs {juego['pitcher_local_nombre']}")

        park_factor = PARK_FACTORS.get('CIN', 1.00)
        resultado = analizar_partido_hoy(
            equipo_local='CIN', equipo_visitante='CLE',
            pitcher_id_local=juego['pitcher_local_id'], pitcher_id_visitante=juego['pitcher_visitante_id'],
            park_factor=park_factor, cuota_ml_local=-170, cuota_ml_visitante=142,
            fecha_hoy=fecha_hoy
        )
        print(f"Prob. modelo: CIN {resultado['prob_local']:.1%} | CLE {1-resultado['prob_local']:.1%}")
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

        total_resultado = analizar_total(resultado['runs_local'], resultado['runs_visitante'], linea=9.5)
        print(f"Total: Linea 9.5 | Proyectado {total_resultado['total_esperado']:.2f}")

if __name__ == "__main__":
    analizar_doubleheader()
