import requests
from motor import (
    obtener_cartelera_dia, analizar_partido_hoy, analizar_handicap_multiple, PARK_FACTORS
)

def probar_handicap():
    fecha = "2026-07-23"
    partidos = obtener_cartelera_dia(fecha)
    juego = next(p for p in partidos if p['local'] == 'ATL' and p['visitante'] == 'SD')

    resultado = analizar_partido_hoy(
        equipo_local='ATL', equipo_visitante='SD',
        pitcher_id_local=juego['pitcher_local_id'], pitcher_id_visitante=juego['pitcher_visitante_id'],
        park_factor=PARK_FACTORS.get('ATL', 1.00),
        cuota_ml_local=-259, cuota_ml_visitante=209,
        fecha_hoy=fecha
    )
    print(f"Runs esperados: ATL={resultado['runs_local']:.2f} | SD={resultado['runs_visitante']:.2f}")

    handicap = analizar_handicap_multiple(
        runs_local=resultado['runs_local'], runs_visitante=resultado['runs_visitante'],
        favorito="local", lineas=(1.5,), cuotas_handicap={1.5: -127}
    )
    print(handicap)

if __name__ == "__main__":
    probar_handicap()
