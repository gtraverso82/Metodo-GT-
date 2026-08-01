"""
Script puntual (tipo test_splits.py) para evaluar manualmente partidos que la
corrida automatica salto por falta de cuotas en el momento de la ejecucion.

Uso: python test_splits.py

Resuelve el ID de cada abridor por nombre via la MLB Stats API
(people/search), y corre analizar_partido_hoy() con las cuotas de mercado
capturadas manualmente (screenshot del sportsbook).
"""

import requests
from motor import analizar_partido_hoy, PARK_FACTORS

FECHA_HOY = "2026-08-01"

def buscar_pitcher_id(nombre):
    """Busca el ID de un jugador por nombre via MLB Stats API. Retorna el
    primer resultado que coincida (revisar manualmente si hay ambiguedad)."""
    url = f"https://statsapi.mlb.com/api/v1/people/search?names={nombre}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        personas = data.get("people", [])
        if not personas:
            print(f"  ADVERTENCIA: no se encontro a '{nombre}' en MLB Stats API")
            return None, None
        p = personas[0]
        return p.get("id"), p.get("fullName")
    except Exception as e:
        print(f"  ERROR buscando '{nombre}': {e}")
        return None, None

# --- Partidos a evaluar manualmente (capturados de screenshot del sportsbook) ---
PARTIDOS = [
    {
        "local": "TB", "visitante": "CWS",
        "pitcher_local_nombre": "Drew Rasmussen", "pitcher_visitante_nombre": "Noah Schultz",
        "cuota_ml_local": -185, "cuota_ml_visitante": 140,
    },
    {
        "local": "CLE", "visitante": "AZ",
        "pitcher_local_nombre": "Parker Messick", "pitcher_visitante_nombre": "Kohl Drake",
        "cuota_ml_local": -196, "cuota_ml_visitante": 145,
    },
    {
        "local": "SD", "visitante": "SF",
        "pitcher_local_nombre": "Walker Buehler", "pitcher_visitante_nombre": "Tyler Mahle",
        "cuota_ml_local": -161, "cuota_ml_visitante": 120,
    },
]

def main():
    print(f"=== Evaluacion manual - partidos sin cuotas en corrida automatica ({FECHA_HOY}) ===\n")

    for p in PARTIDOS:
        print(f"--- {p['visitante']} @ {p['local']} ---")

        id_local, nombre_local_confirmado = buscar_pitcher_id(p["pitcher_local_nombre"])
        id_visitante, nombre_visitante_confirmado = buscar_pitcher_id(p["pitcher_visitante_nombre"])

        if id_local is None or id_visitante is None:
            print(f"  No se pudo resolver ambos abridores, se omite este partido.\n")
            continue

        print(f"  Abridor local: {nombre_local_confirmado} (ID {id_local})")
        print(f"  Abridor visitante: {nombre_visitante_confirmado} (ID {id_visitante})")

        park_factor = PARK_FACTORS.get(p["local"], 1.00)

        try:
            resultado = analizar_partido_hoy(
                equipo_local=p["local"], equipo_visitante=p["visitante"],
                pitcher_id_local=id_local, pitcher_id_visitante=id_visitante,
                park_factor=park_factor,
                cuota_ml_local=p["cuota_ml_local"], cuota_ml_visitante=p["cuota_ml_visitante"],
                fecha_hoy=FECHA_HOY
            )
            print(f"  Prob. modelo (local): {resultado['prob_local']:.1%}")
            print(f"  Runs esperados: {p['local']}={resultado['runs_local']:.2f} | "
                  f"{p['visitante']}={resultado['runs_visitante']:.2f}")
            print(f"  Bandera: {resultado['bandera']} | Confianza: {resultado['confianza']}%")
            print(f"  Recomendacion: {resultado['recomendacion']}")
        except Exception as e:
            print(f"  ERROR al analizar el partido: {e}")

        print()

    print("=== Evaluacion completa ===")
    print("NOTA: estos resultados NO se guardaron en diagnostico_total ni en")
    print("contexto_partido (esto es solo evaluacion puntual, no la corrida oficial).")
    print("Si se quiere conservar el resultado, guardarlo manualmente via SQL.")


if __name__ == "__main__":
    main()
