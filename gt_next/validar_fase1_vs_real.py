"""
GT Next - Validación de Fase 1 contra datos reales
====================================================

QUÉ VALIDA ESTE SCRIPT (y qué NO):

  SÍ valida: si la distribución agregada de carreras que produce el
  simulador de Fase 1 (media, varianza, forma) se parece a la
  distribución real de carreras observada en los 3,642 juegos de
  backtesting_resultados (2022-2023).

  NO valida (todavía): precisión predictiva partido-por-partido
  (Brier Score, Log Loss) contra GT Classic. Fase 1 usa probabilidades
  de liga fijas — no diferencia entre equipos. Esa comparación tiene
  sentido a partir de Fase 2 (lineups).
"""

import os
from collections import Counter

from supabase import create_client

from simulador_fase1 import monte_carlo, resumen_distribucion

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
TABLA = "backtesting_resultados"
COL_CARRERAS_LOCAL = "runs_reales_local"
COL_CARRERAS_VISITANTE = "runs_reales_visitante"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]


def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def inspeccionar_esquema():
    sb = get_client()
    muestra = sb.table(TABLA).select("*").limit(3).execute()
    if not muestra.data:
        print(f"La tabla {TABLA} está vacía o no se pudo leer.")
        return
    print(f"Columnas disponibles en {TABLA}:")
    for col in muestra.data[0].keys():
        print(f"  - {col}")
    print("\nMuestra de 3 filas:")
    for fila in muestra.data:
        print(fila)


def cargar_carreras_reales() -> list[int]:
    sb = get_client()
    carreras_totales = []
    page_size = 1000
    start = 0
    while True:
        resp = (
            sb.table(TABLA)
            .select(f"{COL_CARRERAS_LOCAL},{COL_CARRERAS_VISITANTE}")
            .range(start, start + page_size - 1)
            .execute()
        )
        filas = resp.data
        if not filas:
            break
        for fila in filas:
            local = fila.get(COL_CARRERAS_LOCAL)
            visitante = fila.get(COL_CARRERAS_VISITANTE)
            if local is not None and visitante is not None:
                carreras_totales.append(local + visitante)
        if len(filas) < page_size:
            break
        start += page_size

    print(f"Cargados {len(carreras_totales)} juegos reales de {TABLA}.")
    return carreras_totales


def bootstrap_diferencia_medias(
    reales: list[int], simulados: list[int], n_iter: int = 2000
) -> dict:
    import random

    diferencias = []
    for _ in range(n_iter):
        muestra_real = [random.choice(reales) for _ in range(len(reales))]
        muestra_sim = [random.choice(simulados) for _ in range(len(simulados))]
        diferencias.append(
            (sum(muestra_sim) / len(muestra_sim)) - (sum(muestra_real) / len(muestra_real))
        )
    diferencias.sort()
    n = len(diferencias)
    return {
        "diferencia_media_puntual": round(
            (sum(simulados) / len(simulados)) - (sum(reales) / len(reales)), 3
        ),
        "ic_95_inferior": round(diferencias[int(0.025 * n)], 3),
        "ic_95_superior": round(diferencias[int(0.975 * n)], 3),
    }


def comparar_distribuciones(reales: list[int], simulados: list[int]) -> None:
    res_reales = resumen_distribucion(reales)
    res_sim = resumen_distribucion(simulados)

    print("\n=== Comparación distribución: REAL vs SIMULADO (Fase 1) ===")
    print(f"{'Métrica':<20}{'Real':>12}{'Simulado':>12}")
    print(f"{'Media':<20}{res_reales['media_carreras']:>12}{res_sim['media_carreras']:>12}")
    print(f"{'Desv. estándar':<20}{res_reales['desv_estandar']:>12}{res_sim['desv_estandar']:>12}")
    print(f"{'Mínimo':<20}{res_reales['min']:>12}{res_sim['min']:>12}")
    print(f"{'Máximo':<20}{res_reales['max']:>12}{res_sim['max']:>12}")

    ci = bootstrap_diferencia_medias(reales, simulados, n_iter=2000)
    print("\n=== Bootstrap (2,000 iter) — diferencia de medias (Simulado - Real) ===")
    print(f"Diferencia puntual: {ci['diferencia_media_puntual']}")
    print(f"IC 95%: [{ci['ic_95_inferior']}, {ci['ic_95_superior']}]")
    if ci["ic_95_inferior"] <= 0 <= ci["ic_95_superior"]:
        print("-> El IC 95% incluye 0: no hay evidencia de diferencia significativa.")
        print("   Fase 1 pasa la validación distribucional.")
    else:
        print("-> El IC 95% NO incluye 0: hay diferencia significativa.")
        print("   Revisar PA_PROBS o la lógica de avance de corredores.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "inspeccionar":
        inspeccionar_esquema()
        sys.exit(0)

    print("Cargando carreras reales de backtesting_resultados...")
    reales = cargar_carreras_reales()

    print("Corriendo Monte Carlo del motor de Fase 1...")
    n = len(reales) if reales else 10_000
    local_sim = monte_carlo(n_juegos=n, semilla=42)
    visitante_sim = monte_carlo(n_juegos=n, semilla=43)
    simulados = [a + b for a, b in zip(local_sim, visitante_sim)]

    comparar_distribuciones(reales, simulados)
