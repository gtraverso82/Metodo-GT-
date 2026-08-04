"""
GT Next - Validación de Fase 1 contra datos reales
====================================================

QUÉ VALIDA ESTE SCRIPT (y qué NO):

  SÍ valida: si la distribución agregada de carreras que produce el
  simulador de Fase 1 (media, varianza, forma) se parece a la
  distribución real de carreras observada en los 3,642 juegos de
  backtesting_resultados (2022-2023). Esto confirma si la mecánica
  del motor (estados base-out, avance de corredores, Monte Carlo)
  está bien construida.

  NO valida (todavía): precisión predictiva partido-por-partido
  (Brier Score, Log Loss) contra GT Classic. Fase 1 usa probabilidades
  de liga fijas — no diferencia entre equipos, así que predice lo
  mismo para cualquier partido. Esa comparación solo tiene sentido
  a partir de Fase 2 (lineups), cuando el motor pueda diferenciar
  ofensivas. Hacerla ahora daría una "derrota" de Fase 1 que no
  significa nada real, solo que le falta la información que Fase 2
  va a agregar.

PASO 0 (ejecutar primero, una sola vez): inspeccionar el esquema real
de backtesting_resultados, porque los nombres de columna abajo son
un supuesto razonable, no un hecho confirmado. Ajustar CONFIG según
lo que devuelva inspeccionar_esquema().
"""

import os
from collections import Counter

from supabase import create_client

# Importa el motor de Fase 1 (debe estar en el mismo repo, gt_next/)
from simulador_fase1 import monte_carlo, resumen_distribucion

# ---------------------------------------------------------------------
# CONFIG — ajustar tras correr inspeccionar_esquema()
# ---------------------------------------------------------------------
TABLA = "backtesting_resultados"
COL_CARRERAS_LOCAL = "runs_reales_local"
COL_CARRERAS_VISITANTE = "runs_reales_visitante"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]


def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def extraer_equipos_unicos():
    """
    Extrae los códigos de equipo reales desde game_id (formato
    'VISITANTE@LOCAL_YYYY-MM-DD'), en vez de adivinar abreviaturas.
    Evita repetir el bug de mismatch AZ/CWS que ya costó tiempo en
    la integración con ESPN.
    """
    sb = get_client()
    equipos = set()
    page_size = 1000
    start = 0
    while True:
        resp = sb.table(TABLA).select("game_id").range(start, start + page_size - 1).execute()
        filas = resp.data
        if not filas:
            break
        for fila in filas:
            gid = fila.get("game_id", "")
            partes = gid.rsplit("_", 1)  # separa la fecha del final
            if len(partes) == 2 and "@" in partes[0]:
                visitante, local = partes[0].split("@")
                equipos.add(visitante)
                equipos.add(local)
        if len(filas) < page_size:
            break
        start += page_size

    print(f"Equipos únicos encontrados ({len(equipos)}):")
    for eq in sorted(equipos):
        print(f"  - {eq}")
    return sorted(equipos)


def inspeccionar_esquema():
    """Corre esto primero para confirmar nombres de columna reales."""
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
    """
    Trae carreras totales (local + visitante) por juego, paginando
    completo con .range() para no quedar truncado en 1,000 filas.
    """
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
    """
    Bootstrap para intervalo de confianza al 95% de la diferencia
    entre la media real y la media simulada (carreras totales por juego).
    """
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
        print("   La mecánica del motor produce un ambiente de anotación consistente")
        print("   con la realidad. Fase 1 pasa la validación distribucional.")
    else:
        print("-> El IC 95% NO incluye 0: hay diferencia significativa.")
        print("   Antes de avanzar a Fase 2, revisar probabilidades de PA_PROBS")
        print("   o la lógica de avance de corredores en simulador_fase1.py.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "inspeccionar":
        inspeccionar_esquema()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "equipos":
        extraer_equipos_unicos()
        sys.exit(0)

    print("Cargando carreras reales de backtesting_resultados...")
    reales = cargar_carreras_reales()

    print("Corriendo Monte Carlo del motor de Fase 1 (carreras totales por juego = local + visitante)...")
    n = len(reales) if reales else 10_000
    local_sim = monte_carlo(n_juegos=n, semilla=42)
    visitante_sim = monte_carlo(n_juegos=n, semilla=43)
    simulados = [a + b for a, b in zip(local_sim, visitante_sim)]

    comparar_distribuciones(reales, simulados)