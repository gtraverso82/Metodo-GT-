"""
GT Next - Fase 1: Simulador básico por aparición al plato (PA)
================================================================

Roadmap GT Next: simulador básico -> lineups -> bullpen -> matchups ->
clima -> defensa -> framing.

Esta fase NO incluye:
  - Bullpen ni cambios de pitcher (un solo "lanzador genérico" todo el juego)
  - Lineups específicos (probabilidades de PA = promedio de liga fijo)
  - Doble matanza ni elevado de sacrificio condicional (avance simplificado)
  - Matchup L/R, clima, defensa (fases futuras)

Objetivo: validar que la mecánica de simulación (estados base-out,
Monte Carlo, agregación de carreras) funciona correctamente, comparando
la distribución de carreras resultante contra GT Classic (Binomial
Negativa) sobre el mismo set de backtesting 2022-2023.

Nunca reemplaza GT Classic sin validación fuera de muestra rigurosa
(Brier Score, Log Loss, calibración) con el mismo protocolo de las
hipótesis descartadas en GT Classic (train/val/test 60/20/20, bootstrap
2,000 iteraciones).
"""

import random
from dataclasses import dataclass
from collections import Counter

# ---------------------------------------------------------------------
# 1. Probabilidades de PA - promedio de liga MLB (fijo para Fase 1)
# ---------------------------------------------------------------------
PA_PROBS = {
    "K": 0.224,
    "BB": 0.086,
    "HBP": 0.010,
    "1B": 0.145,
    "2B": 0.043,
    "3B": 0.004,
    "HR": 0.030,
    "OUT_EN_JUEGO": 0.458,
}
assert abs(sum(PA_PROBS.values()) - 1.0) < 1e-9, "Las probabilidades de PA deben sumar 1.0"

EVENTOS = list(PA_PROBS.keys())
PESOS = list(PA_PROBS.values())


def simular_pa() -> str:
    """Devuelve un evento de PA según las probabilidades de liga."""
    return random.choices(EVENTOS, weights=PESOS, k=1)[0]


# ---------------------------------------------------------------------
# 2. Estado del juego: bases (tupla de 3 bool: 1B, 2B, 3B) + outs
# ---------------------------------------------------------------------
@dataclass
class EstadoEntrada:
    primera: bool = False
    segunda: bool = False
    tercera: bool = False
    outs: int = 0

    def bases_vacias(self) -> bool:
        return not (self.primera or self.segunda or self.tercera)


def avanzar_corredores(estado: EstadoEntrada, evento: str) -> tuple[EstadoEntrada, int]:
    """
    Aplica el resultado del PA al estado de la entrada.
    Devuelve (nuevo_estado, carreras_anotadas).
    """
    p, s, t, outs = estado.primera, estado.segunda, estado.tercera, estado.outs
    carreras = 0

    if evento == "K":
        outs += 1

    elif evento == "OUT_EN_JUEGO":
        outs += 1

    elif evento in ("BB", "HBP"):
        if p and s and t:
            carreras += 1
            p, s, t = True, True, True
        elif p and s and not t:
            t = True
            p, s = True, True
        elif p and not s:
            s = True
            p = True
        else:
            p = True

    elif evento == "1B":
        if t:
            carreras += 1
        if s:
            carreras += 1
            s = False
        if p:
            s = True
        t = False
        p = True

    elif evento == "2B":
        if t:
            carreras += 1
        if s:
            carreras += 1
        if p:
            t = True
        s, p = True, False

    elif evento == "3B":
        carreras += sum([p, s, t])
        p, s, t = False, False, True

    elif evento == "HR":
        carreras += 1 + sum([p, s, t])
        p, s, t = False, False, False

    return EstadoEntrada(p, s, t, outs), carreras


# ---------------------------------------------------------------------
# 3. Simulación de una entrada completa
# ---------------------------------------------------------------------
def simular_entrada() -> int:
    estado = EstadoEntrada()
    carreras_entrada = 0
    while estado.outs < 3:
        evento = simular_pa()
        estado, carreras = avanzar_corredores(estado, evento)
        carreras_entrada += carreras
    return carreras_entrada


# ---------------------------------------------------------------------
# 4. Simulación de un juego completo (9 entradas, sin bullpen)
# ---------------------------------------------------------------------
def simular_juego(entradas: int = 9) -> int:
    return sum(simular_entrada() for _ in range(entradas))


# ---------------------------------------------------------------------
# 5. Monte Carlo
# ---------------------------------------------------------------------
def monte_carlo(n_juegos: int = 10_000, semilla: int | None = None) -> list[int]:
    if semilla is not None:
        random.seed(semilla)
    return [simular_juego() for _ in range(n_juegos)]


def resumen_distribucion(resultados: list[int]) -> dict:
    n = len(resultados)
    media = sum(resultados) / n
    varianza = sum((x - media) ** 2 for x in resultados) / n
    conteo = Counter(resultados)
    return {
        "n_juegos": n,
        "media_carreras": round(media, 3),
        "varianza": round(varianza, 3),
        "desv_estandar": round(varianza ** 0.5, 3),
        "min": min(resultados),
        "max": max(resultados),
        "distribucion_pct": {
            k: round(100 * v / n, 2) for k, v in sorted(conteo.items())
        },
    }


# ---------------------------------------------------------------------
# 6. Prueba de humo (smoke test)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    resultados = monte_carlo(n_juegos=10_000, semilla=42)
    resumen = resumen_distribucion(resultados)

    print("=== GT Next Fase 1 — Smoke Test ===")
    print(f"Juegos simulados: {resumen['n_juegos']}")
    print(f"Media de carreras por equipo por juego: {resumen['media_carreras']}")
    print(f"Desviación estándar: {resumen['desv_estandar']}")
    print(f"Rango: {resumen['min']} - {resumen['max']}")
