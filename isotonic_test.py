import os
import numpy as np
from supabase import create_client
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from motor import calibrar_platt

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LIGA_ERA_PROMEDIO = 4.20
PRIOR_IP_ABRIDOR = 45

def runs_desde_era(era_rival):
    f_era = era_rival / LIGA_ERA_PROMEDIO
    return 4.35 * f_era

def correr_comparacion():
    print("Descargando datos de backtesting_resultados...")
    todos = []
    offset = 0
    while True:
        res = supabase.table("backtesting_resultados").select(
            "era_shrink_local, era_shrink_visitante, gano_local"
        ).range(offset, offset + 999).execute()
        if not res.data:
            break
        todos.extend(res.data)
        offset += 1000

    print(f"Total filas: {len(todos)}")

    prob_cruda = []
    gano = []
    for fila in todos:
        era_l = fila["era_shrink_local"]
        era_v = fila["era_shrink_visitante"]
        if era_l is None or era_v is None:
            continue
        runs_l = runs_desde_era(era_v)
        runs_v = runs_desde_era(era_l)
        p_cruda = runs_l / (runs_l + runs_v)
        prob_cruda.append(p_cruda)
        gano.append(1 if fila["gano_local"] else 0)

    prob_cruda = np.array(prob_cruda)
    gano = np.array(gano)
    print(f"Filas usables: {len(prob_cruda)}")

    prob_platt = np.array([calibrar_platt(p) for p in prob_cruda])
    brier_platt = brier_score_loss(gano, prob_platt)
    brier_cruda = brier_score_loss(gano, prob_cruda)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    prob_isotonic_oof = np.zeros_like(prob_cruda)
    for train_idx, test_idx in kf.split(prob_cruda):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob_cruda[train_idx], gano[train_idx])
        prob_isotonic_oof[test_idx] = iso.predict(prob_cruda[test_idx])
    brier_isotonic = brier_score_loss(gano, prob_isotonic_oof)

    print(f"\n=== RESULTADOS ===")
    print(f"Brier Score (cruda, sin calibrar): {brier_cruda:.5f}")
    print(f"Brier Score (Platt Scaling actual): {brier_platt:.5f}")
    print(f"Brier Score (Isotonic Regression, out-of-fold): {brier_isotonic:.5f}")

if __name__ == "__main__":
    correr_comparacion()
