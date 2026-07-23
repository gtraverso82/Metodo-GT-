import os
import numpy as np
from supabase import create_client
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from motor import logit, PLATT_A, PLATT_B

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def invertir_platt(prob_calibrada):
    z_calibrado = logit(prob_calibrada)
    z_cruda = (z_calibrado - PLATT_B) / PLATT_A
    return 1 / (1 + np.exp(-z_cruda))

def correr_comparacion():
    print("Descargando datos de backtesting_resultados...")
    todos = []
    offset = 0
    while True:
        res = supabase.table("backtesting_resultados").select(
            "prob_local_era, gano_local"
        ).range(offset, offset + 999).execute()
        if not res.data:
            break
        todos.extend(res.data)
        offset += 1000

    print(f"Total filas: {len(todos)}")

    prob_platt = np.array([f["prob_local_era"] for f in todos if f["prob_local_era"] is not None])
    gano = np.array([f["gano_local"] for f in todos if f["prob_local_era"] is not None]).astype(int)

    print(f"Filas usables: {len(prob_platt)}")

    prob_cruda = invertir_platt(prob_platt)

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
