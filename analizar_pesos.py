import os
import numpy as np
from supabase import create_client
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

PESO_ERA_ACTUAL = 0.70
PESO_KBB_ACTUAL = 0.30

def descargar_todo():
    todos = []
    offset = 0
    while True:
        res = supabase.table("backtesting_pesos").select("*").range(offset, offset + 999).execute()
        if not res.data:
            break
        todos.extend(res.data)
        offset += 1000
    return todos

def analizar():
    print("Descargando backtesting_pesos...")
    todos = descargar_todo()
    print(f"Total filas: {len(todos)}")

    datos_2022 = [f for f in todos if f["fecha"].startswith("2022")]
    datos_2023 = [f for f in todos if f["fecha"].startswith("2023")]
    print(f"2022 (train): {len(datos_2022)} | 2023 (test): {len(datos_2023)}")

    if len(datos_2022) < 50 or len(datos_2023) < 50:
        print("Advertencia: muestra insuficiente en alguno de los dos años para un ajuste confiable.")

    def features_y_target(filas):
        X, y = [], []
        for f in filas:
            fe_l, fk_l = f["f_era_local"], f["f_kbb_local"]
            fe_v, fk_v = f["f_era_visitante"], f["f_kbb_visitante"]
            if None in (fe_l, fk_l, fe_v, fk_v):
                continue
            X.append([fe_l, fk_l, fe_v, fk_v])
            y.append(1 if f["gano_local"] else 0)
        return np.array(X), np.array(y)

    X_train, y_train = features_y_target(datos_2022)
    X_test, y_test = features_y_target(datos_2023)
    print(f"Filas usables — train: {len(X_train)} | test: {len(X_test)}")

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    print(f"\nPesos aprendidos (2022): {modelo.coef_[0]}")
    print(f"  [f_era_local, f_kbb_local, f_era_visitante, f_kbb_visitante]")

    prob_optimizada = modelo.predict_proba(X_test)[:, 1]
    brier_optimizado = brier_score_loss(y_test, prob_optimizada)

    prob_fija = []
    for f in datos_2023:
        fe_l, fk_l = f.get("f_era_local"), f.get("f_kbb_local")
        fe_v, fk_v = f.get("f_era_visitante"), f.get("f_kbb_visitante")
        if None in (fe_l, fk_l, fe_v, fk_v):
            continue
        f_abridor_l = fe_l * PESO_ERA_ACTUAL + fk_l * PESO_KBB_ACTUAL
        f_abridor_v = fe_v * PESO_ERA_ACTUAL + fk_v * PESO_KBB_ACTUAL
        prob_fija.append(f_abridor_v / (f_abridor_l + f_abridor_v))
    prob_fija = np.array(prob_fija)
    brier_fijo = brier_score_loss(y_test, prob_fija)

    print(f"\n=== RESULTADOS (evaluado en 2023, nunca visto en el ajuste) ===")
    print(f"Brier Score — pesos actuales (0.70/0.30 fijo): {brier_fijo:.5f}")
    print(f"Brier Score — pesos optimizados (regresión sobre 2022): {brier_optimizado:.5f}")

if __name__ == "__main__":
    analizar()
