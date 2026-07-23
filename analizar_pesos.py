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

def analizar():
    print("Descargando backtesting_pesos...")
    todos = descargar_todo()
    print(f"Total filas: {len(todos)}")

    datos_2022 = [f for f in todos if f["fecha"].startswith("2022")]
    datos_2023 = [f for f in todos if f["fecha"].startswith("2023")]
    print(f"2022 (train): {len(datos_2022)} | 2023 (test): {len(datos_2023)}")

    X_train, y_train = features_y_target(datos_2022)
    X_test, y_test = features_y_target(datos_2023)
    print(f"Filas usables — train: {len(X_train)} | test: {len(X_test)}")

    resultados = {}

    modelo_sin_reg = LogisticRegression(penalty=None)
    modelo_sin_reg.fit(X_train, y_train)
    prob_sin_reg = modelo_sin_reg.predict_proba(X_test)[:, 1]
    resultados["Sin regularizar (original)"] = brier_score_loss(y_test, prob_sin_reg)
    print(f"Coeficientes sin regularizar: {modelo_sin_reg.coef_[0]}")

    mejores_ridge = {"C": None, "brier": 999}
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        modelo_ridge = LogisticRegression(penalty='l2', C=C)
        modelo_ridge.fit(X_train, y_train)
        prob_ridge = modelo_ridge.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, prob_ridge)
        print(f"Ridge (C={C}): Brier={brier:.5f} | coef={modelo_ridge.coef_[0]}")
        if brier < mejores_ridge["brier"]:
            mejores_ridge = {"C": C, "brier": brier}
    resultados[f"Ridge (mejor C={mejores_ridge['C']})"] = mejores_ridge["brier"]

    mejores_en = {"C": None, "l1_ratio": None, "brier": 999}
    for C in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.3, 0.5, 0.7]:
            modelo_en = LogisticRegression(penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio, max_iter=2000)
            modelo_en.fit(X_train, y_train)
            prob_en = modelo_en.predict_proba(X_test)[:, 1]
            brier = brier_score_loss(y_test, prob_en)
            if brier < mejores_en["brier"]:
                mejores_en = {"C": C, "l1_ratio": l1_ratio, "brier": brier}
    resultados[f"Elastic Net (mejor C={mejores_en['C']}, l1_ratio={mejores_en['l1_ratio']})"] = mejores_en["brier"]

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
    resultados["Pesos actuales (fijo)"] = brier_score_loss(y_test, prob_fija)

    print(f"\n=== RESULTADOS FINALES (evaluado en 2023) ===")
    for nombre, brier in sorted(resultados.items(), key=lambda x: x[1]):
        print(f"{nombre}: {brier:.5f}")

if __name__ == "__main__":
    analizar()
