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

    datos_2022 = sorted([f for f in todos if f["fecha"].startswith("2022")], key=lambda x: x["fecha"])
    datos_2023 = [f for f in todos if f["fecha"].startswith("2023")]

    mitad = len(datos_2022) // 2
    datos_train = datos_2022[:mitad]
    datos_val = datos_2022[mitad:]
    print(f"Train (2022 primera mitad): {len(datos_train)} | Validacion (2022 segunda mitad): {len(datos_val)} | Test (2023, intacto): {len(datos_2023)}")

    X_train, y_train = features_y_target(datos_train)
    X_val, y_val = features_y_target(datos_val)
    X_test, y_test = features_y_target(datos_2023)
    print(f"Filas usables — train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    print("\n=== FASE 1: Seleccionar hiperparametros usando SOLO validacion (2022 segunda mitad) ===")
    mejor_config = {"C": None, "l1_ratio": None, "brier_val": 999}
    for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            modelo = LogisticRegression(penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio, max_iter=2000)
            modelo.fit(X_train, y_train)
            prob_val = modelo.predict_proba(X_val)[:, 1]
            brier_val = brier_score_loss(y_val, prob_val)
            if brier_val < mejor_config["brier_val"]:
                mejor_config = {"C": C, "l1_ratio": l1_ratio, "brier_val": brier_val}
    print(f"Mejor configuracion en validacion: C={mejor_config['C']}, l1_ratio={mejor_config['l1_ratio']} (Brier val={mejor_config['brier_val']:.5f})")

    print("\n=== FASE 2: Entrenar con train+validacion (todo 2022), hiperparametros congelados ===")
    X_train_completo, y_train_completo = features_y_target(datos_2022)
    modelo_final = LogisticRegression(penalty='elasticnet', solver='saga',
                                        C=mejor_config["C"], l1_ratio=mejor_config["l1_ratio"], max_iter=2000)
    modelo_final.fit(X_train_completo, y_train_completo)

    print("\n=== FASE 3: Evaluacion UNICA sobre 2023 (nunca visto hasta ahora) ===")
    prob_test_elastic = modelo_final.predict_proba(X_test)[:, 1]
    brier_elastic = brier_score_loss(y_test, prob_test_elastic)

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

    print(f"\n=== RESULTADO FINAL (validacion metodologicamente correcta) ===")
    print(f"Pesos actuales (fijo): {brier_fijo:.5f}")
    print(f"Elastic Net (C={mejor_config['C']}, l1_ratio={mejor_config['l1_ratio']}, hiperparametros elegidos sin ver test): {brier_elastic:.5f}")

if __name__ == "__main__":
    analizar()
