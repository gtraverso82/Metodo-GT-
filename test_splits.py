"""
Hipotesis 3: Consenso ERA vs xFIP (Prioridad Muy Alta)
Pregunta: Es el modelo mas preciso cuando ERA y xFIP favorecen al mismo equipo,
comparado con cuando discrepan?

Requiere: tabla backtesting_resultados (columnas: game_id, prob_local_era,
prob_local_xfip, gano_local, runs_reales_local, runs_reales_visitante)

Metodologia: split train/val/test TEMPORAL (por fecha), igual que en el
experimento #4 ya validado, para evitar fuga de informacion. Las categorias
de consenso se definen y los umbrales se fijan usando SOLO el set de train;
val y test solo evaluan.

Uso: python analizar_consenso_era_xfip.py
"""

import os
from supabase import create_client
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def cargar_datos():
    """Carga backtesting_resultados completo, ordenado por fecha para split temporal."""
    resultado = supabase.table("backtesting_resultados").select(
        "game_id, fecha, prob_local_era, prob_local_xfip, gano_local, "
        "runs_reales_local, runs_reales_visitante"
    ).execute()
    df = pd.DataFrame(resultado.data)
    df = df.dropna(subset=["prob_local_era", "prob_local_xfip", "gano_local"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


def clasificar_consenso(df, umbral_similar):
    """
    Clasifica cada juego segun el nivel de acuerdo entre ERA y xFIP.
    - Consenso fuerte: ambos favorecen al mismo equipo Y difieren poco en probabilidad
    - Consenso moderado: ambos favorecen al mismo equipo pero difieren bastante
    - Desacuerdo: favorecen equipos distintos (uno > 0.5, el otro < 0.5)
    """
    favorito_era = (df["prob_local_era"] >= 0.5)
    favorito_xfip = (df["prob_local_xfip"] >= 0.5)
    mismo_favorito = favorito_era == favorito_xfip
    diferencia_abs = (df["prob_local_era"] - df["prob_local_xfip"]).abs()

    categoria = np.where(
        ~mismo_favorito, "desacuerdo",
        np.where(diferencia_abs <= umbral_similar, "consenso_fuerte", "consenso_moderado")
    )
    df = df.copy()
    df["categoria_consenso"] = categoria
    df["diferencia_abs_era_xfip"] = diferencia_abs
    # Probabilidad "de consenso" = promedio simple de ambas, para medir Brier/LogLoss del grupo
    df["prob_promedio"] = (df["prob_local_era"] + df["prob_local_xfip"]) / 2
    return df


def evaluar_grupo(df_grupo, nombre_grupo):
    if len(df_grupo) < 5:
        print(f"  {nombre_grupo}: n={len(df_grupo)} (muestra insuficiente, se omite)")
        return None

    y_true = df_grupo["gano_local"].astype(int)
    y_prob = df_grupo["prob_promedio"].clip(0.001, 0.999)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob, labels=[0, 1])

    print(f"  {nombre_grupo}: n={len(df_grupo)} | Accuracy={acc:.4f} | "
          f"Brier={brier:.5f} | LogLoss={logloss:.4f}")
    return {"grupo": nombre_grupo, "n": len(df_grupo), "accuracy": acc,
            "brier": brier, "log_loss": logloss}


def main():
    df = cargar_datos()
    n_total = len(df)
    print(f"=== Consenso ERA vs xFIP ===")
    print(f"Total de juegos con datos completos: {n_total}\n")

    # Split temporal 60/20/20 (mismo criterio ya validado en experimento #4)
    corte_train = int(n_total * 0.6)
    corte_val = int(n_total * 0.8)
    df_train = df.iloc[:corte_train]
    df_val = df.iloc[corte_train:corte_val]
    df_test = df.iloc[corte_val:]

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}\n")

    # Umbral de "similar" definido SOLO con train: usamos la mediana de
    # diferencia_abs entre juegos con mismo favorito, como punto de corte
    favorito_era_train = df_train["prob_local_era"] >= 0.5
    favorito_xfip_train = df_train["prob_local_xfip"] >= 0.5
    mismo_train = df_train[favorito_era_train == favorito_xfip_train]
    diferencias_train = (mismo_train["prob_local_era"] - mismo_train["prob_local_xfip"]).abs()
    umbral_similar = diferencias_train.median()
    print(f"Umbral 'consenso fuerte' (mediana de diferencias en train, mismo favorito): "
          f"{umbral_similar:.4f}\n")

    resultados_todos = []

    for nombre, subset in [("TRAIN", df_train), ("VALIDATION", df_val), ("TEST", df_test)]:
        print(f"--- {nombre} ---")
        subset_clasificado = clasificar_consenso(subset, umbral_similar)
        for cat in ["consenso_fuerte", "consenso_moderado", "desacuerdo"]:
            grupo = subset_clasificado[subset_clasificado["categoria_consenso"] == cat]
            r = evaluar_grupo(grupo, f"{nombre} - {cat}")
            if r:
                r["set"] = nombre
                resultados_todos.append(r)
        print()

    df_resultados = pd.DataFrame(resultados_todos)
    print("=== RESUMEN ===")
    print(df_resultados.to_string(index=False))

    # Guardar para revisión
    df_resultados.to_csv("/tmp/resultados_consenso_era_xfip.csv", index=False)
    print("\nResultados guardados en /tmp/resultados_consenso_era_xfip.csv")

    print("\n=== INTERPRETACION ===")
    print("Buscar: si 'desacuerdo' tiene Brier/LogLoss consistentemente peor que")
    print("'consenso_fuerte' en VALIDATION y TEST (no solo en TRAIN), hay evidencia")
    print("real de que el consenso es una senal util de confianza.")
    print("Si el patron solo aparece en TRAIN, es sobreajuste - descartar.")


if __name__ == "__main__":
    main()
