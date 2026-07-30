"""
Hipotesis 1: Valor del K-BB% condicionado por la magnitud de f_era (Prioridad Alta)
Pregunta: El K-BB% aporta informacion independiente del ERA, o simplemente
duplica lo que el ERA ya capta? Ese aporte, es mayor cuando el ERA es
ambiguo (neutral) que cuando es extremo?

Tabla: backtesting_pesos (columnas: game_id, fecha, f_era_local, f_kbb_local,
f_era_visitante, f_kbb_visitante, gano_local, runs_reales_local, runs_reales_visitante)

NOTA IMPORTANTE DE DISEÑO:
Esta tabla no tiene una probabilidad ya calculada (a diferencia de
backtesting_resultados). Para medir el "aporte" de K-BB%, se entrenan dos
regresiones logisticas por cada bucket de magnitud de ERA, usando SOLO datos
de TRAIN de ese bucket:
  - Modelo A (solo ERA): usa diff_era = f_era_local - f_era_visitante
  - Modelo B (ERA + KBB): usa diff_era y diff_kbb = f_kbb_local - f_kbb_visitante
Los coeficientes se congelan en TRAIN y se aplican sin refit a VAL y TEST del
mismo bucket. Se compara Brier/LogLoss/Accuracy entre A y B, con bootstrap
para saber si la diferencia es estadisticamente significativa o es ruido.

ROI: no disponible en esta tabla (no hay cuotas de mercado guardadas). Se omite.

Uso: python test_splits.py
"""

import os
from supabase import create_client
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

N_BOOTSTRAP = 2000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def cargar_datos():
    """Carga backtesting_pesos completo via paginacion (Supabase limita a 1000
    filas por default)."""
    PAGE_SIZE = 1000
    todas_las_filas = []
    inicio = 0
    while True:
        fin = inicio + PAGE_SIZE - 1
        resultado = supabase.table("backtesting_pesos").select(
            "game_id, fecha, f_era_local, f_kbb_local, f_era_visitante, "
            "f_kbb_visitante, gano_local, runs_reales_local, runs_reales_visitante"
        ).range(inicio, fin).execute()
        filas = resultado.data
        todas_las_filas.extend(filas)
        print(f"  Pagina traida: filas {inicio}-{fin} -> {len(filas)} registros")
        if len(filas) < PAGE_SIZE:
            break
        inicio += PAGE_SIZE

    df = pd.DataFrame(todas_las_filas)
    print(f"Total de filas traidas de Supabase (antes de dropna): {len(df)}")

    cols_clave = ["f_era_local", "f_era_visitante", "f_kbb_local",
                  "f_kbb_visitante", "gano_local"]
    df = df.dropna(subset=cols_clave)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    df["diff_era"] = df["f_era_local"] - df["f_era_visitante"]
    df["diff_kbb"] = df["f_kbb_local"] - df["f_kbb_visitante"]
    df["gano_local_int"] = df["gano_local"].astype(int)

    if len(df) < 3000:
        print(f"  ADVERTENCIA: se esperaban ~3642 juegos, se obtuvieron {len(df)}. "
              f"Revisar paginacion o NULLs.")
    return df


def definir_buckets_era(df_train):
    """Calcula percentiles 20/50/80 de |diff_era| usando SOLO train."""
    abs_diff = df_train["diff_era"].abs()
    p20, p50, p80 = np.percentile(abs_diff, [20, 50, 80])
    print(f"Percentiles de |diff_era| (definidos en TRAIN): "
          f"p20={p20:.4f} | p50={p50:.4f} | p80={p80:.4f}")
    return p20, p50, p80


def asignar_bucket(df, p20, p50, p80):
    abs_diff = df["diff_era"].abs()
    condiciones = [
        abs_diff <= p20,
        (abs_diff > p20) & (abs_diff <= p50),
        (abs_diff > p50) & (abs_diff <= p80),
        abs_diff > p80,
    ]
    etiquetas = ["neutral", "moderado", "fuerte", "extremo"]
    df = df.copy()
    df["bucket_era"] = np.select(condiciones, etiquetas, default="extremo")
    return df


def entrenar_modelos(df_train_bucket):
    """Entrena Modelo A (solo ERA) y Modelo B (ERA+KBB) sobre datos de train
    de un bucket especifico. Retorna ambos modelos ya ajustados."""
    y = df_train_bucket["gano_local_int"].values

    X_a = df_train_bucket[["diff_era"]].values
    modelo_a = LogisticRegression()
    modelo_a.fit(X_a, y)

    X_b = df_train_bucket[["diff_era", "diff_kbb"]].values
    modelo_b = LogisticRegression()
    modelo_b.fit(X_b, y)

    return modelo_a, modelo_b


def evaluar_modelo(modelo, X, y_true):
    probs = modelo.predict_proba(X)[:, 1]
    probs_clip = np.clip(probs, 0.001, 0.999)
    preds = (probs_clip >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "brier": brier_score_loss(y_true, probs_clip),
        "log_loss": log_loss(y_true, probs_clip, labels=[0, 1]),
    }, probs_clip


def bootstrap_diferencia_brier(y_true, probs_a, probs_b, n_iter=N_BOOTSTRAP):
    """
    Bootstrap sobre las filas de evaluacion (sin refit de modelos) para estimar
    un intervalo de confianza al 95% de la diferencia:
      mejora = brier_modelo_A - brier_modelo_B
    mejora > 0 significa que el modelo B (con KBB) tiene MENOR Brier (mejor).
    """
    n = len(y_true)
    if n < 10:
        return None, None, None

    diffs = np.empty(n_iter)
    y_arr = np.asarray(y_true)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        brier_a = brier_score_loss(y_arr[idx], probs_a[idx])
        brier_b = brier_score_loss(y_arr[idx], probs_b[idx])
        diffs[i] = brier_a - brier_b

    mejora_media = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    significativo = (ci_low > 0) or (ci_high < 0)
    return mejora_media, (ci_low, ci_high), significativo


def main():
    print("=== Hipotesis 1: K-BB% condicionado por magnitud de ERA ===\n")
    df = cargar_datos()
    n_total = len(df)
    print(f"\nTotal de juegos con datos completos: {n_total}\n")

    corte_train = int(n_total * 0.6)
    corte_val = int(n_total * 0.8)
    df_train = df.iloc[:corte_train].copy()
    df_val = df.iloc[corte_train:corte_val].copy()
    df_test = df.iloc[corte_val:].copy()
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}\n")

    p20, p50, p80 = definir_buckets_era(df_train)
    print()

    df_train = asignar_bucket(df_train, p20, p50, p80)
    df_val = asignar_bucket(df_val, p20, p50, p80)
    df_test = asignar_bucket(df_test, p20, p50, p80)

    resumen_filas = []
    conclusiones_por_bucket = {}

    for bucket in ["neutral", "moderado", "fuerte", "extremo"]:
        print(f"\n{'='*60}")
        print(f"BUCKET: {bucket.upper()}")
        print(f"{'='*60}")

        train_bucket = df_train[df_train["bucket_era"] == bucket]
        if len(train_bucket) < 30:
            print(f"  Muestra de train insuficiente (n={len(train_bucket)}), se omite bucket.")
            continue

        modelo_a, modelo_b = entrenar_modelos(train_bucket)

        resultados_bucket = {}
        for nombre_set, subset in [("TRAIN", train_bucket),
                                     ("VALIDATION", df_val[df_val["bucket_era"] == bucket]),
                                     ("TEST", df_test[df_test["bucket_era"] == bucket])]:
            if len(subset) < 10:
                print(f"  {nombre_set}: n={len(subset)} (insuficiente, se omite)")
                continue

            y_true = subset["gano_local_int"].values
            X_a = subset[["diff_era"]].values
            X_b = subset[["diff_era", "diff_kbb"]].values

            metricas_a, probs_a = evaluar_modelo(modelo_a, X_a, y_true)
            metricas_b, probs_b = evaluar_modelo(modelo_b, X_b, y_true)

            print(f"\n  --- {nombre_set} (n={len(subset)}) ---")
            print(f"    Modelo A (solo ERA):    Accuracy={metricas_a['accuracy']:.4f} | "
                  f"Brier={metricas_a['brier']:.5f} | LogLoss={metricas_a['log_loss']:.4f}")
            print(f"    Modelo B (ERA + KBB):   Accuracy={metricas_b['accuracy']:.4f} | "
                  f"Brier={metricas_b['brier']:.5f} | LogLoss={metricas_b['log_loss']:.4f}")

            mejora, ci, sig = bootstrap_diferencia_brier(y_true, probs_a, probs_b)
            if mejora is not None:
                print(f"    Bootstrap (Brier A - Brier B), n_iter={N_BOOTSTRAP}: "
                      f"mejora media={mejora:+.5f} | IC 95%=({ci[0]:+.5f}, {ci[1]:+.5f}) | "
                      f"significativo={'SI' if sig else 'no'}")

            resultados_bucket[nombre_set] = {
                "n": len(subset), "brier_a": metricas_a["brier"], "brier_b": metricas_b["brier"],
                "acc_a": metricas_a["accuracy"], "acc_b": metricas_b["accuracy"],
                "mejora_brier": mejora, "ci_low": ci[0] if ci else None,
                "ci_high": ci[1] if ci else None, "significativo": sig
            }
            resumen_filas.append({
                "bucket": bucket, "set": nombre_set, "n": len(subset),
                "brier_solo_era": metricas_a["brier"], "brier_era_kbb": metricas_b["brier"],
                "acc_solo_era": metricas_a["accuracy"], "acc_era_kbb": metricas_b["accuracy"],
                "mejora_brier_kbb": mejora,
                "ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None,
                "significativo": sig
            })

        conclusiones_por_bucket[bucket] = resultados_bucket

    df_resumen = pd.DataFrame(resumen_filas)
    print(f"\n\n{'='*60}")
    print("=== RESUMEN COMPLETO ===")
    print(f"{'='*60}")
    print(df_resumen.to_string(index=False))
    df_resumen.to_csv("/tmp/resultados_hipotesis1_kbb_era.csv", index=False)
    print("\nResultados guardados en /tmp/resultados_hipotesis1_kbb_era.csv")

    # --- Conclusion automatica ---
    print(f"\n\n{'='*60}")
    print("=== CONCLUSION AUTOMATICA ===")
    print(f"{'='*60}")

    sig_val_test_por_bucket = {}
    for bucket, resultados in conclusiones_por_bucket.items():
        val_sig = resultados.get("VALIDATION", {}).get("significativo", False)
        test_sig = resultados.get("TEST", {}).get("significativo", False)
        val_mejora = resultados.get("VALIDATION", {}).get("mejora_brier", 0) or 0
        test_mejora = resultados.get("TEST", {}).get("mejora_brier", 0) or 0
        # Consistente = significativo en AMBOS val y test, Y la mejora favorece a KBB (positiva) en ambos
        consistente = val_sig and test_sig and val_mejora > 0 and test_mejora > 0
        sig_val_test_por_bucket[bucket] = consistente
        print(f"  Bucket '{bucket}': significativo y consistente (KBB mejora) en VAL+TEST = {consistente}")

    hay_evidencia_dinamica = any(sig_val_test_por_bucket.values())
    # Patron esperado especifico: mayor beneficio en neutral, menor/nulo en extremo
    neutral_gana = sig_val_test_por_bucket.get("neutral", False)
    extremo_gana = sig_val_test_por_bucket.get("extremo", False)

    print()
    if neutral_gana and not extremo_gana:
        print("RECOMENDACION: B. IMPLEMENTAR PESOS DINAMICOS")
        print("Evidencia consistente (significativa en VALIDATION y TEST) de que K-BB%")
        print("aporta valor real cuando el ERA es neutral, y ese aporte no se sostiene")
        print("de la misma forma cuando el ERA es extremo. Esto respalda evolucionar")
        print("de pesos fijos (0.70/0.30) hacia pesos que dependan de la magnitud del ERA.")
    elif hay_evidencia_dinamica:
        print("RECOMENDACION: revisar con mas detalle antes de decidir")
        print("Hay evidencia significativa de aporte de K-BB% en algun bucket, pero no")
        print("sigue el patron esperado (mayor en neutral, menor en extremo). Antes de")
        print("implementar pesos dinamicos, vale la pena entender por que el patron no")
        print("es el esperado - podria ser una senal real distinta a la hipotesis original,")
        print("o podria ser ruido en un bucket especifico.")
    else:
        print("RECOMENDACION: A. MANTENER PESOS FIJOS")
        print("No existe evidencia estadisticamente significativa y consistente (en")
        print("VALIDATION y TEST simultaneamente) de que el aporte de K-BB% cambie con")
        print("la magnitud del ERA. Los pesos fijos actuales (0.70 ERA / 0.30 KBB) no")
        print("deberian modificarse con la evidencia disponible.")


if __name__ == "__main__":
    main()
