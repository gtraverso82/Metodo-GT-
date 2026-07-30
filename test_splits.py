"""
Hipotesis 2: Sesgo local vs. visitante (Prioridad Media-Alta) - ULTIMA
hipotesis evaluable con el historico actual.

Pregunta: Es el modelo (GT Classic, usando ERA-shrink que gano el experimento #1)
mas preciso cuando favorece al equipo local que cuando favorece al visitante?

Tabla: backtesting_resultados (columnas: game_id, fecha, prob_local_era,
prob_local_xfip, gano_local, runs_reales_local, runs_reales_visitante)

NOTA DE DISEÑO: se usa prob_local_era (no prob_local_xfip) porque ERA-shrink
es la metrica que gano el experimento #1 y es la que corre en produccion hoy
en motor.py. No se entrena ningun modelo nuevo aqui - se evalua el modelo
existente. La separacion TRAIN/VAL/TEST se mantiene para verificar que
cualquier patron encontrado se sostenga en el tiempo (no sea un artefacto de
un periodo especifico), no para ajustar nada.

ROI: no disponible (no hay cuotas de mercado en esta tabla). Se omite.

Uso: python test_splits.py
"""

import os
from supabase import create_client
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

N_BOOTSTRAP = 2000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def cargar_datos():
    """Carga backtesting_resultados completo via paginacion."""
    PAGE_SIZE = 1000
    todas_las_filas = []
    inicio = 0
    while True:
        fin = inicio + PAGE_SIZE - 1
        resultado = supabase.table("backtesting_resultados").select(
            "game_id, fecha, prob_local_era, prob_local_xfip, gano_local, "
            "runs_reales_local, runs_reales_visitante"
        ).range(inicio, fin).execute()
        filas = resultado.data
        todas_las_filas.extend(filas)
        print(f"  Pagina traida: filas {inicio}-{fin} -> {len(filas)} registros")
        if len(filas) < PAGE_SIZE:
            break
        inicio += PAGE_SIZE

    df = pd.DataFrame(todas_las_filas)
    print(f"Total de filas traidas de Supabase (antes de dropna): {len(df)}")

    df = df.dropna(subset=["prob_local_era", "gano_local"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["gano_local_int"] = df["gano_local"].astype(int)
    df["favorito"] = np.where(df["prob_local_era"] >= 0.5, "local", "visitante")

    if len(df) < 3000:
        print(f"  ADVERTENCIA: se esperaban ~3642 juegos, se obtuvieron {len(df)}. "
              f"Revisar paginacion o NULLs.")
    return df


def evaluar_grupo(df_grupo, nombre_grupo):
    if len(df_grupo) < 10:
        print(f"    {nombre_grupo}: n={len(df_grupo)} (muestra insuficiente, se omite)")
        return None

    y_true = df_grupo["gano_local_int"].values
    y_prob = df_grupo["prob_local_era"].clip(0.001, 0.999).values
    y_pred = (y_prob >= 0.5).astype(int)

    # "Acierto del favorito": si favorito era local, acierto = gano_local; si
    # favorito era visitante, acierto = no gano_local
    favorito_acerto = (y_pred == y_true)

    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob, labels=[0, 1])
    tasa_acierto_favorito = favorito_acerto.mean()
    tasa_sorpresas = 1 - tasa_acierto_favorito

    print(f"    {nombre_grupo}: n={len(df_grupo)} | Accuracy={acc:.4f} | "
          f"Brier={brier:.5f} | LogLoss={logloss:.4f} | "
          f"Favorito acerto={tasa_acierto_favorito:.1%} | Sorpresas={tasa_sorpresas:.1%}")

    return {"grupo": nombre_grupo, "n": len(df_grupo), "accuracy": acc,
            "brier": brier, "log_loss": logloss,
            "tasa_acierto_favorito": tasa_acierto_favorito,
            "y_true": y_true, "y_prob": y_prob}


def bootstrap_diferencia_brier(y_true_a, y_prob_a, y_true_b, y_prob_b, n_iter=N_BOOTSTRAP):
    """
    Bootstrap independiente sobre cada grupo (local vs visitante) para estimar
    IC 95% de la diferencia: brier_visitante - brier_local
    (positivo = el modelo es MEJOR -menor Brier- prediciendo cuando el
    favorito es local que cuando es visitante)
    """
    n_a, n_b = len(y_true_a), len(y_true_b)
    if n_a < 10 or n_b < 10:
        return None, None, None

    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        brier_a = brier_score_loss(y_true_a[idx_a], y_prob_a[idx_a])  # local
        brier_b = brier_score_loss(y_true_b[idx_b], y_prob_b[idx_b])  # visitante
        diffs[i] = brier_b - brier_a

    mejora_media = diffs.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    significativo = (ci_low > 0) or (ci_high < 0)
    return mejora_media, (ci_low, ci_high), significativo


def main():
    print("=== Hipotesis 2: Sesgo local vs. visitante ===\n")
    df = cargar_datos()
    n_total = len(df)
    print(f"\nTotal de juegos con datos completos: {n_total}\n")

    corte_train = int(n_total * 0.6)
    corte_val = int(n_total * 0.8)
    df_train = df.iloc[:corte_train]
    df_val = df.iloc[corte_train:corte_val]
    df_test = df.iloc[corte_val:]
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}\n")

    n_favorito_local_total = (df["favorito"] == "local").sum()
    n_favorito_visitante_total = (df["favorito"] == "visitante").sum()
    print(f"Distribucion global: favorito local en {n_favorito_local_total} juegos "
          f"({n_favorito_local_total/n_total:.1%}), favorito visitante en "
          f"{n_favorito_visitante_total} juegos ({n_favorito_visitante_total/n_total:.1%})\n")

    resumen_filas = []
    resultados_por_set = {}

    for nombre_set, subset in [("TRAIN", df_train), ("VALIDATION", df_val), ("TEST", df_test)]:
        print(f"--- {nombre_set} ---")
        grupo_local = subset[subset["favorito"] == "local"]
        grupo_visitante = subset[subset["favorito"] == "visitante"]

        r_local = evaluar_grupo(grupo_local, f"{nombre_set} - Favorito LOCAL")
        r_visitante = evaluar_grupo(grupo_visitante, f"{nombre_set} - Favorito VISITANTE")

        if r_local and r_visitante:
            mejora, ci, sig = bootstrap_diferencia_brier(
                r_local["y_true"], r_local["y_prob"],
                r_visitante["y_true"], r_visitante["y_prob"]
            )
            print(f"    Bootstrap (Brier visitante - Brier local), n_iter={N_BOOTSTRAP}: "
                  f"diferencia media={mejora:+.5f} | IC 95%=({ci[0]:+.5f}, {ci[1]:+.5f}) | "
                  f"significativo={'SI' if sig else 'no'}")
            print("    (positivo = el modelo predice MEJOR cuando el favorito es local)\n")

            resultados_por_set[nombre_set] = {"mejora": mejora, "ci": ci, "significativo": sig}

            for r, etiqueta in [(r_local, "local"), (r_visitante, "visitante")]:
                resumen_filas.append({
                    "set": nombre_set, "favorito": etiqueta, "n": r["n"],
                    "accuracy": r["accuracy"], "brier": r["brier"], "log_loss": r["log_loss"],
                    "tasa_acierto_favorito": r["tasa_acierto_favorito"],
                    "diferencia_brier_visitante_menos_local": mejora,
                    "ci95_low": ci[0], "ci95_high": ci[1], "significativo": sig
                })
        print()

    df_resumen = pd.DataFrame(resumen_filas)
    print("=== RESUMEN COMPLETO ===")
    print(df_resumen.to_string(index=False))
    df_resumen.to_csv("/tmp/resultados_hipotesis2_local_visitante.csv", index=False)
    print("\nResultados guardados en /tmp/resultados_hipotesis2_local_visitante.csv")

    # --- Conclusion automatica ---
    print(f"\n{'='*60}")
    print("=== CONCLUSION AUTOMATICA ===")
    print(f"{'='*60}")

    val_sig = resultados_por_set.get("VALIDATION", {}).get("significativo", False)
    test_sig = resultados_por_set.get("TEST", {}).get("significativo", False)
    val_mejora = resultados_por_set.get("VALIDATION", {}).get("mejora", 0) or 0
    test_mejora = resultados_por_set.get("TEST", {}).get("mejora", 0) or 0

    consistente = val_sig and test_sig and np.sign(val_mejora) == np.sign(test_mejora)

    print(f"Significativo en VALIDATION: {val_sig} (diferencia={val_mejora:+.5f})")
    print(f"Significativo en TEST: {test_sig} (diferencia={test_mejora:+.5f})")
    print(f"Consistente (mismo signo, ambos significativos): {consistente}\n")

    if consistente and val_mejora > 0:
        print("RECOMENDACION: SI existe sesgo real - el modelo predice sistematicamente")
        print("mejor cuando el favorito es LOCAL que cuando es visitante. Vale la pena")
        print("investigar si el ajuste de ventaja de localia (home field) en motor.py")
        print("necesita recalibracion, o si hay una variable de contexto de local/visitante")
        print("no capturada actualmente.")
    elif consistente and val_mejora < 0:
        print("RECOMENDACION: SI existe sesgo real - el modelo predice sistematicamente")
        print("mejor cuando el favorito es VISITANTE que cuando es local. Patron")
        print("inesperado (contrario a la intuicion de ventaja de localia) - revisar")
        print("con atencion antes de actuar, podria indicar sobre-ajuste del home field")
        print("actual en motor.py.")
    else:
        print("RECOMENDACION: NO hay evidencia consistente de sesgo local/visitante.")
        print("El patron no es significativo en ambos sets simultaneamente, o cambia")
        print("de signo entre VALIDATION y TEST - compatible con ruido estadistico.")
        print("No se justifica modificar el ajuste de ventaja de localia actual.")

    print(f"\n{'='*60}")
    print("=== CONCLUSION GENERAL DE LAS 3 HIPOTESIS (si esta es la ultima) ===")
    print(f"{'='*60}")
    print("Si esta hipotesis tampoco muestra evidencia consistente, las 3 lineas de")
    print("investigacion sobre recombinar variables existentes (K-BB%/ERA, ERA/xFIP,")
    print("local/visitante) habran sido evaluadas con rigor y descartadas. La prioridad")
    print("estrategica deberia pasar de optimizar la combinacion de variables actuales")
    print("a incorporar nuevas fuentes de informacion (bullpen, park factor historico,")
    print("clima real, matchups bateador-lanzador, etc.) segun el plan ya discutido.")


if __name__ == "__main__":
    main()
