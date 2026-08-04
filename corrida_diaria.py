import os
from datetime import datetime
from supabase import create_client
from motor import (
    obtener_cartelera_dia, obtener_cuotas_espn, obtener_handicap_espn, obtener_total_espn,
    analizar_partido_hoy, analizar_total, analizar_f5_completo,
    PARK_FACTORS, imprimir_matchup_lr, calcular_matchup_lr, contexto_cualitativo,
    imprimir_winpct, factor_winpct, proyectar_ponches
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def ya_capturado_hoy(game_id, pitcher_id_local, pitcher_id_visitante):
    """
    Verifica si este partido ya tiene una fila en diagnostico_total. Ademas
    compara el abridor probable guardado contra el actual: si cambio (ej.
    escrachado por trade deadline, lesion de ultima hora), borra las filas
    viejas de las 4 tablas relacionadas y fuerza un reprocesamiento completo
    con los datos correctos, en vez de saltar el partido silenciosamente
    (esto es lo que fallo con el caso Peralta->Senga->Phillips en NYM).
    """
    resultado = supabase.table("diagnostico_total").select(
        "id, pitcher_id_local, pitcher_id_visitante"
    ).eq("game_id", game_id).execute()

    if not resultado.data:
        return False

    fila = resultado.data[0]
    stored_local = fila.get("pitcher_id_local")
    stored_visitante = fila.get("pitcher_id_visitante")

    # Filas antiguas (antes de este fix) no tienen estos campos guardados.
    # No hay con que comparar, asumimos que no cambio para no reprocesar
    # en masa el historico existente.
    if stored_local is None or stored_visitante is None:
        return True

    if stored_local != pitcher_id_local or stored_visitante != pitcher_id_visitante:
        print(f"  [ALERTA] Cambio de abridor detectado en {game_id}: "
              f"local {stored_local}->{pitcher_id_local}, "
              f"visitante {stored_visitante}->{pitcher_id_visitante}. Reprocesando...")
        for tabla in ["diagnostico_total", "odds_snapshots", "contexto_partido", "resultado_moneyline"]:
            try:
                supabase.table(tabla).delete().eq("game_id", game_id).execute()
            except Exception as e:
                print(f"    (no se pudo limpiar {tabla}: {e})")
        return False

    return True

def guardar_snapshot(game_id, bookmaker_key, market_key, outcome_name, price,
                      point=None, modelo_prob=None, modelo_bandera=None,
                      tipo_snapshot='apertura'):
    supabase.table("odds_snapshots").insert({
        "game_id": game_id, "bookmaker_key": bookmaker_key, "market_key": market_key,
        "outcome_name": outcome_name, "point": point, "price": price,
        "modelo_prob_en_captura": modelo_prob, "modelo_bandera": modelo_bandera,
        "tipo_snapshot": tipo_snapshot
    }).execute()
    print(f"Snapshot: {game_id} - {market_key} - {outcome_name}")

def guardar_diagnostico_total(game_id, fecha, total_esperado, linea_mercado,
                                park_factor, era_local, era_visitante,
                                pitcher_id_local, pitcher_id_visitante):
    diferencia = total_esperado - linea_mercado
    supabase.table("diagnostico_total").insert({
        "game_id": game_id, "fecha": fecha, "total_esperado_modelo": total_esperado,
        "linea_mercado": linea_mercado, "diferencia": diferencia,
        "park_factor": park_factor, "era_local": era_local, "era_visitante": era_visitante,
        "pitcher_id_local": pitcher_id_local, "pitcher_id_visitante": pitcher_id_visitante
    }).execute()
    print(f"Diagnostico: {game_id} | Linea mercado: {linea_mercado} | Proyectado modelo: {total_esperado:.2f} | Diferencia: {diferencia:+.2f}")

def guardar_auditoria_descarte(game_id, fecha, tipo_descarte, prob_modelo, prob_mercado,
                                  diferencia_moneyline, linea_total, proyeccion_total,
                                  diferencia_total, bandera_moneyline, motivo):
    supabase.table("auditoria_descartes").insert({
        "game_id": game_id, "fecha": fecha, "tipo_descarte": tipo_descarte,
        "prob_modelo": prob_modelo, "prob_mercado": prob_mercado,
        "diferencia_moneyline": diferencia_moneyline,
        "linea_total": linea_total, "proyeccion_total": proyeccion_total,
        "diferencia_total": diferencia_total, "bandera_moneyline": bandera_moneyline,
        "motivo": motivo
    }).execute()
    print(f"  [AUDITORIA] {game_id}: {motivo}")

def guardar_contexto_partido(game_id, fecha, park_factor, ctx, matchup, winpct_ratio,
                               bullpen_era_local, bullpen_era_visitante,
                               bullpen_n_local, bullpen_n_visitante):
    """
    NUEVO (Fase 0 - instrumentacion). Guarda en contexto_partido las variables
    contextuales que hoy NO estan en diagnostico_total ni backtesting_resultados,
    para poder evaluarlas en el futuro con suficiente historico:
    clima real, lesiones, matchup L/R, win% ratio, bullpen ERA/carga reciente,
    park_factor, y ESPN predictor (como referencia externa).
    No afecta el calculo del modelo ni las tablas existentes.
    """
    clima = ctx.get("clima") or {}
    espn_pred = ctx.get("espn_predictor") or {}
    lesiones_local = ctx.get("lesiones_local") or []
    lesiones_visitante = ctx.get("lesiones_visitante") or []

    supabase.table("contexto_partido").insert({
        "game_id": game_id,
        "fecha": fecha,
        "park_factor": park_factor,
        "clima_temp_f": clima.get("temperatura_f"),
        "clima_viento_mph": clima.get("viento_mph"),
        "clima_direccion_viento": clima.get("direccion_viento"),
        "bullpen_era_local": bullpen_era_local,
        "bullpen_era_visitante": bullpen_era_visitante,
        "bullpen_n_relevistas_local": bullpen_n_local,
        "bullpen_n_relevistas_visitante": bullpen_n_visitante,
        "matchup_factor_pitcher_visitante_vs_lineup_local":
            matchup.get("factor_pitcher_visitante_vs_lineup_local"),
        "matchup_factor_pitcher_local_vs_lineup_visitante":
            matchup.get("factor_pitcher_local_vs_lineup_visitante"),
        "winpct_ratio_local_visitante": winpct_ratio,
        "espn_predictor_prob_local": espn_pred.get("prob_local"),
        "espn_predictor_prob_visitante": espn_pred.get("prob_visitante"),
        "lesiones_local": ", ".join(lesiones_local) if lesiones_local else None,
        "lesiones_visitante": ", ".join(lesiones_visitante) if lesiones_visitante else None,
    }).execute()
    print(f"  [CONTEXTO] {game_id}: guardado (bullpen_local={bullpen_era_local}, "
          f"bullpen_visitante={bullpen_era_visitante}, winpct_ratio={winpct_ratio})")

def guardar_resultado_moneyline(game_id, fecha, equipo_local, equipo_visitante,
                                  prob_local, bandera):
    """
    NUEVO. Registra la prediccion de moneyline (favorito + probabilidad + bandera)
    con resultado pendiente (gano_local=NULL). El script resolver_resultados.py
    la actualiza al dia siguiente con el resultado real via MLB Stats API.
    Esto elimina la necesidad de reconstruir el balance de precision a mano
    revisando el chat cada vez.
    """
    favorito = equipo_local if prob_local >= 0.5 else equipo_visitante
    prob_favorito = prob_local if prob_local >= 0.5 else 1 - prob_local
    supabase.table("resultado_moneyline").insert({
        "game_id": game_id, "fecha": fecha,
        "equipo_local": equipo_local, "equipo_visitante": equipo_visitante,
        "equipo_favorito": favorito, "prob_favorito": prob_favorito,
        "bandera": bandera,
    }).execute()
    print(f"  [MONEYLINE] {game_id}: favorito {favorito} ({prob_favorito:.1%}) "
          f"registrado, pendiente de resolucion")

def correr_jornada():
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Corrida diaria: {fecha_hoy} ===")
    partidos = obtener_cartelera_dia(fecha_hoy)
    print(f"Partidos encontrados: {len(partidos)}")

    ranking_del_dia = []
    ranking_ponches_del_dia = []
    ranking_totales_del_dia = []

    for p in partidos:
        if p['pitcher_local_id'] is None or p['pitcher_visitante_id'] is None:
            print(f"Saltando: {p['visitante']} @ {p['local']} - abridor no confirmado")
            continue

        game_id = f"{p['visitante']}@{p['local']}_{fecha_hoy.replace('-','')}"

        if ya_capturado_hoy(game_id, p['pitcher_local_id'], p['pitcher_visitante_id']):
            print(f"Ya capturado: {p['visitante']} @ {p['local']} (corrida anterior de hoy)")
            continue

        cuota_local, cuota_visitante = obtener_cuotas_espn(p['local'], p['visitante'], fecha_hoy)
        if cuota_local is None:
            print(f"Sin cuotas: {p['visitante']} @ {p['local']}")
            continue

        park_factor = PARK_FACTORS.get(p['local'], 1.00)

        try:
            resultado = analizar_partido_hoy(
                equipo_local=p['local'], equipo_visitante=p['visitante'],
                pitcher_id_local=p['pitcher_local_id'], pitcher_id_visitante=p['pitcher_visitante_id'],
                park_factor=park_factor, cuota_ml_local=cuota_local, cuota_ml_visitante=cuota_visitante,
                fecha_hoy=fecha_hoy
            )

            favorito = p['local'] if resultado['prob_local'] >= 0.5 else p['visitante']
            prob_favorito = resultado['prob_local'] if resultado['prob_local'] >= 0.5 else 1 - resultado['prob_local']
            ranking_del_dia.append({
                "partido": f"{p['visitante']} @ {p['local']}",
                "favorito": favorito,
                "prob": prob_favorito,
                "bandera": resultado['bandera']
            })

            try:
                guardar_resultado_moneyline(game_id, fecha_hoy, p['local'], p['visitante'],
                                              resultado['prob_local'], resultado['bandera'])
            except Exception as e:
                print(f"  (error guardando resultado_moneyline: {e})")

            matchup = {}
            try:
                matchup = calcular_matchup_lr(p, fecha_hoy)
                fv = matchup.get("factor_pitcher_visitante_vs_lineup_local")
                fl = matchup.get("factor_pitcher_local_vs_lineup_visitante")
                if fv is not None:
                    print(f"  Matchup {p['pitcher_visitante_nombre']} vs lineup {p['local']}: {fv:.3f}")
                if fl is not None:
                    print(f"  Matchup {p['pitcher_local_nombre']} vs lineup {p['visitante']}: {fl:.3f}")
            except Exception as e:
                print(f"  (matchup L/R no disponible: {e})")

            winpct_ratio = None
            try:
                winpct_ratio = factor_winpct(p['local'], p['visitante'], fecha_hoy)
                if winpct_ratio is not None:
                    print(f"  Win% ratio {p['local']}/{p['visitante']}: {winpct_ratio:.3f}")
            except Exception as e:
                print(f"  (win% no disponible: {e})")

            try:
                ponches_l = proyectar_ponches(p['pitcher_local_id'], fecha_hoy, 2026)
                ponches_v = proyectar_ponches(p['pitcher_visitante_id'], fecha_hoy, 2026)
                if ponches_l is not None:
                    print(f"  Ponches proyectados {p['pitcher_local_nombre']}: {ponches_l}")
                    ranking_ponches_del_dia.append({
                        "pitcher": p['pitcher_local_nombre'], "equipo": p['local'], "ponches": ponches_l
                    })
                if ponches_v is not None:
                    print(f"  Ponches proyectados {p['pitcher_visitante_nombre']}: {ponches_v}")
                    ranking_ponches_del_dia.append({
                        "pitcher": p['pitcher_visitante_nombre'], "equipo": p['visitante'], "ponches": ponches_v
                    })
            except Exception as e:
                print(f"  (ponches no disponibles: {e})")

            ctx = {}
            try:
                ctx = contexto_cualitativo(p['local'], p['visitante'], fecha_hoy)
                if ctx['clima']:
                    print(f"  Clima: {ctx['clima']['temperatura_f']}F, viento {ctx['clima']['viento_mph']}mph")
                if ctx['espn_predictor']:
                    print(f"  ESPN Predictor: {p['local']} {ctx['espn_predictor']['prob_local']}% - {p['visitante']} {ctx['espn_predictor']['prob_visitante']}%")
                if ctx['lesiones_local']:
                    print(f"  Lesiones {p['local']}: {ctx['lesiones_local']}")
                if ctx['lesiones_visitante']:
                    print(f"  Lesiones {p['visitante']}: {ctx['lesiones_visitante']}")
            except Exception as e:
                print(f"  (contexto cualitativo no disponible: {e})")

            guardar_snapshot(game_id, "espn", "moneyline", p['local'], cuota_local,
                              modelo_prob=resultado['prob_local'], modelo_bandera=resultado['bandera'])
            guardar_snapshot(game_id, "espn", "moneyline", p['visitante'], cuota_visitante,
                              modelo_prob=1-resultado['prob_local'], modelo_bandera=resultado['bandera'])

            diferencia_total_calc = None
            linea_total_calc = None
            proyeccion_total_calc = None

            total_info = obtener_total_espn(p['local'], p['visitante'], fecha_hoy)
            if total_info:
                total_resultado = analizar_total(resultado['runs_local'], resultado['runs_visitante'],
                                                    total_info['over_odds'], total_info['under_odds'], total_info['linea'])
                guardar_diagnostico_total(game_id, fecha_hoy, total_resultado['total_esperado'],
                                            total_info['linea'], park_factor,
                                            resultado.get('era_local', 0), resultado.get('era_visitante', 0),
                                            p['pitcher_local_id'], p['pitcher_visitante_id'])
                linea_total_calc = total_info['linea']
                proyeccion_total_calc = total_resultado['total_esperado']
                diferencia_total_calc = proyeccion_total_calc - linea_total_calc
                ranking_totales_del_dia.append({
                    "partido": f"{p['visitante']} @ {p['local']}",
                    "linea": linea_total_calc,
                    "proyectado": proyeccion_total_calc,
                    "diferencia": diferencia_total_calc
                })

            # --- Auditoria de descartes: Total con diferencia >= 1.0 pero moneyline "alineado" ---
            if resultado['bandera'] == 'alineado' and diferencia_total_calc is not None and abs(diferencia_total_calc) >= 1.0:
                direccion = "Over" if diferencia_total_calc > 0 else "Under"
                motivo = (f"Moneyline sin edge (alineado), pero Total con diferencia significativa "
                          f"({direccion} {diferencia_total_calc:+.2f}). Revisar manualmente: posible variable "
                          f"no capturada por el modelo (bullpen, clima, lineup) o simplemente varianza normal.")
                guardar_auditoria_descarte(
                    game_id, fecha_hoy, tipo_descarte="total_alto_moneyline_alineado",
                    prob_modelo=resultado['prob_local'], prob_mercado=None,
                    diferencia_moneyline=None,
                    linea_total=linea_total_calc, proyeccion_total=proyeccion_total_calc,
                    diferencia_total=diferencia_total_calc, bandera_moneyline=resultado['bandera'],
                    motivo=motivo
                )

            # --- NUEVO: guardar contexto del partido (Fase 0 - instrumentacion) ---
            try:
                guardar_contexto_partido(
                    game_id, fecha_hoy, park_factor, ctx, matchup, winpct_ratio,
                    resultado.get("bullpen_era_local"), resultado.get("bullpen_era_visitante"),
                    resultado.get("bullpen_n_relevistas_local"), resultado.get("bullpen_n_relevistas_visitante")
                )
            except Exception as e:
                print(f"  (error guardando contexto_partido: {e})")

            print(f"{p['visitante']} @ {p['local']}: {resultado['recomendacion']} (bandera: {resultado['bandera']})")

        except Exception as e:
            print(f"Error en {p['visitante']} @ {p['local']}: {e}")

    print("\n=== RANKING DEL DIA - FAVORITOS (SOLO SEGUIMIENTO Y APRENDIZAJE, NO ES RECOMENDACION DE APUESTA) ===")
    ranking_ordenado = sorted(ranking_del_dia, key=lambda x: x['prob'], reverse=True)
    for i, r in enumerate(ranking_ordenado, 1):
        print(f"{i}. {r['favorito']} favorito ({r['prob']:.1%}) - {r['partido']} [bandera: {r['bandera']}]")

    print("\n=== RANKING DEL DIA - PONCHES PROYECTADOS (SOLO SEGUIMIENTO Y APRENDIZAJE) ===")
    ranking_ponches_ordenado = sorted(ranking_ponches_del_dia, key=lambda x: x['ponches'], reverse=True)
    for i, r in enumerate(ranking_ponches_ordenado, 1):
        print(f"{i}. {r['pitcher']} ({r['equipo']}): {r['ponches']} ponches proyectados")

    print("\n=== RANKING DEL DIA - TOTALES (LINEA VS PROYECCION, SOLO SEGUIMIENTO) ===")
    ranking_totales_ordenado = sorted(ranking_totales_del_dia, key=lambda x: abs(x['diferencia']), reverse=True)
    for i, r in enumerate(ranking_totales_ordenado, 1):
        direccion = "Over" if r['diferencia'] > 0 else "Under"
        print(f"{i}. {r['partido']}: Linea {r['linea']} | Proyectado {r['proyectado']:.2f} | {direccion} ({r['diferencia']:+.2f})")

    print("\n=== Corrida completa ===")

if __name__ == "__main__":
    correr_jornada()
