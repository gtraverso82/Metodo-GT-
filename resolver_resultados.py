"""
Resuelve automaticamente las predicciones de resultado_moneyline que estan
pendientes (gano_local IS NULL), consultando el resultado real de cada
partido via MLB Stats API.

Diseñado para correr una vez al dia (ej. temprano en la mañana), resolviendo
los partidos de dias anteriores que ya terminaron. Los partidos de hoy que
aun no se jugaron simplemente se saltan (quedan pendientes para la proxima
corrida de este script).

Uso: python resolver_resultados.py
"""

import os
import requests
from datetime import datetime, timedelta
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def obtener_pendientes():
    """Trae todas las filas de resultado_moneyline sin resolver, paginando
    completo por si acumulan mas de 1000."""
    PAGE_SIZE = 1000
    todas = []
    inicio = 0
    while True:
        fin = inicio + PAGE_SIZE - 1
        r = supabase.table("resultado_moneyline").select("*") \
            .is_("gano_local", "null").range(inicio, fin).execute()
        filas = r.data
        todas.extend(filas)
        if len(filas) < PAGE_SIZE:
            break
        inicio += PAGE_SIZE
    return todas


def obtener_resultados_reales_fecha(fecha):
    """Consulta MLB Stats API y retorna un dict {(home_abbr, away_abbr): (score_home, score_away)}
    para todos los juegos finalizados de esa fecha.

    hydrate=linescore,team (no solo linescore): sin el hydrate de 'team',
    MLB Stats API a veces devuelve el objeto team resumido (solo id/name/link,
    sin 'abbreviation'), lo que rompia esta funcion con un KeyError y tumbaba
    la corrida completa. Ademas, un try/except por partido evita que un caso
    puntual con estructura rara bloquee la resolucion de todos los demas.
    """
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha}"
           f"&hydrate=linescore,team&gameType=R")
    r = requests.get(url, timeout=15)
    data = r.json()
    resultados = {}
    for fecha_obj in data.get("dates", []):
        for juego in fecha_obj.get("games", []):
            estado = juego.get("status", {}).get("abstractGameState")
            if estado != "Final":
                continue
            try:
                home = juego["teams"]["home"]
                away = juego["teams"]["away"]
                home_abbr = home["team"]["abbreviation"]
                away_abbr = away["team"]["abbreviation"]
            except KeyError as e:
                game_pk = juego.get("gamePk", "desconocido")
                print(f"    (aviso: juego {game_pk} sin '{e.args[0]}' en la respuesta de "
                      f"MLB Stats API, se omite este partido para {fecha})")
                continue
            score_home = home.get("score")
            score_away = away.get("score")
            if score_home is not None and score_away is not None:
                resultados[(home_abbr, away_abbr)] = (score_home, score_away)
    return resultados


def main():
    print("=== Resolviendo predicciones pendientes de resultado_moneyline ===\n")
    pendientes = obtener_pendientes()
    print(f"Predicciones pendientes encontradas: {len(pendientes)}\n")

    if not pendientes:
        print("Nada pendiente. Fin.")
        return

    # Agrupar pendientes por fecha para minimizar llamadas a la API
    fechas_unicas = sorted(set(p["fecha"] for p in pendientes))
    resultados_por_fecha = {}
    for fecha in fechas_unicas:
        print(f"Consultando resultados reales para {fecha}...")
        resultados_por_fecha[fecha] = obtener_resultados_reales_fecha(fecha)

    resueltos = 0
    aun_pendientes = 0

    for p in pendientes:
        fecha = p["fecha"]
        local = p["equipo_local"]
        visitante = p["equipo_visitante"]
        resultados_fecha = resultados_por_fecha.get(fecha, {})
        marcador = resultados_fecha.get((local, visitante))

        if marcador is None:
            aun_pendientes += 1
            continue

        score_local, score_visitante = marcador
        if score_local == score_visitante:
            print(f"  {p['game_id']}: empate raro ({score_local}-{score_visitante}), se omite")
            continue

        gano_local = score_local > score_visitante
        favorito_era_local = (p["equipo_favorito"] == local)
        favorito_acerto = (favorito_era_local and gano_local) or \
                          (not favorito_era_local and not gano_local)

        supabase.table("resultado_moneyline").update({
            "gano_local": gano_local,
            "favorito_acerto": favorito_acerto,
            "resuelto_at": datetime.utcnow().isoformat()
        }).eq("id", p["id"]).execute()

        estado = "ACERTO" if favorito_acerto else "FALLO"
        print(f"  {p['game_id']}: favorito {p['equipo_favorito']} ({p['bandera']}) -> {estado} "
              f"(resultado {score_local}-{score_visitante})")
        resueltos += 1

    print(f"\n=== Resueltos: {resueltos} | Aun pendientes (partido no terminado): {aun_pendientes} ===")


if __name__ == "__main__":