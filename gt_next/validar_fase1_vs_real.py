def extraer_equipos_unicos():
    """
    Extrae los códigos de equipo reales desde game_id (formato
    'VISITANTE@LOCAL_YYYY-MM-DD'), en vez de adivinar abreviaturas.
    Evita repetir el bug de mismatch AZ/CWS que ya costó tiempo en
    la integración con ESPN.
    """
    sb = get_client()
    equipos = set()
    todos_game_ids = []
    page_size = 1000
    start = 0
    while True:
        resp = sb.table(TABLA).select("game_id").range(start, start + page_size - 1).execute()
        filas = resp.data
        if not filas:
            break
        for fila in filas:
            gid = fila.get("game_id", "")
            todos_game_ids.append(gid)
            partes = gid.rsplit("_", 1)
            if len(partes) == 2 and "@" in partes[0]:
                visitante, local = partes[0].split("@")
                equipos.add(visitante)
                equipos.add(local)
        if len(filas) < page_size:
            break
        start += page_size

    print(f"Equipos únicos encontrados ({len(equipos)}):")
    for eq in sorted(equipos):
        print(f"  - {eq}")

    if len(equipos) != 30:
        print(f"\n⚠️  Se esperaban 30 equipos, se encontraron {len(equipos)}.")
        print("Buscando variantes de abreviatura sospechosas...")
        variantes = ["ARI", "AZ", "CWS", "CHW", "WSH", "WAS", "SD", "SDP", "TB", "TBR", "KC", "KCR", "SF", "SFG"]
        for v in variantes:
            cuenta = sum(1 for g in todos_game_ids if v in g)
            if cuenta > 0:
                print(f"  '{v}' aparece en {cuenta} game_id")

    return sorted(equipos)
