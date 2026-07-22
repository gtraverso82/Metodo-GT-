def probar_batside():
    ids_prueba = "592450,624413,665742"  # Judge, Rizzo, Torres como referencia
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_prueba}&hydrate=stats"
    r = requests.get(url)
    data = r.json()
    for persona in data.get("people", []):
        print(persona.get("fullName"), "->", persona.get("batSide"))

if __name__ == "__main__":
    probar()
    probar_batside()
