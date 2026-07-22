import requests, json

PITCHER_ID_PRUEBA = 543037

def probar():
    url = f"https://statsapi.mlb.com/api/v1/people/{PITCHER_ID_PRUEBA}/stats?stats=statSplits&group=pitching&season=2024&sitCodes=vl,vr"
    r = requests.get(url)
    print("Status:", r.status_code)
    data = r.json()
    print(json.dumps(data, indent=2)[:3000])

def probar_batside():
    ids_prueba = "592450,624413,665742"
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_prueba}&hydrate=stats"
    r = requests.get(url)
    data = r.json()
    for persona in data.get("people", []):
        print(persona.get("fullName"), "->", persona.get("batSide"))

if __name__ == "__main__":
    probar_batside()
