import requests, json

PITCHER_ID_PRUEBA = 543037  # Gerrit Cole, como referencia conocida

def probar():
    url = f"https://statsapi.mlb.com/api/v1/people/{PITCHER_ID_PRUEBA}/stats?stats=statSplits&group=pitching&season=2024&sitCodes=vl,vr"
    r = requests.get(url)
    print("Status:", r.status_code)
    data = r.json()
    print(json.dumps(data, indent=2)[:3000])

if __name__ == "__main__":
    probar()
