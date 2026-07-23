import requests, json

def probar_standings():
    fecha = "2026-07-22"
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2026&date={fecha}"
    r = requests.get(url, timeout=10)
    print("Status:", r.status_code)
    data = r.json()
    print(json.dumps(data, indent=2)[:2000])

if __name__ == "__main__":
    probar_standings()
