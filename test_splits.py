import requests, json

def probar_campos_completos():
    pitcher_id = 543037
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=statSplits&group=pitching&season=2024&sitCodes=vl,vr"
    r = requests.get(url)
    data = r.json()
    splits = data["stats"][0]["splits"]
    for s in splits:
        stat = s.get("stat", {})
        codigo = s.get("split", {}).get("code")
        print(f"\n--- Split: {codigo} ---")
        print(json.dumps(stat, indent=2))

if __name__ == "__main__":
    probar_campos_completos()
