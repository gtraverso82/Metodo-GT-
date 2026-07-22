import requests, json

def probar_earnedruns():
    pitcher_id = 543037  # Gerrit Cole
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=statSplits&group=pitching&season=2024&sitCodes=vl,vr"
    r = requests.get(url)
    data = r.json()
    splits = data["stats"][0]["splits"]
    for s in splits:
        stat = s.get("stat", {})
        codigo = s.get("split", {}).get("code")
        print(f"Split: {codigo} | earnedRuns: {stat.get('earnedRuns')} | inningsPitched: {stat.get('inningsPitched')} | era: {stat.get('era')}")

if __name__ == "__main__":
    probar_earnedruns()
