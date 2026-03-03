"""
extract_wards.py
Extracts ward placement data from S3 events JSONL files.
Outputs wards.csv with positions, types, teams, timing.

Run:
    python extract_wards.py
"""
import json
import os
import re
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DATA_DIR   = Path(__file__).parent
S3_BUCKET  = os.environ["S3_BUCKET"]
S3_PREFIX  = os.environ.get("S3_PREFIX", "Competitive")
AWS_KEY    = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]


def fetch_events(s3_client, series_id: str, game_num: int) -> list[dict] | None:
    key = f"{S3_PREFIX}/events_{series_id}_{game_num}_riot.jsonl"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    except Exception as e:
        print(f"  WARN: {e}")
        return None


def parse_game_meta(path: Path) -> tuple[str, str]:
    """Returns (patch, date_str) from end_state_summary JSON."""
    try:
        with open(path) as f:
            d = json.load(f)
        ts_ms = d.get("gameStartTimestamp") or d.get("gameCreation", 0)
        ver   = d.get("gameVersion", "")
        parts = ver.split(".")
        patch = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else ver
        date  = pd.to_datetime(ts_ms, unit="ms").normalize().date().isoformat()
        return patch, date
    except Exception:
        return "", ""


def team_from_display(display_name: str) -> str:
    parts = display_name.strip().split(" ", 1)
    return parts[0] if parts else display_name


def main():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="eu-west-1",
    )

    summary_files = sorted(
        DATA_DIR.glob("end_state_summary_riot_*.json"),
        key=lambda p: (
            re.search(r"riot_(\d+)_(\d+)", p.name).group(1),
            int(re.search(r"riot_(\d+)_(\d+)", p.name).group(2)),
        ),
    )

    all_wards: list[dict] = []

    for path in summary_files:
        m = re.search(r"riot_(\d+)_(\d+)", path.name)
        if not m:
            continue
        series_id, game_num = m.group(1), int(m.group(2))
        patch, date = parse_game_meta(path)

        print(f"Processing series={series_id} game={game_num} ...", end=" ")

        events = fetch_events(s3, series_id, game_num)
        if not events:
            print("SKIP (no events)")
            continue

        # Build participantID → {team_name, side, player, role, champion} from game_info
        gi = next((e for e in events if e.get("rfc461Schema") == "game_info"), None)
        pid_info: dict[int, dict] = {}
        blue_team = red_team = ""
        if gi:
            for p in gi.get("participants", []):
                pid      = p["participantID"]
                display  = p.get("riotId", {}).get("displayName", "")
                tname    = team_from_display(display)
                # Player name is everything after the team prefix
                player   = display.split(" ", 1)[1] if " " in display else display
                side     = "Blue" if p["teamID"] == 100 else "Red"
                role_raw = p.get("role", "")
                # Normalise role to standard abbreviation
                role_map = {
                    "Top": "TOP", "Jungle": "JGL", "Middle": "MID",
                    "Bottom": "BOT", "Support": "SUP",
                }
                role = role_map.get(role_raw, role_raw.upper())
                pid_info[pid] = {
                    "team": tname, "side": side,
                    "player": player, "role": role,
                    "champion": p.get("championName", ""),
                }
                if side == "Blue" and not blue_team:
                    blue_team = tname
                elif side == "Red" and not red_team:
                    red_team = tname
        else:
            # fallback: 1-5 blue, 6-10 red
            for i in range(1, 6):
                pid_info[i] = {"team": "Blue", "side": "Blue",
                               "player": "", "role": "", "champion": ""}
            for i in range(6, 11):
                pid_info[i] = {"team": "Red", "side": "Red",
                               "player": "", "role": "", "champion": ""}

        ward_count = 0
        for e in events:
            if e.get("rfc461Schema") != "ward_placed":
                continue
            placer   = e.get("placer", 0)
            pos      = e.get("position", {})
            x        = pos.get("x")
            z        = pos.get("z")
            if x is None or z is None:
                continue
            ward_type = e.get("wardType", "unknown")
            info = pid_info.get(placer, {
                "team": "Unknown", "side": "Unknown",
                "player": "", "role": "", "champion": "",
            })

            all_wards.append({
                "series_id":   series_id,
                "game_num":    game_num,
                "game_time_s": e.get("gameTime", 0) / 1000,
                "x":           x,
                "z":           z,
                "ward_type":   ward_type,
                "placer_id":   placer,
                "side":        info["side"],
                "team":        info["team"],
                "player":      info["player"],
                "role":        info["role"],
                "champion":    info["champion"],
                "blue_team":   blue_team,
                "red_team":    red_team,
                "patch":       patch,
                "date":        date,
            })
            ward_count += 1

        print(f"OK ({ward_count} wards)")

    df = pd.DataFrame(all_wards)
    out = DATA_DIR / "wards.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Saved {len(df)} ward rows → {out}")
    print(df.head())


if __name__ == "__main__":
    main()
