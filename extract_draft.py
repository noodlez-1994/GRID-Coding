"""
extract_draft.py
Extracts draft phase data (bans + picks) from GRID-2026-comp local summary files
and S3 events JSONL files. Outputs picks.csv and bans.csv.
"""
import json
import os
import re
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
S3_BUCKET  = os.environ["S3_BUCKET"]
S3_PREFIX  = os.environ.get("S3_PREFIX", "Competitive")
AWS_KEY    = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]

# ParticipantId → role  (Riot standard ordering for competitive)
ROLE_BY_PID = {1: "TOP", 2: "JGL", 3: "MID", 4: "BOT", 5: "SUP",
               6: "TOP", 7: "JGL", 8: "MID", 9: "BOT", 10: "SUP"}

# Draft pick-turn ranges (global, inclusive) that are PICK actions
# Phase-1 picks: 7-12, Phase-2 picks: 17-20
PICK_TURN_RANGES = [(7, 12), (17, 20)]


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_champ_map() -> dict:
    """Returns {champion_key_int: champion_name}."""
    with open(DATA_DIR / "champs.json") as f:
        data = json.load(f)
    return {int(v["key"]): v["name"] for v in data["data"].values()}


def is_pick_turn(t: int) -> bool:
    return any(lo <= t <= hi for lo, hi in PICK_TURN_RANGES)


def pick_phase(t: int) -> int:
    return 1 if 7 <= t <= 12 else 2


def ban_phase(t: int) -> int:
    """Summary ban turns 1-6 = phase 1, 7-10 = phase 2."""
    return 1 if t <= 6 else 2


def team_from_display(display_name: str) -> tuple[str, str]:
    """'DK Siwoo' → ('DK', 'Siwoo').  Handles multi-word team names."""
    parts = display_name.strip().split(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return display_name, display_name


# ── S3 fetch ─────────────────────────────────────────────────────────────────
def fetch_events(s3_client, series_id: str, game_num: int) -> list[dict] | None:
    key = f"{S3_PREFIX}/events_{series_id}_{game_num}_riot.jsonl"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    except Exception as e:
        print(f"  WARN: could not fetch s3://{S3_BUCKET}/{key}: {e}")
        return None


# ── Draft extraction from events ─────────────────────────────────────────────
def extract_pick_turns(events: list[dict]) -> dict[str, tuple[int, int]]:
    """
    Returns {display_name: (pick_turn_global, first_locked_champion_id)}.

    Scans CHAMP_SELECT events chronologically and records, for each player,
    their scheduled pick turn AND the champion they FIRST locked in (before any
    end-of-draft swaps).  This is the champion that was actually committed to
    that draft slot, not the champion the player ended up playing after swaps.
    """
    cs_events = sorted(
        [e for e in events if e.get("gameState") == "CHAMP_SELECT"],
        key=lambda e: (e["pickTurn"], e.get("sequenceIndex", 0)),
    )
    if not cs_events:
        return {}

    # First pass: collect each player's scheduled pick turn from the final snapshot
    final = cs_events[-1]
    scheduled: dict[str, int] = {}
    for player in final.get("teamOne", []) + final.get("teamTwo", []):
        pt = player.get("pickTurn", 0)
        name = player.get("displayName") or player.get("gameName") or player.get("summonerName", "")
        if name and is_pick_turn(pt):
            scheduled[name] = pt

    # Second pass: find the FIRST non-zero championID for each player as we walk
    # forward through the event timeline.  The first lock-in is the actual draft
    # pick; any later change is a post-draft swap and should be ignored.
    first_champ: dict[str, int] = {}
    for e in cs_events:
        for player in e.get("teamOne", []) + e.get("teamTwo", []):
            name = player.get("displayName") or player.get("gameName") or player.get("summonerName", "")
            cid  = player.get("championID", 0)
            if name and name in scheduled and name not in first_champ and cid != 0:
                first_champ[name] = cid

    return {name: (scheduled[name], first_champ.get(name, 0)) for name in scheduled}


# ── Summary parsing ───────────────────────────────────────────────────────────
def parse_summary(path: Path, champ_map: dict) -> tuple[list[dict], list[dict]] | None:
    with open(path) as f:
        data = json.load(f)

    teams_by_id = {t["teamId"]: t for t in data["teams"]}
    participants = data["participants"]

    # ── team-name lookup from participant display names ────────────────────
    # build {team_id: team_name}
    team_id_to_name: dict[int, str] = {}
    for p in participants:
        name = p.get("riotIdGameName") or p.get("summonerName", "")
        team_name, _ = team_from_display(name)
        if team_name:
            tid = p["teamId"]
            if tid not in team_id_to_name:
                team_id_to_name[tid] = team_name

    team_ids = sorted(team_id_to_name)
    opp_of = {}
    if len(team_ids) == 2:
        opp_of[team_ids[0]] = team_id_to_name[team_ids[1]]
        opp_of[team_ids[1]] = team_id_to_name[team_ids[0]]

    blue_team = team_id_to_name.get(100, "?")
    red_team  = team_id_to_name.get(200, "?")

    # ── bans ──────────────────────────────────────────────────────────────
    bans: list[dict] = []
    for tid, team in teams_by_id.items():
        team_wins = team["win"]
        team_name = team_id_to_name.get(tid, "?")
        opp_name  = opp_of.get(tid, "?")
        side      = "Blue" if tid == 100 else "Red"

        sorted_bans = sorted(team["bans"], key=lambda b: b["pickTurn"])
        for i, b in enumerate(sorted_bans, start=1):
            cid = b["championId"]
            if cid <= 0:
                continue
            bans.append({
                "team":            team_name,
                "opp_team":        opp_name,
                "blue_team":       blue_team,
                "red_team":        red_team,
                "side":            side,
                "champion":        champ_map.get(cid, f"ID:{cid}"),
                "ban_order":       i,           # 1-5 within team
                "ban_turn_global": b["pickTurn"],# 1-10 across all bans
                "ban_phase":       ban_phase(b["pickTurn"]),
                "win":             team_wins,
                "result":          "Win" if team_wins else "Loss",
            })

    # ── picks (from participants) ─────────────────────────────────────────
    picks: list[dict] = []
    for p in participants:
        pid     = p["participantId"]
        tid     = p["teamId"]
        name    = p.get("riotIdGameName") or p.get("summonerName", "")
        team_name, player_name = team_from_display(name)
        opp_name = opp_of.get(tid, "?")
        side     = "Blue" if tid == 100 else "Red"
        role     = ROLE_BY_PID.get(pid, "UNK")
        team_wins = teams_by_id[tid]["win"]

        picks.append({
            "team":         team_name,
            "opp_team":     opp_name,
            "blue_team":    blue_team,
            "red_team":     red_team,
            "side":         side,
            "player":       player_name,
            "display_name": name,
            "champion":     p["championName"],
            "champion_id":  p["championId"],
            "role":         role,
            "participant_id": pid,
            "win":          team_wins,
            "result":       "Win" if team_wins else "Loss",
        })

    return bans, picks


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    champ_map = load_champ_map()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="eu-west-1",
    )

    all_picks: list[dict] = []
    all_bans:  list[dict] = []

    # Discover all summary files
    summary_files = sorted(
        DATA_DIR.glob("end_state_summary_riot_*.json"),
        key=lambda p: (
            re.search(r"riot_(\d+)_(\d+)", p.name).group(1),
            int(re.search(r"riot_(\d+)_(\d+)", p.name).group(2)),
        )
    )

    for path in summary_files:
        m = re.search(r"riot_(\d+)_(\d+)", path.name)
        if not m:
            continue
        series_id, game_num = m.group(1), int(m.group(2))
        print(f"Processing series={series_id}  game={game_num} ...", end=" ")

        result = parse_summary(path, champ_map)
        if result is None:
            print("SKIP (parse error)")
            continue
        bans, picks = result

        # Enrich picks with pick-order and draft champion from S3 events
        events = fetch_events(s3, series_id, game_num)
        pick_turns: dict[str, tuple[int, int]] = {}
        if events:
            pick_turns = extract_pick_turns(events)

        for p in picks:
            info = pick_turns.get(p["display_name"])
            if info:
                pt, first_cid = info
            else:
                pt, first_cid = None, None
            p["pick_turn_global"] = pt          # 7-20 (None if events unavailable)
            p["pick_phase"]       = pick_phase(pt) if pt else None
            # draft_champion: champion actually committed at this pick slot
            # (before end-of-draft swaps).  Falls back to final champion when
            # events are unavailable.
            p["draft_champion"]   = champ_map.get(first_cid, p["champion"]) if first_cid else p["champion"]
            p["series_id"]        = series_id
            p["game_num"]         = game_num
            # Remove helper key
            del p["display_name"]

        for b in bans:
            b["series_id"] = series_id
            b["game_num"]  = game_num

        all_picks.extend(picks)
        all_bans.extend(bans)
        print(f"OK ({len(picks)} picks, {len(bans)} bans)")

    # ── Build DataFrames ─────────────────────────────────────────────────
    picks_df = pd.DataFrame(all_picks)
    bans_df  = pd.DataFrame(all_bans)

    # Assign pick_order (1-10) per game, sorted by pick_turn_global
    # For rows where pick_turn_global is None, fall back to participant_id ordering
    picks_df = picks_df.sort_values(
        ["series_id", "game_num", "pick_turn_global", "participant_id"],
        na_position="last",
    )
    picks_df["pick_order"] = (
        picks_df.groupby(["series_id", "game_num"]).cumcount() + 1
    )

    picks_cols = [
        "series_id", "game_num",
        "blue_team", "red_team",
        "team", "opp_team", "side",
        "player", "champion", "draft_champion", "role",
        "pick_order", "pick_turn_global", "pick_phase",
        "win", "result",
    ]
    bans_cols = [
        "series_id", "game_num",
        "blue_team", "red_team",
        "team", "opp_team", "side",
        "champion",
        "ban_order", "ban_turn_global", "ban_phase",
        "win", "result",
    ]

    picks_df = picks_df[picks_cols]
    bans_df  = bans_df[bans_cols]

    picks_out = DATA_DIR / "picks.csv"
    bans_out  = DATA_DIR / "bans.csv"
    picks_df.to_csv(picks_out, index=False)
    bans_df.to_csv(bans_out,  index=False)

    print(f"\n✓ Saved {len(picks_df)} pick rows → {picks_out}")
    print(f"✓ Saved {len(bans_df)} ban rows  → {bans_out}")
    print("\nPicks preview:")
    print(picks_df.head(10).to_string())
    print("\nBans preview:")
    print(bans_df.head(10).to_string())


if __name__ == "__main__":
    main()
