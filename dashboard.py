"""
dashboard.py  –  GRID 2026 Competitive Draft Dashboard

Run locally:
    streamlit run dashboard.py

Public access options:
  1) Streamlit Community Cloud – push to GitHub then share.streamlit.io
  2) ngrok tunnel – run `ngrok http 8501` after starting streamlit
"""
from pathlib import Path
import io
import os
import re
import json

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from dotenv import load_dotenv
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter
from yaml.loader import SafeLoader

load_dotenv(Path(__file__).parent / ".env")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GRID 2026 – Draft Dashboard",
    page_icon="⚔️",
    layout="wide",
)

# ── Authentication ─────────────────────────────────────────────────────────────
with open(Path(__file__).parent / "config.yaml") as _f:
    _auth_config = yaml.load(_f, Loader=SafeLoader)

_authenticator = stauth.Authenticate(
    _auth_config["credentials"],
    _auth_config["cookie"]["name"],
    _auth_config["cookie"]["key"],
    _auth_config["cookie"]["expiry_days"],
    auto_hash=False,
)

_authenticator.login()

if st.session_state.get("authentication_status") is False:
    st.error("Incorrect username or password.")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.stop()

# ── Authenticated ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    picks = pd.read_csv(DATA_DIR / "picks.csv")
    bans  = pd.read_csv(DATA_DIR / "bans.csv")

    # Build date + patch from local summary JSON files
    meta: list[dict] = []
    for path in DATA_DIR.glob("end_state_summary_riot_*.json"):
        m = re.search(r"riot_(\d+)_(\d+)", path.name)
        if not m:
            continue
        sid, gnum = m.group(1), int(m.group(2))
        try:
            with open(path) as f:
                d = json.load(f)
            ts_ms = d.get("gameStartTimestamp") or d.get("gameCreation", 0)
            ver   = d.get("gameVersion", "")
            parts = ver.split(".")
            patch = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else ver
            date  = pd.to_datetime(ts_ms, unit="ms").normalize()
            meta.append({"series_id": sid, "game_num": gnum,
                          "date": date, "patch": patch})
        except Exception:
            pass

    picks["series_id"] = picks["series_id"].astype(str)
    bans["series_id"]  = bans["series_id"].astype(str)

    if meta:
        meta_df = pd.DataFrame(meta)
        meta_df["series_id"] = meta_df["series_id"].astype(str)
        picks = picks.merge(meta_df, on=["series_id", "game_num"], how="left")
        bans  = bans.merge(meta_df,  on=["series_id", "game_num"], how="left")
    else:
        picks["date"]  = pd.NaT
        picks["patch"] = ""
        bans["date"]   = pd.NaT
        bans["patch"]  = ""

    picks["date"] = pd.to_datetime(picks["date"])
    bans["date"]  = pd.to_datetime(bans["date"])
    return picks, bans


picks_raw, bans_raw = load_data()

all_teams     = sorted(set(picks_raw["team"].unique()))
all_roles     = ["TOP", "JGL", "MID", "BOT", "SUP"]
all_game_nums = sorted(picks_raw["game_num"].unique())
all_patches   = sorted(picks_raw["patch"].dropna().unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")

sel_teams = st.sidebar.multiselect(
    "Teams",
    options=all_teams,
    default=[],
    placeholder="All teams",
)

all_role_opts = ["All"] + all_roles
sel_roles = st.sidebar.multiselect(
    "Roles",
    options=all_role_opts,
    default=["All"],
)

sel_game_nums = st.sidebar.multiselect(
    "Game # in series (fearless)",
    options=all_game_nums,
    default=[],
    placeholder="All games",
)

st.sidebar.markdown("---")

# Date filter
has_dates = picks_raw["date"].notna().any()
if has_dates:
    min_date = picks_raw["date"].min().date()
    max_date = picks_raw["date"].max().date()
    sel_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range",
    )
else:
    sel_dates = None

# Patch filter
if all_patches:
    sel_patches = st.sidebar.multiselect(
        "Patch / Game version",
        options=all_patches,
        default=[],
        placeholder="All patches",
    )
else:
    sel_patches = []

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Dataset:** {picks_raw['series_id'].nunique()} series · "
    f"{len(picks_raw) // 10} games · "
    f"{len(all_teams)} teams"
)

# ── Apply filters ─────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, team_col: str = "team") -> pd.DataFrame:
    out = df.copy()
    if sel_teams:
        out = out[out[team_col].isin(sel_teams)]
    if sel_game_nums:
        out = out[out["game_num"].isin(sel_game_nums)]
    if sel_dates and len(sel_dates) == 2 and "date" in out.columns:
        d0, d1 = pd.Timestamp(sel_dates[0]), pd.Timestamp(sel_dates[1])
        out = out[(out["date"] >= d0) & (out["date"] <= d1)]
    if sel_patches and "patch" in out.columns:
        out = out[out["patch"].isin(sel_patches)]
    return out


picks = apply_filters(picks_raw)
bans  = apply_filters(bans_raw)

# Filter roles only for picks
if "All" not in sel_roles and sel_roles:
    picks = picks[picks["role"].isin(sel_roles)]

# ── Helpers ───────────────────────────────────────────────────────────────────
def color_wr(val):
    """Cell colour for winrate columns in styled dataframes."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 60:
        return "background-color: #1e7e34; color: white"
    elif v >= 50:
        return "background-color: #28a745; color: white"
    elif v >= 40:
        return "background-color: #ffc107"
    else:
        return "background-color: #dc3545; color: white"


def winrate_table(df: pd.DataFrame, group_col: str, min_games: int = 1) -> pd.DataFrame:
    agg = (
        df.groupby(group_col)
        .agg(games=("win", "count"), wins=("win", "sum"))
        .reset_index()
    )
    agg["winrate"] = (agg["wins"] / agg["games"] * 100).round(1)
    agg = agg[agg["games"] >= min_games].sort_values("winrate", ascending=False)
    return agg


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("⚔️ GRID 2026 Competitive Draft Dashboard")

team_label  = ", ".join(sel_teams) if sel_teams else "All teams"
game_label  = f"Game(s) {sel_game_nums}" if sel_game_nums else "All games"
role_label  = ", ".join([r for r in sel_roles if r != "All"]) or "All roles"
patch_label = ", ".join(sel_patches) if sel_patches else "All patches"
st.caption(f"Showing: **{team_label}** · **{game_label}** · **{role_label}** · **{patch_label}**")

if picks.empty:
    st.warning("No picks match the current filters.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_wr, tab_game, tab_team, tab_bans, tab_raw, tab_wards, tab_moves = st.tabs([
    "🏆 Champion Winrates",
    "🎮 By Game #",
    "🛡️ Team Deep Dive",
    "🚫 Bans",
    "📋 Raw Data",
    "🗺️ Ward Heatmap",
    "🎬 Movements",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 – Champion Winrates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_wr:
    st.subheader("Champion Winrates (Picks)")

    c1, c2 = st.columns([1, 3])
    min_games = c1.slider("Min. games picked", 1, 10, 2, key="min_g_wr")

    wr = winrate_table(picks, "champion", min_games)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Champions picked", len(wr))
    m2.metric("Total picks", picks.shape[0])
    m3.metric("Avg winrate", f"{wr['winrate'].mean():.1f}%")

    blue_wr = picks[picks["side"] == "Blue"]["win"].mean() * 100
    m4.metric("Blue side winrate", f"{blue_wr:.1f}%")

    st.markdown("---")

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.markdown("**Winrate by Champion** (sorted by winrate)")
        chart_df = wr.set_index("champion")[["winrate"]].sort_values("winrate")
        st.bar_chart(chart_df, color="#5B9BD5", height=500)

    with col_table:
        st.markdown("**Champion stats**")
        display = wr.rename(columns={
            "champion": "Champion",
            "games":    "Games",
            "wins":     "Wins",
            "winrate":  "WR %",
        })
        st.dataframe(
            display.style.map(color_wr, subset=["WR %"]),
            hide_index=True,
            use_container_width=True,
            height=500,
        )

    # ── Draft Slot Analysis ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Draft Slot Analysis")
    st.caption(
        "Each game has 10 pick slots. The table below shows the standard order: "
        "**Slot 1** = Blue's 1st pick · **Slot 2** = Red's 1st · "
        "**Slots 3-4** = Red's 2nd & 3rd · **Slots 5-6** = Blue's 2nd & 3rd · "
        "**Slots 7-8** = Red's 4th & Blue's 4th · **Slots 9-10** = Blue's 5th & Red's 5th. "
        "Expand a slot to see which champions are most picked there and their winrates."
    )

    # Standard LoL competitive draft slot → side mapping
    SLOT_SIDE = {
        1: "Blue", 2: "Red",  3: "Red",  4: "Blue", 5: "Blue",
        6: "Red",  7: "Red",  8: "Blue", 9: "Blue", 10: "Red",
    }

    picks_slots = picks.dropna(subset=["pick_order"]).copy()
    picks_slots["pick_order"] = picks_slots["pick_order"].astype(int)

    min_slot_picks = st.slider("Min. picks per champion to show", 1, 5, 1, key="min_slot")

    for slot in range(1, 11):
        slot_df = picks_slots[picks_slots["pick_order"] == slot]
        if slot_df.empty:
            continue
        side     = SLOT_SIDE.get(slot, "?")
        total    = len(slot_df)
        slot_wr  = winrate_table(slot_df, "champion", min_slot_picks).head(15)

        with st.expander(
            f"Slot {slot}  ({side} side)  —  {total} picks across all games",
            expanded=False,
        ):
            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.bar_chart(
                    slot_wr.set_index("champion")[["winrate"]].sort_values("winrate"),
                    color="#E8734A",
                    height=280,
                )
            with c_right:
                st.dataframe(
                    slot_wr.rename(columns={
                        "champion": "Champion",
                        "games":    "Picks",
                        "wins":     "Wins",
                        "winrate":  "WR %",
                    }).style.map(color_wr, subset=["WR %"]),
                    hide_index=True,
                    use_container_width=True,
                    height=280,
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 – Champion Winrates by Game Number in Series
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_game:
    st.subheader("Fearless Format – Champion Winrates by Game Number")
    st.caption(
        "In a fearless series each champion may only be picked once per series. "
        "**Game 1** = first game of the series, **Game 2** = second, etc. "
        "Expand a game to see which champions were picked in that slot and their winrates."
    )

    # Blue/Red side winrate per game slot
    side_game = (
        picks_raw.groupby(["game_num", "side"])
        .agg(games=("win", "count"), wins=("win", "sum"))
        .reset_index()
    )
    side_game["winrate"] = (side_game["wins"] / side_game["games"] * 100).round(1)
    pivot_side = side_game.pivot(index="game_num", columns="side", values="winrate").fillna(0)

    st.markdown("**Blue / Red side winrate across series game slots**")
    st.bar_chart(pivot_side, color=["#1a6cb4", "#c0392b"])

    st.markdown("---")

    min_games_gn = st.slider("Min. picks to show a champion", 1, 5, 1, key="min_g_gn")

    st.markdown("**Champion winrates per game slot** — expand to see champion stats")

    for gn in sorted(picks_raw["game_num"].unique()):
        gn_picks = apply_filters(picks_raw[picks_raw["game_num"] == gn])
        if gn_picks.empty:
            continue

        # One unique series_id = one game played at this game number
        n_games  = gn_picks["series_id"].nunique()
        n_champs = gn_picks["champion"].nunique()

        with st.expander(
            f"Game {gn} in series  —  {n_games} games played · {n_champs} unique champions",
            expanded=(gn == 1),
        ):
            wr_gn = winrate_table(gn_picks, "champion", min_games_gn)

            if wr_gn.empty:
                st.info(f"No champions with ≥{min_games_gn} picks.")
                continue

            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.markdown(f"**Champion winrates in Game {gn}** (sorted by winrate)")
                st.bar_chart(
                    wr_gn.set_index("champion")[["winrate"]].sort_values("winrate"),
                    color="#9B59B6",
                    height=350,
                )
            with c_right:
                st.dataframe(
                    wr_gn.rename(columns={
                        "champion": "Champion",
                        "games":    "Picks",
                        "wins":     "Wins",
                        "winrate":  "WR %",
                    }).style.map(color_wr, subset=["WR %"]),
                    hide_index=True,
                    use_container_width=True,
                    height=350,
                )

    # Unique champions per game number
    st.markdown("---")
    st.subheader("Champion Pool Utilisation per Game")
    uniq = (
        picks_raw.groupby("game_num")["champion"]
        .nunique()
        .reset_index()
        .rename(columns={"champion": "unique_champions"})
    )
    st.bar_chart(uniq.set_index("game_num"), color="#27AE60")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 – Team Deep Dive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_team:
    st.subheader("Team Deep Dive")

    chosen_team = st.selectbox("Select a team", options=all_teams, key="team_dd")

    team_picks = picks_raw[picks_raw["team"] == chosen_team].copy()
    team_bans  = bans_raw[bans_raw["team"] == chosen_team].copy()

    if sel_game_nums:
        team_picks = team_picks[team_picks["game_num"].isin(sel_game_nums)]
        team_bans  = team_bans[team_bans["game_num"].isin(sel_game_nums)]

    # Each game the chosen team contributes exactly 5 rows (one per player)
    t_games = len(team_picks) // 5

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", t_games)
    t_wins = int(team_picks["win"].sum() // 5) if t_games > 0 else 0
    c2.metric("Wins", t_wins)
    c3.metric("Win rate", f"{t_wins / t_games * 100:.1f}%" if t_games > 0 else "-")
    c4.metric("Champions used", team_picks["champion"].nunique())

    st.markdown("---")

    col_picks, col_bans = st.columns(2)

    with col_picks:
        st.markdown(f"**{chosen_team} – Pick pool**")
        tp_wr = winrate_table(team_picks, "champion", 1).head(20)
        st.dataframe(
            tp_wr.rename(columns={
                "champion": "Champion",
                "games": "Games",
                "wins": "Wins",
                "winrate": "WR %",
            }).style.map(color_wr, subset=["WR %"]),
            hide_index=True,
            use_container_width=True,
            height=400,
        )

    with col_bans:
        st.markdown(f"**{chosen_team} – Ban pool**")
        tb_agg = (
            team_bans.groupby("champion")
            .agg(times_banned=("champion", "count"), wins=("win", "sum"))
            .reset_index()
            .sort_values("times_banned", ascending=False)
        )
        tb_agg["ban_wr"] = (tb_agg["wins"] / tb_agg["times_banned"] * 100).round(1)
        st.dataframe(
            tb_agg.rename(columns={
                "champion": "Champion",
                "times_banned": "Times Banned",
                "wins": "Wins",
                "ban_wr": "Win % (when banned)",
            }),
            hide_index=True,
            use_container_width=True,
            height=400,
        )

    st.markdown("---")
    st.subheader(f"{chosen_team} – Pick breakdown by role")
    for role in all_roles:
        role_picks = team_picks[team_picks["role"] == role]
        if role_picks.empty:
            continue
        with st.expander(f"{role}  ({role_picks['champion'].nunique()} unique champions)"):
            role_wr = winrate_table(role_picks, "champion", 1)
            c_r1, c_r2 = st.columns([3, 2])
            with c_r1:
                st.bar_chart(
                    role_wr.set_index("champion")[["winrate"]].sort_values("winrate"),
                    height=250,
                )
            with c_r2:
                st.dataframe(
                    role_wr.rename(columns={
                        "champion": "Champion",
                        "games": "Games",
                        "wins": "Wins",
                        "winrate": "WR %",
                    }),
                    hide_index=True,
                    use_container_width=True,
                    height=250,
                )

    st.markdown("---")
    st.subheader(f"{chosen_team} – Results by series game number")
    team_by_game = (
        team_picks.groupby("game_num")
        .apply(
            lambda df: pd.Series({
                "games": df["series_id"].nunique(),
                "wins":  int(df.groupby("series_id")["win"].first().sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    team_by_game["winrate"] = (team_by_game["wins"] / team_by_game["games"].replace(0, 1) * 100).round(1)
    st.bar_chart(team_by_game.set_index("game_num")[["winrate"]], color="#E67E22")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 – Bans
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_bans:
    st.subheader("Ban Analysis")

    ban_agg = (
        bans.groupby("champion")
        .agg(
            times_banned=("champion", "count"),
            wins=("win", "sum"),
        )
        .reset_index()
        .sort_values("times_banned", ascending=False)
    )
    ban_agg["ban_win_pct"] = (ban_agg["wins"] / ban_agg["times_banned"] * 100).round(1)

    b1, b2 = st.columns(4)[:2]
    b1.metric("Unique champions banned", ban_agg.shape[0])
    b2.metric("Total bans", bans.shape[0])

    st.markdown("---")

    c_freq, c_wr = st.columns(2)

    with c_freq:
        st.markdown("**Most Banned Champions**")
        top_banned = ban_agg.head(20)
        st.bar_chart(
            top_banned.set_index("champion")[["times_banned"]].sort_values("times_banned"),
            color="#E74C3C",
            height=400,
        )

    with c_wr:
        st.markdown("**Win % of team that banned the champion**")
        st.dataframe(
            ban_agg.rename(columns={
                "champion":     "Champion",
                "times_banned": "# Bans",
                "wins":         "Wins",
                "ban_win_pct":  "WR % (banning team)",
            }).style.map(color_wr, subset=["WR % (banning team)"]),
            hide_index=True,
            use_container_width=True,
            height=400,
        )

    # Phase-split view
    st.markdown("---")
    st.subheader("Bans by Phase")
    p1_bans = bans[bans["ban_phase"] == 1]
    p2_bans = bans[bans["ban_phase"] == 2]

    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown("**Phase 1 bans (turns 1-6)**")
        p1_agg = (
            p1_bans.groupby("champion")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(15)
        )
        st.bar_chart(p1_agg.set_index("champion")[["count"]], color="#8E44AD")

    with cp2:
        st.markdown("**Phase 2 bans (turns 7-10)**")
        p2_agg = (
            p2_bans.groupby("champion")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(15)
        )
        st.bar_chart(p2_agg.set_index("champion")[["count"]], color="#2980B9")

    # Ban order within team
    st.markdown("---")
    st.subheader("What teams ban by slot (1-5)")
    if sel_teams:
        ban_slot_df = bans[bans["team"].isin(sel_teams)].copy()
    else:
        ban_slot_df = bans.copy()

    ban_slot_pivot = (
        ban_slot_df.groupby(["ban_order", "champion"])
        .size()
        .reset_index(name="count")
        .sort_values(["ban_order", "count"], ascending=[True, False])
    )
    for slot in range(1, 6):
        slot_df = ban_slot_pivot[ban_slot_pivot["ban_order"] == slot].head(10)
        with st.expander(f"Ban slot {slot}"):
            st.bar_chart(slot_df.set_index("champion")[["count"]], height=200)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 – Raw Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_raw:
    st.subheader("Raw Data")

    raw_tab1, raw_tab2 = st.tabs(["Picks", "Bans"])

    with raw_tab1:
        st.caption(f"{len(picks)} rows after filters")
        st.dataframe(picks, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ Download picks (filtered)",
            picks.to_csv(index=False).encode(),
            file_name="picks_filtered.csv",
            mime="text/csv",
        )

    with raw_tab2:
        st.caption(f"{len(bans)} rows after filters")
        st.dataframe(bans, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ Download bans (filtered)",
            bans.to_csv(index=False).encode(),
            file_name="bans_filtered.csv",
            mime="text/csv",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 – Ward Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAP_IMG_PATH = Path("lolmap.png")
MAP_SIZE     = 15000  # LoL coordinate space (0-15000 for both x and z)

WARD_TYPE_LABELS = {
    "yellowTrinket": "Yellow Trinket",
    "control":       "Control Ward",
    "blueTrinket":   "Blue Trinket",
    "sight":         "Sight Ward",
    "unknown":       "Unknown",
}
WARD_TYPE_COLORS = {
    "yellowTrinket": "#F1C40F",
    "control":       "#9B59B6",
    "blueTrinket":   "#3498DB",
    "sight":         "#2ECC71",
    "unknown":       "#95A5A6",
}


@st.cache_data
def load_ward_data() -> pd.DataFrame:
    path = DATA_DIR / "wards.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["series_id"] = df["series_id"].astype(str)
    df["date"]      = pd.to_datetime(df["date"])
    return df


wards_raw = load_ward_data()


with tab_wards:
    st.subheader("Ward Heatmap")
    st.caption(
        "Ward placements overlaid on the Summoner's Rift minimap. "
        "Use the sidebar filters (Teams, Patch, Game #, Date) together with the "
        "controls below to narrow down which wards are shown."
    )

    if wards_raw.empty:
        st.error("wards.csv not found. Run `python extract_wards.py` to generate it.")
        st.stop()

    # ── Apply sidebar filters ────────────────────────────────────────────────
    def apply_ward_filters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if sel_teams:
            out = out[out["team"].isin(sel_teams)]
        if sel_game_nums:
            out = out[out["game_num"].isin(sel_game_nums)]
        if sel_dates and len(sel_dates) == 2 and "date" in out.columns:
            d0, d1 = pd.Timestamp(sel_dates[0]), pd.Timestamp(sel_dates[1])
            out = out[(out["date"] >= d0) & (out["date"] <= d1)]
        if sel_patches and "patch" in out.columns:
            out = out[out["patch"].isin(sel_patches)]
        return out

    wards = apply_ward_filters(wards_raw)

    # ── Tab-specific filter row ──────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([1, 2, 2, 1])

    sel_ward_side = fc1.selectbox(
        "Side",
        options=["Both", "Blue", "Red"],
        key="ward_side",
    )

    all_ward_types = list(WARD_TYPE_LABELS.keys())
    sel_ward_types = fc2.multiselect(
        "Ward types",
        options=all_ward_types,
        default=[t for t in all_ward_types if t != "unknown"],
        format_func=lambda t: WARD_TYPE_LABELS[t],
        key="ward_types",
    )

    all_roles_ward = ["TOP", "JGL", "MID", "BOT", "SUP"]
    sel_ward_roles = fc3.multiselect(
        "Roles (ward placer)",
        options=all_roles_ward,
        default=[],
        placeholder="All roles",
        key="ward_roles",
    )

    viz_mode = fc4.selectbox(
        "Viz mode",
        options=["Scatter", "Heatmap"],
        key="ward_viz",
    )

    # Time range slider (minutes)
    max_min = max(1, int(wards["game_time_s"].max() / 60) + 1) if not wards.empty else 45
    time_range = st.slider(
        "Game time window (minutes)",
        min_value=0,
        max_value=max_min,
        value=(0, 30),
        key="ward_time",
    )

    # Apply tab-specific filters
    if sel_ward_side != "Both":
        wards = wards[wards["side"] == sel_ward_side]
    if sel_ward_types:
        wards = wards[wards["ward_type"].isin(sel_ward_types)]
    if sel_ward_roles:
        wards = wards[wards["role"].isin(sel_ward_roles)]
    t_lo, t_hi = time_range[0] * 60, time_range[1] * 60
    wards = wards[(wards["game_time_s"] >= t_lo) & (wards["game_time_s"] <= t_hi)]

    # ── Metric row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total wards", f"{len(wards):,}")
    m2.metric("Blue side", f"{(wards['side'] == 'Blue').sum():,}")
    m3.metric("Red side",  f"{(wards['side'] == 'Red').sum():,}")
    n_games = wards[["series_id", "game_num"]].drop_duplicates().shape[0]
    m4.metric("Games", n_games)

    st.markdown("---")

    if wards.empty:
        st.info("No wards match the current filters.")
        st.stop()

    if not MAP_IMG_PATH.exists():
        st.warning(f"Map image not found: {MAP_IMG_PATH}")
        st.stop()

    # ── Build the map figure ─────────────────────────────────────────────────
    img_arr  = np.array(PILImage.open(MAP_IMG_PATH))
    img_h, img_w = img_arr.shape[:2]

    # Game coords → pixel coords.
    # LoL map: x increases left→right, z increases bottom→top.
    # Image: row 0 is top, so we flip z.
    px = (wards["x"].values / MAP_SIZE * img_w).clip(0, img_w - 1)
    py = ((1 - wards["z"].values / MAP_SIZE) * img_h).clip(0, img_h - 1)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    # Display map image; with origin="upper" row-0 is at top, matching our py
    ax.imshow(img_arr, origin="upper", extent=[0, img_w, img_h, 0], zorder=0)

    if viz_mode == "Scatter":
        # Colour per ward type; transparency to show density
        for wtype in (sel_ward_types or all_ward_types):
            mask = wards["ward_type"] == wtype
            if not mask.any():
                continue
            ax.scatter(
                px[mask.values], py[mask.values],
                c=WARD_TYPE_COLORS.get(wtype, "#AAAAAA"),
                alpha=0.55, s=50, linewidths=0,
                label=WARD_TYPE_LABELS[wtype],
                zorder=2,
            )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.75,
                  facecolor="#1e1e2e", labelcolor="white")

    else:  # Heatmap
        h, xedges, yedges = np.histogram2d(
            px, py, bins=80,
            range=[[0, img_w], [0, img_h]],
        )
        h_smooth = gaussian_filter(h.T, sigma=3)
        if h_smooth.max() > 0:
            h_masked = np.ma.masked_where(
                h_smooth < h_smooth.max() * 0.02, h_smooth
            )
            ax.imshow(
                h_masked, origin="upper",
                extent=[0, img_w, img_h, 0],
                cmap="hot", alpha=0.72,
                vmin=0, vmax=h_smooth.max(),
                zorder=2,
            )

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)

    col_map, col_stats = st.columns([3, 2])

    with col_map:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Right-column stats ───────────────────────────────────────────────────
    with col_stats:
        # Ward type breakdown
        st.markdown("**Ward type breakdown**")
        type_df = (
            wards["ward_type"]
            .map(WARD_TYPE_LABELS)
            .value_counts()
            .reset_index()
            .rename(columns={"ward_type": "Type", "count": "Count"})
        )
        st.dataframe(type_df, hide_index=True, use_container_width=True, height=190)

        # Wards per role
        st.markdown("**Wards by role**")
        role_df = (
            wards[wards["role"].notna() & (wards["role"] != "")]
            .groupby("role")
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        st.bar_chart(role_df.set_index("role"), color="#27AE60", height=180)

        # Wards per minute timeline
        st.markdown("**Wards placed per minute**")
        wards_tl = wards.copy()
        wards_tl["minute"] = (wards_tl["game_time_s"] // 60).astype(int)
        minute_df = wards_tl.groupby("minute").size().reset_index(name="wards")
        st.bar_chart(minute_df.set_index("minute"), color="#F39C12", height=180)

    # ── Per-player / per-champion breakdown ─────────────────────────────────
    st.markdown("---")
    st.subheader("Ward activity by player")

    player_df = (
        wards[wards["player"].notna() & (wards["player"] != "")]
        .groupby(["team", "player", "role", "champion"])
        .agg(wards_placed=("x", "count"))
        .reset_index()
        .sort_values("wards_placed", ascending=False)
    )
    st.dataframe(
        player_df.rename(columns={
            "team":         "Team",
            "player":       "Player",
            "role":         "Role",
            "champion":     "Champion",
            "wards_placed": "Wards Placed",
        }),
        hide_index=True,
        use_container_width=True,
        height=350,
    )

    # Per-player heatmaps in expanders
    st.markdown("---")
    st.subheader("Per-player ward map")
    st.caption("Expand a player to see their individual ward placement heatmap.")

    for _, row in player_df.iterrows():
        player_wards = wards[
            (wards["player"] == row["player"]) &
            (wards["team"]   == row["team"])
        ]
        label = (
            f"{row['team']} – {row['player']} ({row['role']}, {row['champion']}) "
            f"— {row['wards_placed']} wards"
        )
        with st.expander(label, expanded=False):
            ppx = (player_wards["x"].values / MAP_SIZE * img_w).clip(0, img_w - 1)
            ppy = ((1 - player_wards["z"].values / MAP_SIZE) * img_h).clip(0, img_h - 1)

            fig_p, ax_p = plt.subplots(figsize=(5, 5), facecolor="#0e1117")
            ax_p.set_facecolor("#0e1117")
            ax_p.imshow(img_arr, origin="upper", extent=[0, img_w, img_h, 0], zorder=0)

            for wtype in player_wards["ward_type"].unique():
                mask = player_wards["ward_type"] == wtype
                ax_p.scatter(
                    ppx[mask.values], ppy[mask.values],
                    c=WARD_TYPE_COLORS.get(wtype, "#AAAAAA"),
                    alpha=0.7, s=60, linewidths=0,
                    label=WARD_TYPE_LABELS.get(wtype, wtype),
                    zorder=2,
                )
            ax_p.legend(loc="lower right", fontsize=8, framealpha=0.75,
                        facecolor="#1e1e2e", labelcolor="white")
            ax_p.set_xlim(0, img_w)
            ax_p.set_ylim(img_h, 0)
            ax_p.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7 – Champion Movements
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MOVE_OBJ_TYPES  = {"dragon", "baron", "riftHerald"}
_MOVE_OBJ_LABELS = {"dragon": "Dragon", "baron": "Baron", "riftHerald": "Rift Herald"}
_MOVE_ROLE_NORM  = {
    "Top": "TOP", "Jungle": "JGL", "Middle": "MID",
    "Bottom": "BOT", "Support": "SUP",
}
_TEAM_COLOR      = {"Blue": "#4A90D9", "Red": "#E74C3C"}
_DDRAGON_VERSION = "16.3.1"
_ICON_DIR        = DATA_DIR / "champ_icons"
_ICON_CACHE: dict[str, PILImage.Image | None] = {}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _prep_icon(champion: str, side: str, alive: bool, dimmed: bool,
               size: int = 34) -> PILImage.Image | None:
    """
    Return a PIL RGBA image of the champion icon, sized `size`×`size` plus a
    3-px team-coloured border.  Results are cached in _ICON_CACHE.
    """
    import urllib.request
    from PIL import ImageDraw

    cache_key = f"{champion}_{side}_{alive}_{dimmed}_{size}"
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    _ICON_DIR.mkdir(exist_ok=True)
    raw_path = _ICON_DIR / f"{champion}.png"

    if not raw_path.exists():
        url = (
            f"http://ddragon.leagueoflegends.com/cdn/"
            f"{_DDRAGON_VERSION}/img/champion/{champion}.png"
        )
        try:
            urllib.request.urlretrieve(url, raw_path)
        except Exception:
            _ICON_CACHE[cache_key] = None
            return None

    try:
        img = PILImage.open(raw_path).convert("RGBA").resize(
            (size, size), PILImage.LANCZOS
        )
    except Exception:
        _ICON_CACHE[cache_key] = None
        return None

    # Greyscale for dead champions
    if not alive:
        grey_arr = np.array(img.convert("LA").convert("RGBA"))
        grey_arr[..., 3] = (grey_arr[..., 3] * 0.6).astype(np.uint8)
        img = PILImage.fromarray(grey_arr)

    # Fade out dimmed (non-highlighted) champions
    if dimmed:
        arr = np.array(img)
        arr[..., 3] = (arr[..., 3] * 0.30).astype(np.uint8)
        img = PILImage.fromarray(arr)

    # Add team-coloured border
    border     = 3
    total      = size + 2 * border
    bordered   = PILImage.new("RGBA", (total, total), (0, 0, 0, 0))
    draw       = ImageDraw.Draw(bordered)
    r, g, b    = _hex_to_rgb(_TEAM_COLOR.get(side, "#888888"))
    b_alpha    = 80 if dimmed else 255
    draw.rectangle([0, 0, total - 1, total - 1], fill=(r, g, b, b_alpha))
    bordered.paste(img, (border, border), img)

    _ICON_CACHE[cache_key] = bordered
    return bordered


@st.cache_resource
def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="eu-west-3",
    )


@st.cache_data(show_spinner="Loading game positions from S3…")
def load_game_positions(series_id: str, game_num: int):
    """
    Returns (positions_df, objectives, pid_info).
    positions_df: game_time_ms, participant_id, x, z, alive, player, champion, side, role
    objectives: list of dicts  {monster_type, game_time_ms, side, label, x, z}
    pid_info: {pid: {player, champion, side, role}}
    """
    key = f"Competitive/events_{series_id}_{game_num}_riot.jsonl"
    try:
        raw    = _s3_client().get_object(Bucket="s3-lol-datastorage", Key=key)["Body"].read().decode()
        events = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    except Exception as e:
        st.error(f"S3 error: {e}")
        return pd.DataFrame(), [], {}

    # participant info from game_info
    gi = next((e for e in events if e.get("rfc461Schema") == "game_info"), None)
    pid_info: dict[int, dict] = {}
    if gi:
        for p in gi.get("participants", []):
            pid     = p["participantID"]
            display = p.get("riotId", {}).get("displayName", "")
            player  = display.split(" ", 1)[1] if " " in display else display
            pid_info[pid] = {
                "player":   player,
                "champion": p.get("championName", ""),
                "side":     "Blue" if p["teamID"] == 100 else "Red",
                "role":     _MOVE_ROLE_NORM.get(p.get("role", ""), p.get("role", "")),
                "team_id":  p["teamID"],
            }

    # positions from stats_update (~1 Hz)
    rows: list[dict] = []
    for e in events:
        if e.get("rfc461Schema") != "stats_update":
            continue
        gt = e.get("gameTime", 0)
        for p in e.get("participants", []):
            pid = p.get("participantID")
            pos = p.get("position", {})
            if "x" not in pos:
                continue
            info = pid_info.get(pid, {})
            rows.append({
                "game_time_ms":  gt,
                "participant_id": pid,
                "x":     pos["x"],
                "z":     pos.get("z", pos.get("y", 0)),
                "alive": p.get("alive", True),
                "player":   info.get("player", "?"),
                "champion": info.get("champion", "?"),
                "side":     info.get("side", ""),
                "role":     info.get("role", ""),
            })

    # objectives from epic_monster_kill
    objectives: list[dict] = []
    for e in events:
        if e.get("rfc461Schema") != "epic_monster_kill":
            continue
        mtype = e.get("monsterType", "")
        if mtype not in _MOVE_OBJ_TYPES:
            continue
        gt  = e.get("gameTime", 0)
        pos = e.get("position", {})
        side = "Blue" if e.get("killerTeamID") == 100 else "Red"
        objectives.append({
            "monster_type": mtype,
            "game_time_ms": gt,
            "game_time_s":  gt / 1000,
            "side":         side,
            "label": (
                f"{_MOVE_OBJ_LABELS.get(mtype, mtype)} "
                f"@ {gt // 60000}:{(gt // 1000) % 60:02d}  ({side})"
            ),
            "x": pos.get("x", 0),
            "z": pos.get("z", pos.get("y", 0)),
        })

    return pd.DataFrame(rows), objectives, pid_info


def _ms_to_mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def _build_gif(pos_df: pd.DataFrame, start_ms: int, end_ms: int,
               sample_s: float, fps: int, highlight_pids: set | None,
               objective: dict | None) -> bytes | None:
    """Render champion movement as an animated GIF using PIL (no matplotlib)."""
    from PIL import ImageDraw

    mask   = (pos_df["game_time_ms"] >= start_ms) & (pos_df["game_time_ms"] <= end_ms)
    window = pos_df[mask]
    if window.empty:
        return None

    # Sample one frame every sample_s seconds
    all_ts: list[int] = sorted(window["game_time_ms"].unique())
    sampled: list[int] = []
    last = -1e9
    for ts in all_ts:
        if ts - last >= sample_s * 1000:
            sampled.append(ts)
            last = ts
    if not sampled:
        return None

    W, H      = 512, 512
    ICON_SIZE = 34
    ICON_HALF = (ICON_SIZE + 6) // 2   # half total icon footprint (inc. border)

    base_img = PILImage.open(MAP_IMG_PATH).resize((W, H), PILImage.LANCZOS).convert("RGBA")

    pil_frames: list[PILImage.Image] = []

    for ts in sampled:
        snap  = window[window["game_time_ms"] == ts]
        frame = base_img.copy()
        draw  = ImageDraw.Draw(frame)

        # Render dimmed champions first so highlighted ones sit on top
        ordered = sorted(
            snap.itertuples(index=False),
            key=lambda r: 0 if (highlight_pids and int(r.participant_id) in highlight_pids) else -1,
        )

        for row in ordered:
            px     = int(float(row.x) / MAP_SIZE * W)
            py     = int((1.0 - float(row.z) / MAP_SIZE) * H)
            alive  = bool(getattr(row, "alive", True))
            dimmed = bool(highlight_pids) and int(row.participant_id) not in highlight_pids

            icon = _prep_icon(str(row.champion), str(row.side), alive, dimmed, ICON_SIZE)

            if icon is not None:
                iw, ih = icon.size
                bx = max(0, min(W - iw, px - iw // 2))
                by = max(0, min(H - ih, py - ih // 2))
                frame.paste(icon, (bx, by), icon)

                # Role label below the icon (only for non-dimmed)
                if not dimmed:
                    role = str(getattr(row, "role", ""))
                    if role:
                        ty = by + ih + 1
                        if ty + 9 < H:
                            draw.text((px, ty), role,
                                      fill=(255, 255, 255, 210), anchor="mt")
            else:
                # Fallback coloured circle when icon unavailable
                cr, cg, cb = _hex_to_rgb(_TEAM_COLOR.get(str(row.side), "#888888"))
                alpha = 80 if dimmed else (140 if not alive else 220)
                dot_r = 5 if dimmed else 9
                draw.ellipse(
                    [px - dot_r, py - dot_r, px + dot_r, py + dot_r],
                    fill=(cr, cg, cb, alpha), outline=(255, 255, 255, alpha),
                )

        # Objective kill flash: marker visible for ±5 s around the kill
        if objective is not None and abs(ts - objective["game_time_ms"]) <= 5000:
            ox  = int(float(objective["x"]) / MAP_SIZE * W)
            oz  = int((1.0 - float(objective["z"]) / MAP_SIZE) * H)
            or_, og, ob = _hex_to_rgb(_TEAM_COLOR.get(objective["side"], "#FFD700"))
            sr  = 14
            draw.ellipse(
                [ox - sr, oz - sr, ox + sr, oz + sr],
                fill=(or_, og, ob, 210), outline=(255, 255, 255, 255), width=2,
            )
            obj_name = _MOVE_OBJ_LABELS.get(objective["monster_type"], "Obj")
            draw.text((ox, oz - sr - 2), obj_name,
                      fill=(255, 255, 255, 240), anchor="mb")

        # Timestamp watermark (bottom-left)
        draw.text((4, H - 4), _ms_to_mmss(ts),
                  fill=(255, 255, 255, 220), anchor="lb")

        pil_frames.append(
            frame.convert("RGB").quantize(colors=256, dither=PILImage.Dither.NONE)
        )

    out = io.BytesIO()
    duration_ms = max(80, 1000 // fps)
    pil_frames[0].save(
        out, format="GIF", append_images=pil_frames[1:],
        save_all=True, duration=duration_ms, loop=0, optimize=False,
    )
    return out.getvalue()


# ── Game metadata for selector ───────────────────────────────────────────────
_games_meta = (
    picks_raw[["series_id", "game_num", "blue_team", "red_team", "date"]]
    .drop_duplicates(subset=["series_id", "game_num"])
    .sort_values(["date", "series_id", "game_num"])
)

with tab_moves:
    st.subheader("Champion Movements")
    st.caption(
        "Select a game and a time window (or an objective event) to generate an "
        "animated GIF of champion positions overlaid on the minimap."
    )

    # ── Step 1 – pick a game ────────────────────────────────────────────────
    st.markdown("#### 1 · Select a game")
    game_options: list[str] = []
    game_key_map: dict[str, tuple[str, int]] = {}
    for _, gr in _games_meta.iterrows():
        label = (
            f"{gr['blue_team']} vs {gr['red_team']}  —  "
            f"Game {gr['game_num']}  ({str(gr['date'])[:10]})"
        )
        game_options.append(label)
        game_key_map[label] = (str(gr["series_id"]), int(gr["game_num"]))

    sel_game_label = st.selectbox("Game", options=game_options, key="move_game")
    sel_sid, sel_gnum = game_key_map[sel_game_label]

    # ── Load positions (cached) ─────────────────────────────────────────────
    pos_df, objectives, pid_info = load_game_positions(sel_sid, sel_gnum)

    if pos_df.empty:
        st.error("Could not load position data for this game from S3.")
        st.stop()

    game_dur_s   = int(pos_df["game_time_ms"].max() / 1000)
    game_dur_min = game_dur_s // 60 + 1

    st.success(
        f"Loaded **{len(pos_df):,}** position snapshots over "
        f"**{game_dur_s // 60}:{game_dur_s % 60:02d}** of game time."
    )

    # ── Step 2 – time selection ──────────────────────────────────────────────
    st.markdown("#### 2 · Choose a time window")

    time_mode = st.radio(
        "Mode",
        options=["Manual range", "Around objective"],
        horizontal=True,
        key="move_time_mode",
    )

    if time_mode == "Manual range":
        t_range = st.slider(
            "Game time window (minutes)",
            min_value=0,
            max_value=game_dur_min,
            value=(0, min(5, game_dur_min)),
            key="move_trange",
        )
        start_ms = t_range[0] * 60_000
        end_ms   = t_range[1] * 60_000
        active_objective = None

    else:  # Around objective
        if not objectives:
            st.warning("No baron / dragon / rift herald kills found in this game.")
            start_ms = 0
            end_ms   = min(300_000, pos_df["game_time_ms"].max())
            active_objective = None
        else:
            obj_labels = [o["label"] for o in objectives]
            sel_obj_label = st.selectbox("Objective", options=obj_labels, key="move_obj")
            active_objective = next(o for o in objectives if o["label"] == sel_obj_label)

            window_s = st.slider(
                "Window around objective (seconds before / after)",
                min_value=15, max_value=120, value=60, step=15,
                key="move_obj_window",
            )
            start_ms = max(0, active_objective["game_time_ms"] - window_s * 1000)
            end_ms   = min(
                pos_df["game_time_ms"].max(),
                active_objective["game_time_ms"] + window_s * 1000,
            )

    clip_s = (end_ms - start_ms) // 1000
    st.caption(
        f"Window: **{_ms_to_mmss(start_ms)}** → **{_ms_to_mmss(end_ms)}** "
        f"({clip_s}s)"
    )

    # ── Step 3 – champion filter ─────────────────────────────────────────────
    st.markdown("#### 3 · Filter champions (optional)")

    all_champs_in_game = sorted(
        {info["champion"] for info in pid_info.values() if info.get("champion")}
    )
    sel_champs = st.multiselect(
        "Highlight specific champions (leave empty = show all equally)",
        options=all_champs_in_game,
        default=[],
        key="move_champs",
    )
    highlight_pids: set | None = None
    if sel_champs:
        highlight_pids = {
            pid for pid, info in pid_info.items()
            if info.get("champion") in sel_champs
        }

    # ── Step 4 – animation settings ─────────────────────────────────────────
    with st.expander("Animation settings", expanded=False):
        ac1, ac2 = st.columns(2)
        sample_s = ac1.slider(
            "Sample every N seconds",
            min_value=1, max_value=10, value=3, step=1,
            key="move_sample",
        )
        fps = ac2.slider(
            "Playback speed (fps)",
            min_value=1, max_value=8, value=4, step=1,
            key="move_fps",
        )

    est_frames = max(1, clip_s // sample_s)
    st.caption(
        f"Estimated frames: **{est_frames}** · "
        f"GIF duration: ~**{est_frames / fps:.1f}s** at {fps} fps"
    )

    # ── Generate ─────────────────────────────────────────────────────────────
    if st.button("▶ Generate GIF", type="primary", key="move_gen"):
        with st.spinner(f"Rendering {est_frames} frames…"):
            gif_bytes = _build_gif(
                pos_df, start_ms, end_ms,
                sample_s=sample_s, fps=fps,
                highlight_pids=highlight_pids,
                objective=active_objective,
            )

        if gif_bytes is None:
            st.warning("No position data found in the selected window.")
        else:
            st.image(gif_bytes, caption=sel_game_label, use_container_width=False)
            st.download_button(
                "⬇️ Download GIF",
                data=gif_bytes,
                file_name=f"movements_{sel_sid}_g{sel_gnum}.gif",
                mime="image/gif",
                key="move_dl",
            )

    # ── Participant reference ─────────────────────────────────────────────────
    if pid_info:
        with st.expander("Participant reference", expanded=False):
            ref_rows = [
                {
                    "Side":     info["side"],
                    "Role":     info["role"],
                    "Player":   info["player"],
                    "Champion": info["champion"],
                }
                for info in sorted(pid_info.values(),
                                   key=lambda x: (0 if x["side"] == "Blue" else 1,
                                                  x["role"]))
            ]
            st.dataframe(pd.DataFrame(ref_rows), hide_index=True,
                         use_container_width=True)
