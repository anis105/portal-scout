"""
Fetch current player stats from ESPN for all teams in espn_team_ids.json.
Writes data/current_players.csv.

Usage:
    python3 scripts/refresh_data.py
    python3 scripts/refresh_data.py --conferences "Big Ten,SEC,Big 12,ACC,Big East"
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
import urllib.error

ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams/{team_id}/roster"
)
STATS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/basketball"
    "/mens-college-basketball/athletes/{athlete_id}/overview"
)

# ESPN stat labels in order: GP, MIN, FG%, 3P%, FT%, REB, AST, BLK, STL, PF, TO, PTS
STAT_LABELS = ["gp", "min", "fg_pct", "three_pct", "ft_pct",
               "reb", "ast", "blk", "stl", "pf", "to", "pts"]

EXPERIENCE_MAP = {
    "Freshman": "Fr", "Sophomore": "So", "Junior": "Jr", "Senior": "Sr",
    "5th Year": "5th", "Graduate": "Grad", "Redshirt Freshman": "R-Fr",
    "Redshirt Sophomore": "R-So",
}

CSV_COLUMNS = [
    "name", "school", "conference", "position", "height", "weight",
    "pts", "reb", "ast", "stl", "blk", "to", "min", "gp",
    "fg_pct", "three_pct", "ft_pct", "year_label",
]


def fetch_json(url, retries=2):
    """Fetch URL with retries and rate limiting."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(1)
            else:
                return None


def inches_to_height(inches):
    """Convert 77 -> '6-5'."""
    try:
        inches = int(float(inches))
        return str(inches // 12) + "-" + str(inches % 12)
    except (ValueError, TypeError):
        return "6-6"


def refine_position(espn_pos, height_inches):
    """Refine ESPN G/F/C to PG/SG/SF/PF/C."""
    try:
        h = int(float(height_inches))
    except (ValueError, TypeError):
        h = 78

    if espn_pos == "G":
        return "PG" if h < 75 else "SG"
    elif espn_pos == "F":
        return "SF" if h < 79 else "PF"
    elif espn_pos == "C":
        return "C"
    else:
        # Unknown position, guess from height
        if h < 75:
            return "PG"
        elif h < 78:
            return "SG"
        elif h < 80:
            return "SF"
        elif h < 82:
            return "PF"
        return "C"


def parse_stat(val):
    """Parse a stat string to float."""
    try:
        return round(float(val), 1)
    except (ValueError, TypeError):
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conferences", type=str, default=None,
                        help='Comma-separated list of conferences to fetch, e.g. "Big Ten,SEC"')
    args = parser.parse_args()

    with open("data/espn_team_ids.json") as f:
        team_ids = json.load(f)

    # Filter by conferences if specified
    if args.conferences:
        target_confs = set(c.strip() for c in args.conferences.split(","))
        team_ids = {k: v for k, v in team_ids.items() if v["conf"] in target_confs}
        print(f"Filtering to {len(team_ids)} teams in: {', '.join(sorted(target_confs))}")

    total_teams = len(team_ids)
    players = []
    skipped_no_stats = 0
    skipped_filter = 0

    for i, (school, info) in enumerate(sorted(team_ids.items())):
        team_id = info["id"]
        conf = info["conf"]
        sys.stdout.write(f"[{i+1}/{total_teams}] {school} ({conf})...")
        sys.stdout.flush()

        roster_data = fetch_json(ROSTER_URL.format(team_id=team_id))
        if not roster_data:
            print(" roster failed")
            continue

        athletes = roster_data.get("athletes", [])
        team_count = 0

        for athlete in athletes:
            athlete_id = athlete.get("id")
            name = athlete.get("fullName", "").strip()
            if not athlete_id or not name:
                continue

            espn_pos = athlete.get("position", {}).get("abbreviation", "G")
            height_raw = athlete.get("height", 78)
            weight = athlete.get("weight", 200)
            exp = athlete.get("experience", {}).get("displayValue", "Freshman")

            height_str = inches_to_height(height_raw)
            position = refine_position(espn_pos, height_raw)
            year_label = EXPERIENCE_MAP.get(exp, exp)

            # Fetch individual stats
            stats_data = fetch_json(STATS_URL.format(athlete_id=athlete_id))
            if not stats_data:
                skipped_no_stats += 1
                continue

            stat_info = stats_data.get("statistics", {})
            splits = stat_info.get("splits", [])
            if not splits or not splits[0].get("stats"):
                skipped_no_stats += 1
                continue

            raw_stats = splits[0]["stats"]
            if len(raw_stats) < 12:
                skipped_no_stats += 1
                continue

            stats = {}
            for j, label in enumerate(STAT_LABELS):
                stats[label] = parse_stat(raw_stats[j])

            # Filter: GP >= 5 and MIN >= 8
            if stats["gp"] < 5 or stats["min"] < 8:
                skipped_filter += 1
                continue

            # Convert percentages from 0-100 to 0-1 scale
            fg_pct = round(stats["fg_pct"] / 100.0, 3) if stats["fg_pct"] > 1 else stats["fg_pct"]
            three_pct = round(stats["three_pct"] / 100.0, 3) if stats["three_pct"] > 1 else stats["three_pct"]
            ft_pct = round(stats["ft_pct"] / 100.0, 3) if stats["ft_pct"] > 1 else stats["ft_pct"]

            players.append({
                "name": name,
                "school": school,
                "conference": conf,
                "position": position,
                "height": height_str,
                "weight": int(float(weight)) if weight else 200,
                "pts": stats["pts"],
                "reb": stats["reb"],
                "ast": stats["ast"],
                "stl": stats["stl"],
                "blk": stats["blk"],
                "to": stats["to"],
                "min": stats["min"],
                "gp": int(stats["gp"]),
                "fg_pct": fg_pct,
                "three_pct": three_pct,
                "ft_pct": ft_pct,
                "year_label": year_label,
            })
            team_count += 1

        print(f" {team_count} players")
        # Small delay between teams to be polite to ESPN
        time.sleep(0.2)

    # Write CSV
    out_path = "data/current_players.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for p in sorted(players, key=lambda x: (x["school"], x["name"])):
            writer.writerow(p)

    print(f"\nDone: {len(players)} players -> {out_path}")
    print(f"Skipped: {skipped_no_stats} no stats, {skipped_filter} below threshold")


if __name__ == "__main__":
    main()
