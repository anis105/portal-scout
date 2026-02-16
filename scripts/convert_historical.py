"""
Convert transfer_database.csv to the new transfers_historical.csv schema.
Fills missing advanced stats with position-based heuristics.

Usage:
    python3 scripts/convert_historical.py
"""

import csv
import sys

sys.path.insert(0, ".")

# Position-based defaults for missing stats (based on D1 averages by position)
# Format: (stl, blk, to, min, fg_pct, three_pct, ft_pct)
POS_DEFAULTS = {
    "PG": (1.2, 0.2, 2.5, 28.0, 0.42, 0.34, 0.76),
    "SG": (1.0, 0.3, 1.8, 27.0, 0.43, 0.35, 0.74),
    "SF": (0.9, 0.4, 1.5, 26.0, 0.44, 0.33, 0.72),
    "PF": (0.7, 0.8, 1.3, 25.0, 0.48, 0.30, 0.68),
    "C":  (0.5, 1.2, 1.5, 24.0, 0.52, 0.25, 0.65),
}


def last_season(row):
    """Find the most recent non-zero season stats."""
    for yr in ["senior", "junior", "sophomore", "freshman"]:
        pts = float(row[yr + "_pts"])
        if pts > 0:
            return {
                "pts": pts,
                "reb": float(row[yr + "_reb"]),
                "ast": float(row[yr + "_ast"]),
                "year": yr,
            }
    return {"pts": 0, "reb": 0, "ast": 0, "year": "freshman"}


def estimate_stats(pos, pts, reb, ast):
    """Estimate missing per-game stats based on position and known production."""
    defaults = POS_DEFAULTS.get(pos, POS_DEFAULTS["SF"])
    base_stl, base_blk, base_to, base_min, base_fg, base_3p, base_ft = defaults

    # Scale minutes based on scoring volume (more points = more minutes)
    min_scale = min(pts / 10.0, 1.5) if pts > 0 else 0.5
    est_min = round(base_min * min_scale, 1)
    est_min = max(12.0, min(est_min, 38.0))

    # Scale steals/blocks/turnovers with minutes
    min_ratio = est_min / base_min
    est_stl = round(base_stl * min_ratio, 1)
    est_blk = round(base_blk * min_ratio, 1)
    est_to = round(base_to * min_ratio, 1)

    # Guards with high assists have higher TO
    if pos in ("PG", "SG") and ast >= 4.0:
        est_to = round(est_to * 1.2, 1)

    # Adjust FG% based on scoring volume and position
    # High volume scorers tend to have slightly lower FG%
    fg_adj = 0.0
    if pts > 15:
        fg_adj = -0.02
    elif pts > 20:
        fg_adj = -0.04
    elif pts < 5:
        fg_adj = 0.02

    est_fg = round(base_fg + fg_adj, 3)
    est_3p = round(base_3p, 3)
    est_ft = round(base_ft, 3)

    # Big men who score a lot likely have higher FG%
    if pos in ("PF", "C") and pts > 10 and reb > 5:
        est_fg = round(est_fg + 0.03, 3)

    # GP estimate (full season = ~30 games for starters)
    est_gp = 28 if est_min > 20 else 25

    return {
        "stl": est_stl,
        "blk": est_blk,
        "to": est_to,
        "min": est_min,
        "gp": est_gp,
        "fg_pct": est_fg,
        "three_pct": est_3p,
        "ft_pct": est_ft,
    }


def main():
    rows = []
    with open("data/transfer_database.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from transfer_database.csv")

    # Filter to actual transfers only (for training set)
    # Keep ALL rows (transfers + stayed + baseline) since bootstrap needs them all
    # But only actual transfers will have post-transfer stats

    output = []
    for row in rows:
        name = row["name"].strip()
        from_school = row["from_school"].strip()
        to_school = row["to_school"].strip()
        pos = row["position"].strip()

        season = last_season(row)
        pts, reb, ast = season["pts"], season["reb"], season["ast"]

        # Estimate missing stats
        est = estimate_stats(pos, pts, reb, ast)

        is_actual_transfer = (
            to_school != from_school
            and to_school != "NA"
            and to_school != ""
        )

        # Post-transfer stats: only available for actual transfers with known outcomes
        # For now, use pre-transfer stats as rough proxy (conference adjustment happens in predict())
        # TODO: fill with real post-transfer data when available
        post_pts = 0.0
        post_reb = 0.0
        post_ast = 0.0

        if is_actual_transfer:
            outcome = row.get("career_outcome", "unknown")
            # Rough post-transfer estimates based on outcome
            if outcome in ("starter", "starter_injured"):
                post_pts = round(pts * 0.95, 1)
                post_reb = round(reb * 0.95, 1)
                post_ast = round(ast * 0.95, 1)
            elif outcome == "rotation":
                post_pts = round(pts * 0.75, 1)
                post_reb = round(reb * 0.80, 1)
                post_ast = round(ast * 0.75, 1)
            elif outcome == "bench":
                post_pts = round(pts * 0.45, 1)
                post_reb = round(reb * 0.50, 1)
                post_ast = round(ast * 0.45, 1)
            elif "lottery" in outcome or "draft_first" in outcome:
                post_pts = round(pts * 1.15, 1)
                post_reb = round(reb * 1.10, 1)
                post_ast = round(ast * 1.10, 1)
            elif "draft_second" in outcome:
                post_pts = round(pts * 1.0, 1)
                post_reb = round(reb * 1.0, 1)
                post_ast = round(ast * 1.0, 1)
            elif outcome == "transferred_again":
                post_pts = round(pts * 0.60, 1)
                post_reb = round(reb * 0.65, 1)
                post_ast = round(ast * 0.60, 1)
            elif outcome == "grad_transfer":
                post_pts = round(pts * 0.85, 1)
                post_reb = round(reb * 0.85, 1)
                post_ast = round(ast * 0.85, 1)
            else:
                post_pts = round(pts * 0.80, 1)
                post_reb = round(reb * 0.80, 1)
                post_ast = round(ast * 0.80, 1)

        output.append({
            "name": name,
            "from_school": from_school,
            "from_conf": row["from_conf"].strip(),
            "to_school": to_school,
            "to_conf": row["to_conf"].strip(),
            "position": pos,
            "height": row["height"].strip(),
            "weight": int(row["weight"]),
            "transfer_year": int(row["transfer_year"]),
            "pre_pts": pts,
            "pre_reb": reb,
            "pre_ast": ast,
            "pre_stl": est["stl"],
            "pre_blk": est["blk"],
            "pre_to": est["to"],
            "pre_min": est["min"],
            "pre_gp": est["gp"],
            "pre_fg_pct": est["fg_pct"],
            "pre_three_pct": est["three_pct"],
            "pre_ft_pct": est["ft_pct"],
            "post_pts": post_pts,
            "post_reb": post_reb,
            "post_ast": post_ast,
            "career_outcome": row["career_outcome"].strip(),
        })

    # Write new CSV
    out_path = "data/transfers_historical.csv"
    fieldnames = [
        "name", "from_school", "from_conf", "to_school", "to_conf",
        "position", "height", "weight", "transfer_year",
        "pre_pts", "pre_reb", "pre_ast", "pre_stl", "pre_blk", "pre_to",
        "pre_min", "pre_gp", "pre_fg_pct", "pre_three_pct", "pre_ft_pct",
        "post_pts", "post_reb", "post_ast",
        "career_outcome",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output:
            writer.writerow(row)

    actual_transfers = [r for r in output if r["to_school"] != r["from_school"] and r["to_school"] != "NA"]
    print(f"Wrote {len(output)} rows ({len(actual_transfers)} actual transfers) -> {out_path}")


if __name__ == "__main__":
    main()
