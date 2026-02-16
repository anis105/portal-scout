"""
Transfer Portal Evaluator — Maryland Basketball
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from data.conferences import SCHOOL_TO_CONFERENCE, ALL_SCHOOLS, ALL_CONFERENCES
import hashlib

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Transfer Portal Evaluator",
    page_icon="M",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
MARYLAND_RED = "#E03A3E"
MARYLAND_GOLD = "#FFD520"
DARK = "#1A1A1A"
TEXT = "#333333"
MUTED = "#888888"
BG = "#FAFAFA"

css = (
    "<style>"
    ".stApp { background-color: " + BG + "; }"
    "section[data-testid='stSidebar'] { display: none; }"
    "#MainMenu { visibility: hidden; }"
    "header { visibility: hidden; }"
    "footer { visibility: hidden; }"
    ".block-container { padding-top: 2rem; padding-bottom: 1rem; max-width: 900px; }"

    ".header-title {"
    "  font-size: 1.6rem; font-weight: 700; color: " + DARK + ";"
    "  letter-spacing: -0.02em; margin: 0; line-height: 1.2;"
    "}"
    ".header-sub {"
    "  font-size: 0.85rem; color: " + MUTED + ";"
    "  margin: 0.25rem 0 1.5rem 0; font-weight: 400;"
    "}"
    ".section-label {"
    "  font-size: 0.7rem; font-weight: 600; color: " + MUTED + ";"
    "  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;"
    "}"

    "/* --- Avatar --- */"
    ".avatar-card {"
    "  display: flex; align-items: center; gap: 1.2rem;"
    "  margin: 1rem 0 1.5rem 0; padding: 1.2rem 1.5rem;"
    "  background: white; border-radius: 10px; border: 1px solid #E8E8E8;"
    "}"
    ".avatar-circle {"
    "  width: 64px; height: 64px; border-radius: 50%;"
    "  display: flex; align-items: center; justify-content: center;"
    "  font-size: 1.5rem; font-weight: 700; color: white;"
    "  flex-shrink: 0;"
    "}"
    ".avatar-info { display: flex; flex-direction: column; }"
    ".avatar-name {"
    "  font-size: 1.15rem; font-weight: 700; color: " + DARK + ";"
    "  line-height: 1.2;"
    "}"
    ".avatar-detail {"
    "  font-size: 0.78rem; color: " + MUTED + "; margin-top: 0.2rem;"
    "}"
    ".avatar-transfer {"
    "  font-size: 0.78rem; color: " + TEXT + "; margin-top: 0.3rem;"
    "  font-weight: 500;"
    "}"

    "/* --- Stat numbers --- */"
    ".stat-row {"
    "  display: flex; gap: 2.5rem; margin: 1rem 0 1.5rem 0;"
    "}"
    ".stat-block { display: flex; flex-direction: column; }"
    ".stat-num {"
    "  font-size: 2.4rem; font-weight: 700; color: " + DARK + ";"
    "  line-height: 1; font-variant-numeric: tabular-nums;"
    "}"
    ".stat-num-accent {"
    "  font-size: 2.4rem; font-weight: 700; color: " + MARYLAND_RED + ";"
    "  line-height: 1; font-variant-numeric: tabular-nums;"
    "}"
    ".stat-label {"
    "  font-size: 0.7rem; color: " + MUTED + ";"
    "  text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem;"
    "}"
    ".stat-delta { font-size: 0.78rem; margin-top: 0.15rem; }"
    ".stat-delta-up { color: #2E8B57; }"
    ".stat-delta-down { color: " + MARYLAND_RED + "; }"

    "/* --- Verdict --- */"
    ".verdict {"
    "  display: inline-block; padding: 0.4rem 1rem;"
    "  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;"
    "  text-transform: uppercase; border-radius: 3px;"
    "}"
    ".verdict-high { background: " + MARYLAND_RED + "; color: white; }"
    ".verdict-mid { background: " + MARYLAND_GOLD + "; color: " + DARK + "; }"
    ".verdict-low { background: #E0E0E0; color: " + TEXT + "; }"

    "/* --- Table --- */"
    ".comp-table {"
    "  width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 0.5rem 0;"
    "}"
    ".comp-table th {"
    "  text-align: left; font-weight: 600; color: " + MUTED + ";"
    "  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;"
    "  padding: 0.5rem 0.75rem; border-bottom: 2px solid #E0E0E0;"
    "}"
    ".comp-table td {"
    "  padding: 0.6rem 0.75rem; color: " + TEXT + ";"
    "  border-bottom: 1px solid #F0F0F0;"
    "}"
    ".sim-bar {"
    "  display: inline-block; height: 4px; background: " + MARYLAND_RED + ";"
    "  border-radius: 2px; vertical-align: middle; margin-right: 0.4rem;"
    "}"

    ".commentary {"
    "  margin: 1rem 0 1.5rem 0; padding: 1rem 1.2rem;"
    "  background: white; border-left: 3px solid " + MARYLAND_RED + ";"
    "  border-radius: 0 6px 6px 0; font-size: 0.85rem; line-height: 1.6;"
    "  color: " + TEXT + ";"
    "}"

    ".footer {"
    "  margin-top: 3rem; padding-top: 1.5rem;"
    "  border-top: 1px solid #E0E0E0; font-size: 0.72rem; color: #AAAAAA;"
    "}"

    ".stSelectbox label, .stTextInput label, .stNumberInput label {"
    "  font-size: 0.72rem !important; color: " + MUTED + " !important;"
    "  text-transform: uppercase; letter-spacing: 0.05em;"
    "}"
    "</style>"
)
st.markdown(css, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Conference data
# ---------------------------------------------------------------------------
CONFERENCE_STRENGTH = {
    "Big Ten": 15.2, "SEC": 14.8, "Big 12": 14.5, "ACC": 13.2,
    "Big East": 11.8, "Pac-12": 10.0, "Mountain West": 8.5, "American": 8.0,
    "WCC": 7.5, "A-10": 7.0, "Missouri Valley": 6.5, "C-USA": 5.5,
    "CAA": 5.5, "Sun Belt": 5.0, "Ivy League": 5.0, "Horizon": 4.8,
    "WAC": 4.5, "ASUN": 4.5, "Southern": 4.5, "Big Sky": 4.2,
    "Big West": 4.0, "Patriot": 4.0, "MAAC": 3.8, "Summit": 3.8,
    "NEC": 3.5, "Big South": 3.5, "Southland": 3.2, "OVC": 3.2,
    "America East": 3.5, "MEAC": 2.5, "SWAC": 2.5,
}

POSITION_MAP = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
POSITIONS = ["PG", "SG", "SF", "PF", "C"]
CONF_KEYS = list(CONFERENCE_STRENGTH.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _avatar_color(name):
    """Deterministic color from player name."""
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
    colors = ["#E03A3E", "#2C5F8A", "#2E8B57", "#8B5E3C", "#6A5ACD", "#C0392B", "#1A7A4C", "#34495E"]
    return colors[h % len(colors)]


def _avatar_initials(name):
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()


def _fmt(val):
    """Format stat: show integer if .0, else one decimal."""
    if val == int(val):
        return str(int(val))
    return str(round(val, 1))


@st.cache_data
def load_transfer_data():
    return pd.read_csv("data/transfers_historical.csv")

@st.cache_data
def load_current_players():
    return pd.read_csv("data/current_players.csv")


def parse_height(h):
    try:
        if pd.isna(h):
            return 78
        parts = str(h).split("-")
        if len(parts) == 2:
            return int(parts[0]) * 12 + int(parts[1])
    except Exception:
        pass
    return 78


# ---------------------------------------------------------------------------
# Smoothed bootstrap engine
# ---------------------------------------------------------------------------
@st.cache_data
def bootstrap_training_set(df_real, n_per_sample=10, h=0.15, seed=42):
    """Expand real transfers to synthetic training set via smoothed bootstrap.

    Adds Gaussian noise (std = h * feature_std) to numeric columns,
    keeping categorical columns (position, school, conference, outcome) unchanged.
    """
    rng = np.random.default_rng(seed)

    numeric_cols = [
        "pre_pts", "pre_reb", "pre_ast", "pre_stl", "pre_blk", "pre_to",
        "pre_min", "pre_fg_pct", "pre_three_pct", "pre_ft_pct",
        "post_pts", "post_reb", "post_ast",
    ]
    pct_cols = {"pre_fg_pct", "pre_three_pct", "pre_ft_pct"}
    nonneg_cols = set(numeric_cols) - pct_cols

    stds = {}
    for col in numeric_cols:
        if col in df_real.columns:
            stds[col] = df_real[col].std()
            if stds[col] == 0 or pd.isna(stds[col]):
                stds[col] = 1.0

    synthetic_rows = []
    for _ in range(n_per_sample):
        for _, row in df_real.iterrows():
            new_row = row.copy()
            for col in numeric_cols:
                if col in new_row.index:
                    noise = rng.normal(0, h * stds[col])
                    new_row[col] = row[col] + noise
            for col in pct_cols:
                if col in new_row.index:
                    new_row[col] = np.clip(new_row[col], 0.0, 1.0)
            for col in nonneg_cols:
                if col in new_row.index:
                    new_row[col] = max(0.0, new_row[col])
            synthetic_rows.append(new_row)

    df_synth = pd.DataFrame(synthetic_rows)
    return pd.concat([df_real, df_synth], ignore_index=True)


# ---------------------------------------------------------------------------
# Basketball-weighted k-NN (Dean Oliver Four Factors)
# ---------------------------------------------------------------------------
# Features: 15 total
# Shooting Efficiency (40%): fg_pct, three_pct, ft_pct, pts_per_min
# Ball Security (25%): ast_to_ratio, ast, to
# Rebounding (20%): reb, reb_per_min
# Defense (15%): stl, blk, min
# Physical/Context: pos_code, height_in, from_str
FEATURE_COLS = [
    "fg_pct", "three_pct", "ft_pct", "pts_per_min",
    "ast_to_ratio", "ast", "to",
    "reb", "reb_per_min",
    "stl", "blk", "min",
    "pos_code", "height_in", "from_str",
]

BASE_WEIGHTS = np.array([
    1.8, 1.4, 1.0, 1.6,
    1.8, 1.2, 1.0,
    1.4, 1.2,
    1.0, 1.0, 1.1,
    0.8, 0.6, 1.2,
])

POS_MULTIPLIERS = {
    "PG": {"ast_to_ratio": 1.5, "ast": 1.4, "three_pct": 1.2,
           "stl": 1.3, "blk": 0.3, "reb": 0.7, "reb_per_min": 0.6},
    "SG": {"three_pct": 1.4, "pts_per_min": 1.3, "stl": 1.2,
           "ast_to_ratio": 1.2, "blk": 0.4, "reb": 0.8},
    "SF": {"three_pct": 1.2, "stl": 1.1, "blk": 0.7},
    "PF": {"reb": 1.3, "reb_per_min": 1.3, "blk": 1.2, "fg_pct": 1.2,
           "three_pct": 1.1, "ast_to_ratio": 0.8, "stl": 0.8},
    "C":  {"reb": 1.5, "reb_per_min": 1.5, "blk": 1.6, "fg_pct": 1.3,
           "ft_pct": 0.9, "three_pct": 0.8, "ast_to_ratio": 0.5,
           "ast": 0.5, "stl": 0.6},
}


def _get_weights(position):
    """Get final feature weights: base * position multipliers."""
    mults = POS_MULTIPLIERS.get(position, {})
    w = BASE_WEIGHTS.copy()
    for i, col in enumerate(FEATURE_COLS):
        if col in mults:
            w[i] *= mults[col]
    return w


def _compute_derived(row):
    """Compute derived features from raw stats."""
    pts = float(row.get("pre_pts", row.get("pts", 0)) or 0)
    reb = float(row.get("pre_reb", row.get("reb", 0)) or 0)
    ast = float(row.get("pre_ast", row.get("ast", 0)) or 0)
    stl = float(row.get("pre_stl", row.get("stl", 0)) or 0)
    blk = float(row.get("pre_blk", row.get("blk", 0)) or 0)
    to = float(row.get("pre_to", row.get("to", 0)) or 0)
    mpg = float(row.get("pre_min", row.get("min", 20)) or 20)
    fg_pct = float(row.get("pre_fg_pct", row.get("fg_pct", 0.42)) or 0.42)
    three_pct = float(row.get("pre_three_pct", row.get("three_pct", 0.33)) or 0.33)
    ft_pct = float(row.get("pre_ft_pct", row.get("ft_pct", 0.70)) or 0.70)

    pts_per_min = pts / mpg if mpg > 0 else 0.0
    reb_per_min = reb / mpg if mpg > 0 else 0.0
    ast_to_ratio = ast / max(to, 0.5)

    return {
        "fg_pct": fg_pct, "three_pct": three_pct, "ft_pct": ft_pct,
        "pts_per_min": round(pts_per_min, 3),
        "ast_to_ratio": round(ast_to_ratio, 2),
        "ast": ast, "to": to,
        "reb": reb, "reb_per_min": round(reb_per_min, 3),
        "stl": stl, "blk": blk, "min": mpg,
    }


def prepare_features(df):
    """Prepare feature matrix from transfer training data."""
    out = df.copy()
    out["height_in"] = out["height"].apply(parse_height)
    from_conf_col = "from_conf"
    out["from_str"] = out[from_conf_col].map(lambda x: CONFERENCE_STRENGTH.get(x, 5.0))
    out["pos_code"] = out["position"].map(POSITION_MAP).fillna(3)

    derived = out.apply(lambda r: _compute_derived(r), axis=1)
    for col in ["fg_pct", "three_pct", "ft_pct", "pts_per_min",
                "ast_to_ratio", "ast", "to", "reb", "reb_per_min",
                "stl", "blk", "min"]:
        out[col] = [d[col] for d in derived]

    return out


def find_similar(player, df, k=5):
    """Find k most similar historical transfers using basketball-weighted k-NN."""
    transfers = df[
        (df["to_school"] != df["from_school"])
        & (df["to_school"] != "NA")
        & (df["to_school"].notna())
    ].copy().reset_index(drop=True)
    if len(transfers) == 0:
        return []

    feat_df = prepare_features(transfers)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df[FEATURE_COLS])

    # Build input vector from player dict
    player_derived = _compute_derived(player)
    inp_vals = [
        player_derived["fg_pct"], player_derived["three_pct"],
        player_derived["ft_pct"], player_derived["pts_per_min"],
        player_derived["ast_to_ratio"], player_derived["ast"],
        player_derived["to"],
        player_derived["reb"], player_derived["reb_per_min"],
        player_derived["stl"], player_derived["blk"], player_derived["min"],
        POSITION_MAP.get(player.get("position", "SF"), 3),
        parse_height(player.get("height", "6-6")),
        CONFERENCE_STRENGTH.get(player.get("from_conf", "Big Ten"), 5.0),
    ]
    inp = scaler.transform(np.array([inp_vals]))

    # Apply basketball-weighted distance
    pos = player.get("position", "SF")
    weights = _get_weights(pos)
    X_weighted = X * weights
    inp_weighted = inp * weights

    # Fetch more neighbors than needed to allow de-duplication of bootstrap variants
    n_fetch = min(k * 8, len(transfers))
    knn = NearestNeighbors(n_neighbors=n_fetch, metric="euclidean")
    knn.fit(X_weighted)
    dists, idxs = knn.kneighbors(inp_weighted)
    sims = 1.0 / (1.0 + dists[0])

    # De-duplicate: keep best match per unique real player
    results = []
    seen_names = set()
    for idx, sim in zip(idxs[0], sims):
        r = feat_df.iloc[idx]
        player_key = r["name"] + "|" + r["from_school"] + "|" + r["to_school"]
        if player_key in seen_names:
            continue
        seen_names.add(player_key)
        results.append({
            "name": r["name"], "from_school": r["from_school"],
            "from_conf": r["from_conf"], "to_school": r["to_school"],
            "to_conf": r["to_conf"], "position": r["position"],
            "pts": float(r.get("pre_pts", 0)), "reb": float(r.get("pre_reb", 0)),
            "ast": float(r.get("pre_ast", 0)),
            "post_pts": float(r.get("post_pts", 0)),
            "post_reb": float(r.get("post_reb", 0)),
            "post_ast": float(r.get("post_ast", 0)),
            "sim": sim, "outcome": r["career_outcome"],
        })
        if len(results) >= k:
            break
    return results


def predict(player, cases, target_conf):
    """Hybrid prediction: blend player baseline (conference-adjusted) with comparable outcomes."""
    if not cases:
        return None

    from_str = CONFERENCE_STRENGTH.get(player.get("from_conf", "Big Ten"), 10)
    to_str = CONFERENCE_STRENGTH.get(target_conf, 10)
    ratio = from_str / to_str if to_str else 1.0
    adj = max(0.5, min(ratio ** 0.35, 1.5))

    pts = float(player.get("pre_pts", player.get("pts", 10)))
    reb = float(player.get("pre_reb", player.get("reb", 3)))
    ast = float(player.get("pre_ast", player.get("ast", 2)))

    baseline_pts = pts * adj
    baseline_reb = reb * adj
    baseline_ast = ast * adj

    # Comparable outcomes (similarity-weighted average of post-transfer stats)
    valid = [c for c in cases if c.get("post_pts", 0) > 0]
    if valid:
        total_w = sum(c["sim"] for c in valid)
        comp_pts = sum(c["post_pts"] * c["sim"] for c in valid) / total_w
        comp_reb = sum(c["post_reb"] * c["sim"] for c in valid) / total_w
        comp_ast = sum(c["post_ast"] * c["sim"] for c in valid) / total_w
    else:
        comp_pts, comp_reb, comp_ast = baseline_pts, baseline_reb, baseline_ast

    # Position-specific blend: bigs trust baseline more, guards trust comps more
    blend_w = {"PG": 0.40, "SG": 0.45, "SF": 0.50, "PF": 0.55, "C": 0.60}
    pos = player.get("position", "SF")
    bw = blend_w.get(pos, 0.50)

    pred_pts = round(baseline_pts * bw + comp_pts * (1 - bw), 1)
    pred_reb = round(baseline_reb * bw + comp_reb * (1 - bw), 1)
    pred_ast = round(baseline_ast * bw + comp_ast * (1 - bw), 1)

    pts_vals = [c["pts"] for c in cases]
    pts_std = np.std(pts_vals) if len(pts_vals) > 1 else 2.0

    return {
        "pts": pred_pts, "reb": pred_reb, "ast": pred_ast,
        "low": round(pred_pts - pts_std, 1),
        "high": round(pred_pts + pts_std, 1),
        "adj": round(adj, 2), "n": len(cases),
    }

# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------
def make_radar(player, cases, pred, target_name="Maryland"):
    cats = ["PTS", "REB", "AST"]
    def norm(pts, reb, ast):
        return [min(pts / 22.0 * 10, 10), min(reb / 11.0 * 10, 10), min(ast / 7.0 * 10, 10)]

    # Raw values for hover display
    cur_raw = [player.get("pts", 0), player.get("reb", 0), player.get("ast", 0)]
    proj_raw = [pred["pts"], pred["reb"], pred["ast"]]
    avg_raw = [round(np.mean([c["pts"] for c in cases]), 1),
               round(np.mean([c["reb"] for c in cases]), 1),
               round(np.mean([c["ast"] for c in cases]), 1)]

    cur = norm(*cur_raw)
    proj = norm(*proj_raw)
    avg = norm(*avg_raw)

    def hover_texts(raw_vals, label):
        return [label + ": " + _fmt(v) for v in raw_vals] + [label + ": " + _fmt(raw_vals[0])]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg + [avg[0]], theta=cats + [cats[0]], fill="toself",
        name="Comparable Avg",
        text=hover_texts(avg_raw, "Avg"),
        hovertemplate="%{text}<extra></extra>",
        line=dict(color=MARYLAND_GOLD, width=1.5, dash="dot"),
        fillcolor="rgba(255,213,32,0.08)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=cur + [cur[0]], theta=cats + [cats[0]], fill="toself",
        name="Current",
        text=hover_texts(cur_raw, "Current"),
        hovertemplate="%{text}<extra></extra>",
        line=dict(color="#888888", width=1.5),
        fillcolor="rgba(136,136,136,0.06)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=proj + [proj[0]], theta=cats + [cats[0]], fill="toself",
        name="Projected at " + target_name,
        text=hover_texts(proj_raw, "Projected"),
        hovertemplate="%{text}<extra></extra>",
        line=dict(color=MARYLAND_RED, width=2.5),
        fillcolor="rgba(224,58,62,0.12)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False,
                gridcolor="rgba(0,0,0,0.06)", linecolor="rgba(0,0,0,0)"),
            angularaxis=dict(gridcolor="rgba(0,0,0,0.06)", linecolor="rgba(0,0,0,0.08)",
                tickfont=dict(size=13, color=TEXT)),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.08,
            xanchor="center", x=0.5, font=dict(size=11, color=MUTED)),
        height=380, margin=dict(l=60, r=60, t=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def generate_commentary(player, pred, cases, target_school, target_conf):
    """Generate scouting-style commentary in real basketball language."""
    name = player.get("name", "This player")
    pos = player.get("position", "SF")
    from_school = player.get("from_school", "")
    from_conf = player.get("from_conf", "")
    pts = player.get("pts", 0)
    reb = player.get("reb", 0)
    ast = player.get("ast", 0)
    adj = pred["adj"]

    # Best comparable that actually transferred
    best = None
    for c in cases:
        if c["from_school"] != c["to_school"]:
            best = c
            break
    if best is None and cases:
        best = cases[0]

    # Outcome tallies
    outcomes = [c["outcome"] for c in cases]
    high_outcomes = [o for o in outcomes if o in (
        "starter", "starter_injured", "lottery_pick", "draft_first_round",
        "draft_second_round", "lottery_pick_candidate")]
    low_outcomes = [o for o in outcomes if o in ("bench", "depth", "overseas", "undrafted")]

    # --- Comp anchor ---
    if best:
        comp_name = best["name"]
        comp_path = best["from_school"] + " to " + best["to_school"]
        comp_outcome = best.get("outcome", "")
        _olabels = {
            "starter": "started right away", "starter_injured": "started before going down",
            "rotation": "carved out a rotation role", "bench": "never cracked the rotation",
            "lottery_pick": "became a lottery pick", "draft_first_round": "heard his name on draft night",
            "draft_second_round": "went in the second round",
            "undrafted": "went undrafted", "overseas": "ended up overseas",
            "transferred": "entered the portal again", "transferred_again": "entered the portal again",
            "grad_transfer": "was a grad transfer piece",
            "undrafted_then_nba": "went undrafted but stuck in the league",
            "draft_first_round_injured": "was a first-rounder despite the injury",
            "lottery_pick_candidate": "put himself in the lottery conversation",
        }
        comp_fate = _olabels.get(comp_outcome, "had a mixed outcome")
        opening = ("Closest comp: " + comp_name + " (" + comp_path + "), who " + comp_fate + ".")
    else:
        opening = "Thin comp pool. Limited portal data for this profile."

    # --- Conference move ---
    if adj < 0.88:
        conf_note = ("Big jump in level of competition. " + from_conf + " to " + target_conf
            + ", longer and more athletic on every possession. "
            + "Empty stats concern is real. Need to see if the production is legit against high-major defenders.")
    elif adj < 0.96:
        conf_note = ("Step up from " + from_conf + " to " + target_conf
            + ". Numbers will dip, but if the skill translates he can still produce. "
            + "Watch early-season matchups closely.")
    elif adj > 1.10:
        conf_note = ("Dropping down in conference strength. His " + from_conf
            + " tape is against better competition than he'll see in " + target_conf
            + ". Production should hold or tick up.")
    elif adj > 1.04:
        conf_note = ("Slight step down, " + from_conf + " to " + target_conf
            + ". Should get easier looks. Numbers are portable.")
    else:
        conf_note = (from_conf + " to " + target_conf + ", lateral move. "
            + "No adjustment factor. The tape is the tape.")

    # --- Player profile in basketball language ---
    if pos in ("C", "PF"):
        if pts >= 12 and reb >= 7:
            profile = ("Two-way big. " + _fmt(pts) + " and " + _fmt(reb)
                + ", that's a guy who can anchor your frontcourt. Rim protector "
                + "who finishes around the basket. " + target_school + " gets an immediate starter.")
        elif pts >= 12 and reb >= 4:
            profile = ("Skilled big who can score, " + _fmt(pts)
                + " with some touch. The " + _fmt(reb) + " boards are a question mark. "
                + "Can he crash the glass against " + target_conf + " bigs, or is he a finesse four?")
        elif reb >= 6:
            profile = ("High-motor big. " + _fmt(pts) + "/" + _fmt(reb)
                + ", not your go-to scorer, but he crashes the glass and does the dirty work. "
                + "Every roster needs a guy like this. Plays bigger than his stat line.")
        elif pts >= 8:
            profile = ("Stretch big or face-up four at " + _fmt(pts)
                + " PPG. Can step out and score, but " + _fmt(reb)
                + " rebounds is light for the position. Fit depends on what else is on the roster.")
        else:
            profile = ("Project big. " + _fmt(pts) + "/" + _fmt(reb)
                + " is developmental-level production. Needs reps and the right coaching staff "
                + "to unlock his tools.")
    elif pos == "PG":
        if ast >= 4.5:
            profile = ("Floor general. " + _fmt(ast) + " dimes, he makes others better "
                + "and operates the ball screen game. At " + _fmt(pts) + " PPG he can also "
                + "get his own bucket when the shot clock winds down. Table-setter.")
        elif ast >= 3 and pts >= 10:
            profile = ("Combo guard who can run the show. " + _fmt(pts) + "/" + _fmt(ast)
                + ", scores enough to keep defenses honest and distributes well enough "
                + "to run an offense. Not a pure point, but effective.")
        elif pts >= 13:
            profile = ("Scoring point guard. " + _fmt(pts) + " PPG with "
                + _fmt(ast) + " assists, gets downhill and scores at the rim "
                + "or off the pull-up. Needs shooters around him to open driving lanes.")
        else:
            profile = ("Backup point guard profile. " + _fmt(pts) + "/" + _fmt(ast)
                + ". Steady, won't lose you the game. Good in spot minutes "
                + "and can manage the offense. Not a difference-maker as a starter.")
    elif pos == "SG":
        if pts >= 15:
            profile = ("Bucket-getter. " + _fmt(pts) + " PPG, he can create off the bounce "
                + "and score at all three levels. The kind of perimeter scorer "
                + "you build a wing rotation around.")
        elif pts >= 10:
            profile = (_fmt(pts) + " PPG from the two. Floor spacer who can "
                + "knock down catch-and-shoot threes and attack closeouts. "
                + "Solid secondary scorer, doesn't need the ball to be effective.")
        elif pts >= 6:
            profile = ("3-and-D wing profile. " + _fmt(pts) + " PPG, value is in "
                + "his shooting gravity and on-ball defense, not usage. "
                + "Glue guy who makes lineups work.")
        else:
            profile = ("Low-usage two guard. " + _fmt(pts) + " PPG "
                + "is limited production. Needs to show he can contribute on the defensive end "
                + "to earn minutes at the next level.")
    else:  # SF, wings
        if pts >= 13 and reb >= 5:
            profile = ("Two-way wing. " + _fmt(pts) + "/" + _fmt(reb)
                + ", he can guard multiple positions and score in transition "
                + "and the half-court. Switchable defender with offensive upside. "
                + "Exactly what every coach is looking for in the portal.")
        elif pts >= 12:
            profile = ("Scoring wing at " + _fmt(pts) + " PPG. "
                + "Can play the three or slide to the four in small ball. "
                + "Gives " + target_school + " another shot-creator on the perimeter.")
        elif pts >= 7 and reb >= 4:
            profile = ("Glue guy. " + _fmt(pts) + "/" + _fmt(reb) + "/" + _fmt(ast)
                + ", does a little bit of everything. Guards his position, "
                + "rebounds, makes the right play. Not a star, but a winning player.")
        else:
            profile = ("Developmental wing. " + _fmt(pts) + " PPG, "
                + "the tools may be there but the production isn't yet. "
                + "Needs a defined role and reps to find his game at this level.")

    # --- Bottom line ---
    if len(high_outcomes) >= 3:
        bottom = "Comps strongly favor this move. Track record says he'll produce."
    elif len(high_outcomes) >= 2:
        bottom = "Good track record among comparable portal moves. Worth the bet."
    elif len(high_outcomes) >= 1 and len(low_outcomes) <= 1:
        bottom = "Comp data leans positive. Not a sure thing, but the profile fits."
    elif len(low_outcomes) >= 3:
        bottom = "Comps are cautionary. This profile has underperformed more often than not."
    else:
        bottom = "Mixed bag among comps. Upside is there, but so is the miss rate."

    return opening + " " + conf_note + " " + profile + " " + bottom

# ---------------------------------------------------------------------------
# Data + player lookup
# ---------------------------------------------------------------------------
df_transfers_raw = load_transfer_data()
df_transfers = bootstrap_training_set(df_transfers_raw)
df_players = load_current_players()

def _build_player_lookup(data):
    lookup = {}
    for _, row in data.iterrows():
        name = str(row["name"]).strip()
        school = str(row["school"]).strip()
        year = str(row.get("year_label", ""))
        label = name + " (" + school + ", " + year + ")"
        # Normalize percentages to 0-1 if stored as 0-100
        fg = float(row.get("fg_pct", 42) or 42)
        tp = float(row.get("three_pct", 33) or 33)
        ft = float(row.get("ft_pct", 70) or 70)
        if fg > 1:
            fg = fg / 100.0
        if tp > 1:
            tp = tp / 100.0
        if ft > 1:
            ft = ft / 100.0

        lookup[label] = {
            "name": name, "position": row["position"],
            "height": str(row["height"]),
            "weight": int(float(row["weight"])),
            "from_school": school,
            "from_conf": row["conference"],
            "pts": float(row["pts"]), "reb": float(row["reb"]),
            "ast": float(row["ast"]),
            "stl": float(row.get("stl", 0) or 0),
            "blk": float(row.get("blk", 0) or 0),
            "to": float(row.get("to", 0) or 0),
            "min": float(row.get("min", 20) or 20),
            "gp": int(float(row.get("gp", 25) or 25)),
            "fg_pct": fg, "three_pct": tp, "ft_pct": ft,
        }
    return lookup

PLAYER_LOOKUP = _build_player_lookup(df_players)
PLAYER_LABELS = ["-- Select a player --"] + sorted(PLAYER_LOOKUP.keys())

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _sync_player():
    label = st.session_state.get("player_select", "")
    if label in PLAYER_LOOKUP:
        p = PLAYER_LOOKUP[label]
        st.session_state["p_position"] = p["position"]
        st.session_state["p_height"] = p["height"]
        st.session_state["p_weight"] = p["weight"]
        st.session_state["p_pts"] = p["pts"]
        st.session_state["p_reb"] = p["reb"]
        st.session_state["p_ast"] = p["ast"]
        # Store advanced stats in session state
        for key in ("stl", "blk", "to", "min", "gp", "fg_pct", "three_pct", "ft_pct"):
            st.session_state["p_" + key] = p.get(key, 0)
        if p["from_school"] in ALL_SCHOOLS:
            st.session_state["from_school"] = p["from_school"]
        conf = p["from_conf"]
        if conf and conf in CONF_KEYS:
            st.session_state["from_conf"] = conf

def _sync_from_conf():
    school = st.session_state.get("from_school", "")
    conf = SCHOOL_TO_CONFERENCE.get(school, "")
    if conf and conf in CONF_KEYS:
        st.session_state["from_conf"] = conf

def _sync_to_conf():
    school = st.session_state.get("to_school", "")
    conf = SCHOOL_TO_CONFERENCE.get(school, "")
    if conf and conf in CONF_KEYS:
        st.session_state["tconf"] = conf

# Defaults
if "page" not in st.session_state:
    st.session_state["page"] = "input"
if "from_school" not in st.session_state:
    st.session_state["from_school"] = "Indiana"
    st.session_state["from_conf"] = "Big Ten"
if "to_school" not in st.session_state:
    st.session_state["to_school"] = "Maryland"
    st.session_state["tconf"] = "Big Ten"
if "p_position" not in st.session_state:
    st.session_state["p_position"] = "C"
if "p_height" not in st.session_state:
    st.session_state["p_height"] = "6-11"
if "p_weight" not in st.session_state:
    st.session_state["p_weight"] = 260
if "p_pts" not in st.session_state:
    st.session_state["p_pts"] = 8.5
if "p_reb" not in st.session_state:
    st.session_state["p_reb"] = 6.8
if "p_ast" not in st.session_state:
    st.session_state["p_ast"] = 1.0

# =========================================================================
# PAGE: INPUT
# =========================================================================
if st.session_state["page"] == "input":

    st.markdown(
        '<p class="header-title">Transfer Portal Evaluator</p>'
        '<p class="header-sub">Player performance projection for conference transfers</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-label">Player</p>', unsafe_allow_html=True)
    player_sel = st.selectbox(
        "Select Player", PLAYER_LABELS,
        key="player_select", on_change=_sync_player, label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Position", POSITIONS, key="p_position")
    with col2:
        st.text_input("Height", key="p_height")
    with col3:
        st.number_input("Weight", min_value=150, max_value=350, key="p_weight")

    st.markdown('<p class="section-label" style="margin-top:1rem;">Origin</p>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        st.selectbox("School", ALL_SCHOOLS, key="from_school", on_change=_sync_from_conf)
    with col5:
        st.selectbox("Conference", CONF_KEYS, key="from_conf")

    st.markdown('<p class="section-label" style="margin-top:1rem;">Last Season Stats</p>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    with col6:
        st.number_input("PTS/G", step=0.1, key="p_pts")
    with col7:
        st.number_input("REB/G", step=0.1, key="p_reb")
    with col8:
        st.number_input("AST/G", step=0.1, key="p_ast")

    st.markdown('<p class="section-label" style="margin-top:1rem;">Target</p>', unsafe_allow_html=True)
    col9, col10 = st.columns(2)
    with col9:
        st.selectbox("School ", ALL_SCHOOLS, key="to_school", on_change=_sync_to_conf)
    with col10:
        st.selectbox("Conference ", CONF_KEYS, key="tconf")

    st.markdown('<div style="margin-top:1.2rem;">', unsafe_allow_html=True)
    if st.button("Evaluate", type="primary", use_container_width=True):
        if st.session_state.get("player_select", "") != "-- Select a player --":
            st.session_state["page"] = "results"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="footer">Transfer Portal Evaluator &middot; Built for Maryland Basketball</p>',
        unsafe_allow_html=True,
    )

# =========================================================================
# PAGE: RESULTS
# =========================================================================
elif st.session_state["page"] == "results":

    # Gather input from session state
    player_sel = st.session_state.get("player_select", "")
    if player_sel in PLAYER_LOOKUP:
        player_name = PLAYER_LOOKUP[player_sel]["name"]
    else:
        player_name = "Unknown"

    position = st.session_state.get("p_position", "C")
    height = st.session_state.get("p_height", "6-11")
    weight = st.session_state.get("p_weight", 260)
    from_school = st.session_state.get("from_school", "Indiana")
    from_conf = st.session_state.get("from_conf", "Big Ten")
    last_pts = st.session_state.get("p_pts", 8.5)
    last_reb = st.session_state.get("p_reb", 6.8)
    last_ast = st.session_state.get("p_ast", 1.0)
    target_school = st.session_state.get("to_school", "Maryland")
    target_conf = st.session_state.get("tconf", "Big Ten")

    player_input = {
        "name": player_name, "position": position, "height": height,
        "weight": weight, "from_school": from_school, "from_conf": from_conf,
        "pts": last_pts, "reb": last_reb, "ast": last_ast,
        "stl": st.session_state.get("p_stl", 0.8),
        "blk": st.session_state.get("p_blk", 0.4),
        "to": st.session_state.get("p_to", 1.5),
        "min": st.session_state.get("p_min", 25.0),
        "fg_pct": st.session_state.get("p_fg_pct", 0.42),
        "three_pct": st.session_state.get("p_three_pct", 0.33),
        "ft_pct": st.session_state.get("p_ft_pct", 0.70),
    }

    cases = find_similar(player_input, df_transfers, k=5)
    pred = predict(player_input, cases, target_conf)

    if pred is None:
        st.warning("Not enough data to project.")
        if st.button("Back"):
            st.session_state["page"] = "input"
            st.rerun()
        st.stop()

    # --- Back button ---
    if st.button("New Evaluation"):
        st.session_state["page"] = "input"
        st.rerun()

    # --- Avatar card ---
    color = _avatar_color(player_name)
    initials = _avatar_initials(player_name)
    direction = "lateral" if abs(pred["adj"] - 1.0) < 0.05 else ("up" if pred["adj"] < 1 else "down")
    dir_label = {"up": "Stronger conference", "down": "Weaker conference", "lateral": "Lateral move"}

    avatar_html = (
        '<div class="avatar-card">'
        '<div class="avatar-circle" style="background:' + color + ';">' + initials + '</div>'
        '<div class="avatar-info">'
        '<span class="avatar-name">' + player_name + '</span>'
        '<span class="avatar-detail">'
        + position + ' &middot; ' + height + ' &middot; ' + str(weight) + ' lbs'
        + '</span>'
        '<span class="avatar-transfer">'
        + from_school + ' (' + from_conf + ') &rarr; ' + target_school + ' (' + target_conf + ')'
        + ' &middot; ' + dir_label[direction]
        + '</span>'
        '</div>'
        '</div>'
    )
    st.markdown(avatar_html, unsafe_allow_html=True)

    # --- Projected stats (clean numbers) ---
    st.markdown('<p class="section-label">Projected Stats</p>', unsafe_allow_html=True)

    def delta_html(pred_val, orig_val):
        d = pred_val - orig_val
        if abs(d) < 0.3:
            return ""
        cls = "stat-delta-up" if d > 0 else "stat-delta-down"
        sign = "+" if d > 0 else ""
        return '<span class="stat-delta ' + cls + '">' + sign + _fmt(d) + '</span>'

    stat_html = (
        '<div class="stat-row">'
        '<div class="stat-block">'
        '<span class="stat-num-accent">' + _fmt(pred["pts"]) + '</span>'
        '<span class="stat-label">Pts / G</span>'
        + delta_html(pred["pts"], last_pts)
        + '</div>'
        '<div class="stat-block">'
        '<span class="stat-num">' + _fmt(pred["reb"]) + '</span>'
        '<span class="stat-label">Reb / G</span>'
        + delta_html(pred["reb"], last_reb)
        + '</div>'
        '<div class="stat-block">'
        '<span class="stat-num">' + _fmt(pred["ast"]) + '</span>'
        '<span class="stat-label">Ast / G</span>'
        + delta_html(pred["ast"], last_ast)
        + '</div>'
        '</div>'
    )
    st.markdown(stat_html, unsafe_allow_html=True)

    # --- Verdict ---
    if pred["pts"] >= 12:
        v_cls, v_txt, v_note = "verdict-high", "HIGH PRIORITY", "Projects as an impact starter."
    elif pred["pts"] >= 8:
        v_cls, v_txt, v_note = "verdict-mid", "ROTATION", "Solid contributor. Consistent minutes."
    else:
        v_cls, v_txt, v_note = "verdict-low", "DEPTH", "Role player. Situational minutes."

    st.markdown(
        '<span class="verdict ' + v_cls + '">' + v_txt + '</span>'
        '<span style="font-size:0.82rem;color:' + TEXT + ';margin-left:0.75rem;">' + v_note + '</span>',
        unsafe_allow_html=True,
    )

    # --- Radar ---
    st.markdown('<p class="section-label" style="margin-top:2rem;">Performance Profile</p>', unsafe_allow_html=True)
    fig = make_radar(player_input, cases, pred, target_name=target_school)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # --- Commentary ---
    st.markdown('<p class="section-label">Scouting Notes</p>', unsafe_allow_html=True)
    commentary = generate_commentary(player_input, pred, cases, target_school, target_conf)
    st.markdown('<div class="commentary">' + commentary + '</div>', unsafe_allow_html=True)

    # --- Comparable transfers ---
    st.markdown('<p class="section-label">Comparable Transfers</p>', unsafe_allow_html=True)

    _outcome_labels = {
        "rotation": "Rotation", "starter": "Starter", "bench": "Bench",
        "starter_injured": "Starter (inj.)", "unknown": "TBD",
        "lottery_pick": "Lottery Pick", "draft_first_round": "1st Round",
        "draft_second_round": "2nd Round", "undrafted": "Undrafted",
        "overseas": "Overseas", "transferred": "Transferred",
        "transferred_again": "Transferred", "grad_transfer": "Grad Transfer",
        "rotation_then_transfer": "Rotation", "stayed": "Stayed",
        "undrafted_then_nba": "Undrafted/NBA",
        "draft_first_round_injured": "1st Round (inj.)",
        "lottery_pick_candidate": "Lottery Cand.", "role_player": "Role Player",
    }

    rows_html = ""
    for c in cases:
        bar_w = int(c["sim"] * 80)
        outcome_label = _outcome_labels.get(c["outcome"], c["outcome"])
        rows_html += (
            '<tr>'
            '<td style="font-weight:500;">' + c["name"] + '</td>'
            '<td>' + c["from_school"] + ' &rarr; ' + c["to_school"] + '</td>'
            '<td>' + c["position"] + '</td>'
            '<td>' + _fmt(c["pts"]) + ' / ' + _fmt(c["reb"]) + ' / ' + _fmt(c["ast"]) + '</td>'
            '<td><span class="sim-bar" style="width:' + str(bar_w) + 'px;"></span>' + str(round(c["sim"] * 100)) + '%</td>'
            '<td>' + outcome_label + '</td>'
            '</tr>'
        )

    table_html = (
        '<table class="comp-table">'
        '<thead><tr>'
        '<th>Player</th><th>Transfer</th><th>Pos</th>'
        '<th>PTS / REB / AST</th><th>Match</th><th>Outcome</th>'
        '</tr></thead>'
        '<tbody>' + rows_html + '</tbody>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # --- Footnote ---
    st.markdown(
        '<p style="font-size:0.72rem;color:' + MUTED + ';margin-top:1.5rem;">'
        'Range: ' + _fmt(pred["low"]) + ' &ndash; ' + _fmt(pred["high"]) + ' PTS/G '
        '&middot; ' + str(pred["n"]) + ' comparable transfers '
        '&middot; Conference strength via SRS</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="footer">Transfer Portal Evaluator &middot; Built for Maryland Basketball</p>',
        unsafe_allow_html=True,
    )
