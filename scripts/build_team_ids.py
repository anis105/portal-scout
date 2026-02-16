"""
Fetch ESPN team IDs and match to SCHOOL_TO_CONFERENCE.
Writes data/espn_team_ids.json.

Usage:
    python3 scripts/build_team_ids.py
"""

import json
import sys
import urllib.request

sys.path.insert(0, ".")
from data.conferences import SCHOOL_TO_CONFERENCE

ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams?limit=400"
)

# Manual overrides for ESPN location -> our school name
ESPN_NAME_OVERRIDES = {
    "UConn": "UConn",
    "Pitt": "Pittsburgh",
    "Ole Miss": "Ole Miss",
    "Miami": "Miami (FL)",
    "Saint Mary's": "Saint Mary's",
    "Loyola Chicago": "Loyola Chicago",
    "Loyola Maryland": "Loyola (MD)",
    "Saint Joseph's": "Saint Joseph's",
    "UTEP": "UTEP",
    "UTSA": "UTSA",
    "UT Arlington": "UT Arlington",
    "UMass": "UMass",
    "UNC Greensboro": "UNC Greensboro",
    "UNC Wilmington": "UNC Wilmington",
    "UNC Asheville": "UNC Asheville",
    "North Carolina": "North Carolina",
    "VCU": "VCU",
    "SIUE": "SIU Edwardsville",
    "LIU": "LIU",
    "FDU": "Fairleigh Dickinson",
    "UMBC": "UMBC",
    "SFA": "Stephen F. Austin",
    "Hawai'i": "Hawaii",
    "San Jose State": "San Jose State",
    "San José State": "San Jose State",
    "California": "California",
    "Bowling Green": "Bowling Green",
    "Ohio": "Ohio",
    "App State": "Appalachian State",
    "Ball State": "Ball State",
    "Buffalo": "Buffalo",
    "Central Michigan": "Central Michigan",
    "Eastern Michigan": "Eastern Michigan",
    "Pennsylvania": "Penn",
    "Kent State": "Kent State",
    "UL Monroe": "Louisiana-Monroe",
    "Northern Illinois": "Northern Illinois",
    "SE Louisiana": "Southeastern Louisiana",
    "Southeast Missouri State": "Southeast Missouri State",
    "Southern": "Southern",
    "Toledo": "Toledo",
    "Western Michigan": "Western Michigan",
    "Akron": "Akron",
}


def main():
    print("Fetching ESPN teams...")
    req = urllib.request.Request(ESPN_TEAMS_URL)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    espn_teams = data["sports"][0]["leagues"][0]["teams"]
    print(f"ESPN returned {len(espn_teams)} teams")

    result = {}
    matched = 0
    unmatched = []

    for entry in espn_teams:
        team = entry["team"]
        espn_id = str(team["id"])
        location = team.get("location", "")
        abbreviation = team.get("abbreviation", "")
        display_name = team.get("displayName", "")

        # Try matching: override -> location -> abbreviation
        school_name = ESPN_NAME_OVERRIDES.get(location, location)
        conf = SCHOOL_TO_CONFERENCE.get(school_name)

        if not conf:
            conf = SCHOOL_TO_CONFERENCE.get(abbreviation)
            if conf:
                school_name = abbreviation

        if not conf:
            # Try display name without mascot (e.g. "Duke Blue Devils" -> "Duke")
            short = display_name.split(" ")[0] if display_name else ""
            conf = SCHOOL_TO_CONFERENCE.get(short)
            if conf:
                school_name = short

        if conf:
            result[school_name] = {"id": espn_id, "conf": conf}
            matched += 1
        else:
            unmatched.append(f"  {espn_id}: {location} ({abbreviation})")

    out_path = "data/espn_team_ids.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"Matched {matched}/{len(espn_teams)} teams -> {out_path}")
    if unmatched:
        print(f"\nUnmatched ({len(unmatched)}):")
        for line in sorted(unmatched)[:20]:
            print(line)
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")


if __name__ == "__main__":
    main()
