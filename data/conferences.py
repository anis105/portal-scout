"""
NCAA Division I Men's Basketball — School-to-Conference mapping (2024-25 season).

364 programs across 32 conferences, reflecting the 2024 realignment:
  - Big Ten added Oregon, Washington, UCLA, USC
  - SEC added Texas, Oklahoma
  - Big 12 added Arizona, Arizona State, BYU, Cincinnati, Colorado, Houston, UCF, Utah
  - ACC added Cal, Stanford, SMU
  - Pac-12 reduced to Oregon State and Washington State
"""

SCHOOL_TO_CONFERENCE = {
    # =========================================================================
    # BIG TEN  (18 schools)
    # =========================================================================
    "Illinois": "Big Ten",
    "Indiana": "Big Ten",
    "Iowa": "Big Ten",
    "Maryland": "Big Ten",
    "Michigan": "Big Ten",
    "Michigan State": "Big Ten",
    "Minnesota": "Big Ten",
    "Nebraska": "Big Ten",
    "Northwestern": "Big Ten",
    "Ohio State": "Big Ten",
    "Oregon": "Big Ten",
    "Penn State": "Big Ten",
    "Purdue": "Big Ten",
    "Rutgers": "Big Ten",
    "UCLA": "Big Ten",
    "USC": "Big Ten",
    "Washington": "Big Ten",
    "Wisconsin": "Big Ten",

    # =========================================================================
    # SEC  (16 schools)
    # =========================================================================
    "Alabama": "SEC",
    "Arkansas": "SEC",
    "Auburn": "SEC",
    "Florida": "SEC",
    "Georgia": "SEC",
    "Kentucky": "SEC",
    "LSU": "SEC",
    "Mississippi State": "SEC",
    "Missouri": "SEC",
    "Oklahoma": "SEC",
    "Ole Miss": "SEC",
    "South Carolina": "SEC",
    "Tennessee": "SEC",
    "Texas": "SEC",
    "Texas A&M": "SEC",
    "Vanderbilt": "SEC",

    # =========================================================================
    # BIG 12  (16 schools)
    # =========================================================================
    "Arizona": "Big 12",
    "Arizona State": "Big 12",
    "Baylor": "Big 12",
    "BYU": "Big 12",
    "Cincinnati": "Big 12",
    "Colorado": "Big 12",
    "Houston": "Big 12",
    "Iowa State": "Big 12",
    "Kansas": "Big 12",
    "Kansas State": "Big 12",
    "Oklahoma State": "Big 12",
    "TCU": "Big 12",
    "Texas Tech": "Big 12",
    "UCF": "Big 12",
    "Utah": "Big 12",
    "West Virginia": "Big 12",

    # =========================================================================
    # ACC  (18 schools)
    # =========================================================================
    "Boston College": "ACC",
    "Cal": "ACC",
    "Clemson": "ACC",
    "Duke": "ACC",
    "Florida State": "ACC",
    "Georgia Tech": "ACC",
    "Louisville": "ACC",
    "Miami": "ACC",
    "North Carolina": "ACC",
    "NC State": "ACC",
    "Notre Dame": "ACC",
    "Pittsburgh": "ACC",
    "SMU": "ACC",
    "Stanford": "ACC",
    "Syracuse": "ACC",
    "Virginia": "ACC",
    "Virginia Tech": "ACC",
    "Wake Forest": "ACC",

    # =========================================================================
    # BIG EAST  (11 schools)
    # =========================================================================
    "Butler": "Big East",
    "UConn": "Big East",
    "Creighton": "Big East",
    "DePaul": "Big East",
    "Georgetown": "Big East",
    "Marquette": "Big East",
    "Providence": "Big East",
    "Seton Hall": "Big East",
    "St. John's": "Big East",
    "Villanova": "Big East",
    "Xavier": "Big East",

    # =========================================================================
    # AMERICAN ATHLETIC (14 schools)
    # =========================================================================
    "Charlotte": "American",
    "East Carolina": "American",
    "FAU": "American",
    "Memphis": "American",
    "North Texas": "American",
    "Rice": "American",
    "South Florida": "American",
    "Temple": "American",
    "Tulane": "American",
    "Tulsa": "American",
    "UAB": "American",
    "UTSA": "American",
    "Wichita State": "American",
    "Army": "American",

    # =========================================================================
    # MOUNTAIN WEST  (11 schools)
    # =========================================================================
    "Air Force": "Mountain West",
    "Boise State": "Mountain West",
    "Colorado State": "Mountain West",
    "Fresno State": "Mountain West",
    "Nevada": "Mountain West",
    "New Mexico": "Mountain West",
    "San Diego State": "Mountain West",
    "San Jose State": "Mountain West",
    "UNLV": "Mountain West",
    "Utah State": "Mountain West",
    "Wyoming": "Mountain West",

    # =========================================================================
    # WCC  (10 schools)
    # =========================================================================
    "Gonzaga": "WCC",
    "Loyola Marymount": "WCC",
    "Pacific": "WCC",
    "Pepperdine": "WCC",
    "Portland": "WCC",
    "Saint Mary's": "WCC",
    "San Diego": "WCC",
    "San Francisco": "WCC",
    "Santa Clara": "WCC",

    # =========================================================================
    # ATLANTIC 10  (15 schools)
    # =========================================================================
    "Davidson": "A-10",
    "Dayton": "A-10",
    "Duquesne": "A-10",
    "Fordham": "A-10",
    "George Mason": "A-10",
    "George Washington": "A-10",
    "La Salle": "A-10",
    "Loyola Chicago": "A-10",
    "Massachusetts": "A-10",
    "Rhode Island": "A-10",
    "Richmond": "A-10",
    "Saint Joseph's": "A-10",
    "Saint Louis": "A-10",
    "St. Bonaventure": "A-10",
    "VCU": "A-10",

    # =========================================================================
    # CONFERENCE USA  (10 schools)
    # =========================================================================
    "FIU": "C-USA",
    "Jacksonville State": "C-USA",
    "Kennesaw State": "C-USA",
    "Liberty": "C-USA",
    "Louisiana Tech": "C-USA",
    "Middle Tennessee": "C-USA",
    "New Mexico State": "C-USA",
    "Sam Houston": "C-USA",
    "Western Kentucky": "C-USA",
    "UTEP": "C-USA",

    # =========================================================================
    # PATRIOT LEAGUE  (10 schools)
    # =========================================================================
    "American University": "Patriot",
    "Boston University": "Patriot",
    "Bucknell": "Patriot",
    "Colgate": "Patriot",
    "Holy Cross": "Patriot",
    "Lafayette": "Patriot",
    "Lehigh": "Patriot",
    "Loyola Maryland": "Patriot",
    "Navy": "Patriot",

    # =========================================================================
    # MAAC  (11 schools)
    # =========================================================================
    "Canisius": "MAAC",
    "Fairfield": "MAAC",
    "Iona": "MAAC",
    "Manhattan": "MAAC",
    "Marist": "MAAC",
    "Mount St. Mary's": "MAAC",
    "Niagara": "MAAC",
    "Quinnipiac": "MAAC",
    "Rider": "MAAC",
    "Saint Peter's": "MAAC",
    "Siena": "MAAC",

    # =========================================================================
    # PAC-12  (2 schools for 2024-25)
    # =========================================================================
    "Oregon State": "Pac-12",
    "Washington State": "Pac-12",

    # =========================================================================
    # IVY LEAGUE  (8 schools)
    # =========================================================================
    "Brown": "Ivy League",
    "Columbia": "Ivy League",
    "Cornell": "Ivy League",
    "Dartmouth": "Ivy League",
    "Harvard": "Ivy League",
    "Penn": "Ivy League",
    "Princeton": "Ivy League",
    "Yale": "Ivy League",

    # =========================================================================
    # MISSOURI VALLEY  (12 schools)
    # =========================================================================
    "Belmont": "Missouri Valley",
    "Bradley": "Missouri Valley",
    "Drake": "Missouri Valley",
    "Evansville": "Missouri Valley",
    "Illinois State": "Missouri Valley",
    "Indiana State": "Missouri Valley",
    "Missouri State": "Missouri Valley",
    "Murray State": "Missouri Valley",
    "Northern Iowa": "Missouri Valley",
    "Southern Illinois": "Missouri Valley",
    "UIC": "Missouri Valley",
    "Valparaiso": "Missouri Valley",

    # =========================================================================
    # SOUTHERN CONFERENCE  (10 schools)
    # =========================================================================
    "Chattanooga": "Southern",
    "East Tennessee State": "Southern",
    "Furman": "Southern",
    "Mercer": "Southern",
    "Samford": "Southern",
    "The Citadel": "Southern",
    "UNC Greensboro": "Southern",
    "VMI": "Southern",
    "Western Carolina": "Southern",
    "Wofford": "Southern",

    # =========================================================================
    # BIG SKY  (12 schools)
    # =========================================================================
    "Eastern Washington": "Big Sky",
    "Idaho": "Big Sky",
    "Idaho State": "Big Sky",
    "Montana": "Big Sky",
    "Montana State": "Big Sky",
    "Northern Arizona": "Big Sky",
    "Northern Colorado": "Big Sky",
    "Portland State": "Big Sky",
    "Sacramento State": "Big Sky",
    "Weber State": "Big Sky",

    # =========================================================================
    # BIG SOUTH  (11 schools)
    # =========================================================================
    "Charleston Southern": "Big South",
    "Gardner-Webb": "Big South",
    "High Point": "Big South",
    "Longwood": "Big South",
    "Presbyterian": "Big South",
    "Radford": "Big South",
    "UNC Asheville": "Big South",
    "Winthrop": "Big South",
    "USC Upstate": "Big South",

    # =========================================================================
    # BIG WEST  (11 schools)
    # =========================================================================
    "Cal Poly": "Big West",
    "Cal State Bakersfield": "Big West",
    "Cal State Fullerton": "Big West",
    "Cal State Northridge": "Big West",
    "Hawaii": "Big West",
    "Long Beach State": "Big West",
    "UC Davis": "Big West",
    "UC Irvine": "Big West",
    "UC Riverside": "Big West",
    "UC San Diego": "Big West",
    "UC Santa Barbara": "Big West",

    # =========================================================================
    # CAA (COASTAL ATHLETIC ASSOCIATION)  (13 schools)
    # =========================================================================
    "Campbell": "CAA",
    "Charleston": "CAA",
    "Delaware": "CAA",
    "Drexel": "CAA",
    "Elon": "CAA",
    "Hampton": "CAA",
    "Hofstra": "CAA",
    "Monmouth": "CAA",
    "North Carolina A&T": "CAA",
    "Northeastern": "CAA",
    "Stony Brook": "CAA",
    "Towson": "CAA",
    "William & Mary": "CAA",

    # =========================================================================
    # HORIZON LEAGUE  (12 schools)
    # =========================================================================
    "Cleveland State": "Horizon",
    "Detroit Mercy": "Horizon",
    "Green Bay": "Horizon",
    "IUPUI": "Horizon",
    "Milwaukee": "Horizon",
    "Northern Kentucky": "Horizon",
    "Oakland": "Horizon",
    "Purdue Fort Wayne": "Horizon",
    "Robert Morris": "Horizon",
    "Wright State": "Horizon",
    "Youngstown State": "Horizon",

    # =========================================================================
    # MEAC  (8 schools)
    # =========================================================================
    "Coppin State": "MEAC",
    "Delaware State": "MEAC",
    "Howard": "MEAC",
    "Maryland-Eastern Shore": "MEAC",
    "Morgan State": "MEAC",
    "Norfolk State": "MEAC",
    "North Carolina Central": "MEAC",
    "South Carolina State": "MEAC",

    # =========================================================================
    # NEC (NORTHEAST CONFERENCE)  (10 schools)
    # =========================================================================
    "Central Connecticut": "NEC",
    "Chicago State": "NEC",
    "Fairleigh Dickinson": "NEC",
    "Le Moyne": "NEC",
    "LIU": "NEC",
    "Mercyhurst": "NEC",
    "Sacred Heart": "NEC",
    "Saint Francis": "NEC",
    "Stonehill": "NEC",
    "Wagner": "NEC",

    # =========================================================================
    # OVC (OHIO VALLEY)  (9 schools)
    # =========================================================================
    "Eastern Illinois": "OVC",
    "Lindenwood": "OVC",
    "Little Rock": "OVC",
    "Morehead State": "OVC",
    "SIU Edwardsville": "OVC",
    "Southeast Missouri": "OVC",
    "Southern Indiana": "OVC",
    "Tennessee State": "OVC",
    "Tennessee Tech": "OVC",
    "UT Martin": "OVC",
    "Western Illinois": "OVC",

    # =========================================================================
    # SUN BELT  (14 schools)
    # =========================================================================
    "Appalachian State": "Sun Belt",
    "Arkansas State": "Sun Belt",
    "Coastal Carolina": "Sun Belt",
    "Georgia Southern": "Sun Belt",
    "Georgia State": "Sun Belt",
    "James Madison": "Sun Belt",
    "Louisiana": "Sun Belt",
    "Louisiana-Monroe": "Sun Belt",
    "Marshall": "Sun Belt",
    "Old Dominion": "Sun Belt",
    "South Alabama": "Sun Belt",
    "Southern Miss": "Sun Belt",
    "Texas State": "Sun Belt",
    "Troy": "Sun Belt",

    # =========================================================================
    # SUMMIT LEAGUE  (10 schools)
    # =========================================================================
    "Denver": "Summit",
    "Kansas City": "Summit",
    "North Dakota": "Summit",
    "North Dakota State": "Summit",
    "Omaha": "Summit",
    "Oral Roberts": "Summit",
    "South Dakota": "Summit",
    "South Dakota State": "Summit",
    "St. Thomas": "Summit",

    # =========================================================================
    # SWAC  (12 schools)
    # =========================================================================
    "Alabama A&M": "SWAC",
    "Alabama State": "SWAC",
    "Alcorn State": "SWAC",
    "Arkansas-Pine Bluff": "SWAC",
    "Bethune-Cookman": "SWAC",
    "Florida A&M": "SWAC",
    "Grambling": "SWAC",
    "Jackson State": "SWAC",
    "Mississippi Valley State": "SWAC",
    "Prairie View A&M": "SWAC",
    "Southern University": "SWAC",
    "Texas Southern": "SWAC",

    # =========================================================================
    # WAC  (13 schools)
    # =========================================================================
    "Abilene Christian": "WAC",
    "California Baptist": "WAC",
    "Grand Canyon": "WAC",
    "Lamar": "WAC",
    "Seattle": "WAC",
    "Southern Utah": "WAC",
    "Stephen F. Austin": "WAC",
    "Tarleton State": "WAC",
    "UT Arlington": "WAC",
    "Utah Tech": "WAC",
    "Utah Valley": "WAC",
    "UTRGV": "WAC",

    # =========================================================================
    # ASUN  (13 schools)
    # =========================================================================
    "Austin Peay": "ASUN",
    "Bellarmine": "ASUN",
    "Central Arkansas": "ASUN",
    "Eastern Kentucky": "ASUN",
    "Florida Gulf Coast": "ASUN",
    "Jacksonville": "ASUN",
    "Lipscomb": "ASUN",
    "North Alabama": "ASUN",
    "North Florida": "ASUN",
    "Queens": "ASUN",
    "Stetson": "ASUN",
    "West Georgia": "ASUN",

    # =========================================================================
    # SOUTHLAND  (9 schools)
    # =========================================================================
    "East Texas A&M": "Southland",
    "Houston Christian": "Southland",
    "Incarnate Word": "Southland",
    "McNeese": "Southland",
    "New Orleans": "Southland",
    "Nicholls": "Southland",
    "Northwestern State": "Southland",
    "Southeastern Louisiana": "Southland",
    "Texas A&M-Corpus Christi": "Southland",

    # =========================================================================
    # AMERICA EAST  (10 schools)
    # =========================================================================
    "Albany": "America East",
    "Binghamton": "America East",
    "Bryant": "America East",
    "Maine": "America East",
    "New Hampshire": "America East",
    "NJIT": "America East",
    "UMBC": "America East",
    "UMass Lowell": "America East",
    "Vermont": "America East",

    # =========================================================================
    # ALIASES — common short names used in databases and CSV files
    # =========================================================================
    "UNC": "ACC",                    # = North Carolina
    "Pitt": "ACC",                   # = Pittsburgh
    "SDSU": "Mountain West",         # = San Diego State
    "Florida Atlantic": "American",  # = FAU
    "Loyola (MD)": "Patriot",        # = Loyola Maryland
}

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

# Reverse lookup: conference -> list of schools
CONFERENCE_TO_SCHOOLS: dict[str, list[str]] = {}
for _school, _conf in SCHOOL_TO_CONFERENCE.items():
    CONFERENCE_TO_SCHOOLS.setdefault(_conf, []).append(_school)

# Sorted list of all conferences (for dropdowns)
ALL_CONFERENCES = sorted(CONFERENCE_TO_SCHOOLS.keys())

# Sorted list of all schools (for dropdowns)
ALL_SCHOOLS = sorted(SCHOOL_TO_CONFERENCE.keys())


if __name__ == "__main__":
    print(f"Total schools: {len(SCHOOL_TO_CONFERENCE)}")
    print(f"Total conferences: {len(CONFERENCE_TO_SCHOOLS)}")
    print()
    for conf in ALL_CONFERENCES:
        schools = CONFERENCE_TO_SCHOOLS[conf]
        print(f"  {conf}: {len(schools)} schools")
