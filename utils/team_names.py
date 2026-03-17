"""
Bidirectional team name normalization between KenPom and ESPN.

ESPN returns full names with mascots (e.g. "NC State Wolfpack").
KenPom uses abbreviated school names (e.g. "N.C. State").
This module maps all variants to a single canonical name.
"""

# Canonical name -> list of known aliases
_ALIAS_MAP: dict[str, list[str]] = {
    "UConn": ["Connecticut", "CONN", "UConn Huskies"],
    "Ole Miss": ["Mississippi", "Ole Miss Rebels"],
    "Miami FL": ["Miami", "Miami (FL)", "Miami Hurricanes"],
    "Miami OH": ["Miami (OH)", "Miami (OH) RedHawks"],
    "Pitt": ["Pittsburgh", "Pittsburgh Panthers", "Pitt Panthers"],
    "SMU": ["Southern Methodist", "SMU Mustangs"],
    "UCF": ["Central Florida", "UCF Knights"],
    "USC": ["Southern California", "USC Trojans"],
    "LSU": ["Louisiana St.", "LSU Tigers"],
    "VCU": ["Virginia Commonwealth", "VCU Rams"],
    "UNLV": ["Nevada Las Vegas", "UNLV Rebels"],
    "BYU": ["Brigham Young", "BYU Cougars"],
    "N.C. State": ["NC State", "North Carolina St.", "NC State Wolfpack"],
    "UNC": ["North Carolina", "North Carolina Tar Heels", "UNC Tar Heels"],
    "UCSB": ["UC Santa Barbara", "UCSB Gauchos"],
    "UCLA": ["UC Los Angeles", "UCLA Bruins"],
    "UMBC": ["Maryland Baltimore County", "UMBC Retrievers"],
    "UNI": ["Northern Iowa", "Northern Iowa Panthers", "UNI Panthers"],
    "ETSU": ["East Tennessee St.", "ETSU Buccaneers"],
    "MTSU": ["Middle Tennessee", "MTSU Blue Raiders"],
    "FGCU": ["Florida Gulf Coast", "FGCU Eagles"],
    "FDU": ["Fairleigh Dickinson", "FDU Knights"],
    "LIU": ["Long Island University", "Long Island University Sharks", "LIU Sharks"],
    "SFA": ["Stephen F. Austin", "Stephen F. Austin Lumberjacks"],
    "UNCG": ["UNC Greensboro", "UNCG Spartans"],
    "UNCW": ["UNC Wilmington", "UNC Wilmington Seahawks", "UNCW Seahawks"],
    "UT Arlington": ["Texas Arlington"],
    "UT Martin": ["Tennessee Martin"],
    "UTEP": ["Texas El Paso"],
    "UTSA": ["Texas San Antonio"],
    "St. John's": ["St. John's (NY)", "St. John's Red Storm"],
    "Saint Mary's": ["St. Mary's", "Saint Mary's (CA)", "Saint Mary's Gaels"],
    "Saint Joseph's": ["St. Joseph's", "Saint Joseph's Hawks"],
    "Saint Peter's": ["St. Peter's", "Saint Peter's Peacocks"],
    "Saint Louis": ["St. Louis", "Saint Louis Billikens"],
    "Saint Bonaventure": ["St. Bonaventure", "Saint Bonaventure Bonnies"],
    "Loyola Chicago": ["Loyola (IL)", "Loyola-Chicago", "Loyola Chicago Ramblers"],
    "Loyola MD": ["Loyola (MD)", "Loyola Maryland"],
    "Loyola Marymount": ["Loyola (CA)"],
    "Texas A&M": ["Texas A&M Aggies"],
    "Penn": ["Pennsylvania", "Pennsylvania Quakers", "Penn Quakers"],
    "Army": ["Army West Point", "Army Black Knights"],
    "Navy": ["Navy Midshipmen"],
    "UAB": ["Alabama Birmingham", "UAB Blazers"],
    "UTRGV": ["Texas Rio Grande Valley"],
    "Hawai'i": ["Hawaii", "Hawai'i Rainbow Warriors", "Hawaii Rainbow Warriors"],
    "UIC": ["Illinois Chicago", "UIC Flames"],
    "Wichita St.": ["Wichita State", "Wichita State Shockers", "Wichita St"],
    "Murray St.": ["Murray State", "Murray State Racers"],
    "Kent St.": ["Kent State", "Kent State Golden Flashes"],
    "Wright St.": ["Wright State", "Wright State Raiders"],
    "Kennesaw St.": ["Kennesaw State", "Kennesaw State Owls"],
    "Seattle": ["Seattle U", "Seattle U Redhawks", "Seattle Redhawks"],
    "St. Thomas": ["St. Thomas-Minnesota", "St. Thomas-Minnesota Tommies", "St. Thomas (MN)"],
    "Prairie View A&M": ["Prairie View A&M Panthers", "Prairie View"],
    "Sam Houston": ["Sam Houston St.", "Sam Houston Bearkats", "Sam Houston State"],
    "McNeese": ["McNeese St.", "McNeese Cowboys", "McNeese State"],
    "High Point": ["High Point Panthers"],
    "Queens": ["Queens University", "Queens University Royals", "Queens (NC)"],
    "Cal Baptist": ["California Baptist", "California Baptist Lancers", "CBU"],
}

# Build reverse lookup: any variant -> canonical name
_LOOKUP: dict[str, str] = {}

for canonical, aliases in _ALIAS_MAP.items():
    _LOOKUP[canonical.lower()] = canonical
    for alias in aliases:
        _LOOKUP[alias.lower()] = canonical


# Common mascot words to strip when doing fallback matching
_MASCOTS = {
    "wildcats", "bulldogs", "bears", "tigers", "cougars", "eagles", "hawks",
    "knights", "warriors", "wolves", "panthers", "mustangs", "cardinals",
    "gators", "huskies", "longhorns", "mountaineers", "boilermakers",
    "cyclones", "jayhawks", "buckeyes", "spartans", "wolverines",
    "cornhuskers", "volunteers", "razorbacks", "aggies", "cowboys",
    "rebels", "seminoles", "cavaliers", "hoosiers", "bruins",
    "beavers", "ducks", "badgers", "terrapins", "blue devils",
    "crimson tide", "sooners", "red raiders", "horned frogs",
    "golden gophers", "demon deacons", "fighting illini", "tar heels",
    "orangemen", "wolfpack", "commodores", "gamecocks", "hokies",
    "yellow jackets", "zips", "bison", "saints", "flames",
    "flyers", "billikens", "gaels", "rams", "braves", "lancers",
    "owls", "golden flashes", "raiders", "lumberjacks", "seahawks",
    "trojans", "retrievers", "racers", "wolf pack", "lobos",
    "anteaters", "paladins", "royals", "redhawks", "tommies",
    "golden hurricane", "bearkats", "broncos", "red storm",
    "scarlet knights", "bluejays", "rain", "rainbow warriors",
    "jaguars", "bulls", "mountain hawks", "sharks", "vandals",
    "revolutionaries",
}


def _strip_mascot(name: str) -> str:
    """Try to remove a mascot suffix from an ESPN-style full name."""
    lower = name.lower()
    for mascot in sorted(_MASCOTS, key=len, reverse=True):
        if lower.endswith(" " + mascot):
            return name[: -(len(mascot) + 1)].strip()
    return name


def normalize(name: str) -> str:
    """Normalize a team name to its canonical form.

    Handles KenPom names, ESPN names (with mascots), abbreviations, and common variants.
    """
    if not name:
        return name
    stripped = name.strip()

    # Direct lookup
    result = _LOOKUP.get(stripped.lower())
    if result:
        return result

    # Try stripping mascot then looking up
    no_mascot = _strip_mascot(stripped)
    if no_mascot != stripped:
        result = _LOOKUP.get(no_mascot.lower())
        if result:
            return result
        return no_mascot

    return stripped


def fuzzy_match(name: str, candidates: list[str]) -> str | None:
    """Find the best match for *name* among *candidates*.

    Tries exact match first, then normalized match, then substring containment.
    """
    norm = normalize(name)

    for c in candidates:
        if c == name or c == norm:
            return c
        if normalize(c) == norm:
            return c

    name_lower = norm.lower()
    for c in candidates:
        if name_lower in c.lower() or c.lower() in name_lower:
            return c

    return None
