"""DuckDB SQL expressions that normalise business names and addresses.

All cleaning runs inside DuckDB so it scales to tens of millions of rows.
Nothing here depends on the country label, so unseen countries (France) are handled
the same way as US and India.
"""

NAME_ABBR = {
    "pvt": "private", "pte": "private", "prvt": "private", "ltd": "limited", "ltda": "limited",
    "corp": "corporation", "co": "company", "cos": "companies", "inc": "incorporated",
    "intl": "international", "mfg": "manufacturing", "bros": "brothers", "svcs": "services",
    "svc": "services", "assoc": "associates", "natl": "national", "mgmt": "management",
    "grp": "group", "tech": "technology", "technologies": "technology", "techs": "technology",
    "engg": "engineering", "eng": "engineering", "inds": "industries", "ind": "industries",
    "cie": "compagnie", "ste": "societe", "st": "saint", "ent": "enterprises",
    "ents": "enterprises", "enterprise": "enterprises", "dist": "distributors",
    "hldgs": "holdings", "sys": "systems", "mfrs": "manufacturers",
}

ADDR_ABBR = {
    "road": "rd", "street": "st", "str": "st", "avenue": "ave", "av": "ave", "lane": "ln",
    "drive": "dr", "boulevard": "blvd", "bd": "blvd", "court": "ct", "place": "pl",
    "highway": "hwy", "parkway": "pkwy", "circle": "cir", "square": "sq", "suite": "ste",
    "apartment": "apt", "floor": "fl", "flr": "fl", "building": "bldg", "near": "nr",
    "opposite": "opp", "north": "n", "south": "s", "east": "e", "west": "w",
    "market": "mkt", "nagar": "ngr", "colony": "clny", "sector": "sec", "station": "stn",
    "number": "no", "chaussee": "chs", "faubourg": "fbg", "route": "rte", "impasse": "imp",
    "allee": "all", "chemin": "chem", "saint": "st", "sainte": "ste",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}

US_STATES_MULTI = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "south carolina": "sc",
    "south dakota": "sd", "west virginia": "wv", "rhode island": "ri",
    "district of columbia": "dc",
}

LEGAL_WORDS = [
    "private", "limited", "llp", "llc", "pllc", "lp", "incorporated", "corporation",
    "company", "companies", "plc", "public", "sarl", "sas", "sasu", "sa", "eurl", "sci",
    "snc", "gmbh", "ag", "the", "and", "of", "opc", "pc", "dba", "fka", "aka", "et",
    "compagnie", "societe", "de", "du", "des", "la", "le", "les", "l", "d",
]

NULL_TOKENS = ["null", "none", "nan", "nil", "unknown"]

DOMAIN_TLDS = "com|net|org|in|co|fr|us|biz|info|io"
DOMAIN_RE = r"\.(" + DOMAIN_TLDS + r")\b"


def q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def sql_list(items) -> str:
    return "[" + ", ".join(q(i) for i in items) + "]"


def _case(var: str, mapping: dict) -> str:
    whens = " ".join(f"WHEN {q(k)} THEN {q(v)}" for k, v in mapping.items())
    return f"(CASE {var} {whens} ELSE {var} END)"


def _translit_lower(x: str) -> str:
    # Only rows containing non-ASCII characters go through the (slower) Python UDF.
    return f"lower(CASE WHEN regexp_matches({x}, '[^\\x00-\\x7F]') THEN translit({x}) ELSE {x} END)"


def _squash(x: str) -> str:
    return f"trim(regexp_replace({x}, '[^a-z0-9]+', ' ', 'g'))"


def _dotted_acronyms(x: str) -> str:
    # l.l.c. -> llc, u.s.a. -> usa, p.c. -> pc
    return rf"regexp_replace({x}, '([a-z])\.([a-z])\.', '\1\2', 'g')"


def name_text(x: str) -> str:
    t = _translit_lower(x)
    # website used as a name: www.gmlabs.com -> gmlabs
    t = (rf"regexp_replace({t}, '^\s*(https?://)?(www\.)?([a-z0-9-]+)\.({DOMAIN_TLDS})"
         rf"(\.[a-z][a-z])?\s*$', '\3')")
    t = f"replace({t}, '&', ' and ')"
    t = _dotted_acronyms(t)
    # digit/letter swaps inside words: regiona1 -> regional, trimb0li -> trimboli, 5unshine -> sunshine
    for d, letter in (("0", "o"), ("1", "l"), ("5", "s")):
        t = rf"regexp_replace(regexp_replace({t}, '([a-z]){d}', '\1{letter}', 'g'), '{d}([a-z])', '{letter}\1', 'g')"
    return _squash(t)


def addr_text(x: str) -> str:
    t = _translit_lower(x)
    t = rf"regexp_replace({t}, '\bn/a\b', ' ', 'g')"
    t = f"replace({t}, '&', ' and ')"
    t = _dotted_acronyms(t)
    t = _squash(t)
    for full, ab in US_STATES_MULTI.items():
        t = rf"regexp_replace({t}, '\b{full}\b', '{ab}', 'g')"
    return t


def name_word(w: str) -> str:
    return _case(w, NAME_ABBR)


def addr_word(w: str) -> str:
    return _case(w, {**ADDR_ABBR, **US_STATES})


def _squeeze(x: str) -> str:
    """Collapse doubled consonants (RE2 has no back-references, so do it letter by letter)."""
    for c in "bcdfghjklmnpqrstvwxz":
        x = f"replace({x}, '{c}{c}', '{c}')"
    return x


def skeleton(w: str) -> str:
    """First letter + consonants, doubles squeezed, y treated as a vowel.

    advisory -> advsr, avsry -> avsr, limited/limittedd -> lmtd, media/miidiyaa -> md
    """
    rest = _squeeze(f"regexp_replace(substr({w}, 2), '[aeiouy]', '', 'g')")
    return f"(CASE WHEN length({w}) <= 3 THEN {w} ELSE substr({w}, 1, 1) || {rest} END)"


# Frequent address words that carry little identity (kept out of address keys).
ADDR_GENERIC = [
    "bldg", "near", "road", "street", "lane", "floor", "house", "shop", "plot", "office",
    "complex", "tower", "block", "phase", "area", "village", "district", "dist", "post",
    "main", "cross", "layout", "city", "town", "unit", "ground", "first", "second", "third",
    "india", "state", "county", "behind", "beside", "next", "opposite", "market",
]
