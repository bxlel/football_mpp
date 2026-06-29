"""Drapeaux des sélections (codes ISO pour flagcdn.com - fiables partout)."""

ISO = {
    "Algeria": "dz", "Argentina": "ar", "Australia": "au", "Austria": "at",
    "Belgium": "be", "Bosnia and Herzegovina": "ba", "Brazil": "br",
    "Canada": "ca", "Cape Verde": "cv", "Colombia": "co", "Croatia": "hr",
    "Curaçao": "cw", "Czech Republic": "cz", "DR Congo": "cd", "Ecuador": "ec",
    "Egypt": "eg", "England": "gb-eng", "France": "fr", "Germany": "de",
    "Ghana": "gh", "Haiti": "ht", "Iran": "ir", "Iraq": "iq",
    "Ivory Coast": "ci", "Japan": "jp", "Jordan": "jo", "Mexico": "mx",
    "Morocco": "ma", "Netherlands": "nl", "New Zealand": "nz", "Norway": "no",
    "Panama": "pa", "Paraguay": "py", "Portugal": "pt", "Qatar": "qa",
    "Saudi Arabia": "sa", "Scotland": "gb-sct", "Senegal": "sn",
    "South Africa": "za", "South Korea": "kr", "Spain": "es", "Sweden": "se",
    "Switzerland": "ch", "Tunisia": "tn", "Turkey": "tr", "United States": "us",
    "Uruguay": "uy", "Uzbekistan": "uz",
}

ACCENT = {
    "Algeria": "#0a8f3c", "Argentina": "#6cabdd", "Australia": "#f4c500",
    "Austria": "#ed2939", "Belgium": "#e30613", "Bosnia and Herzegovina": "#004494",
    "Brazil": "#009b3a", "Canada": "#d52b1e", "Cape Verde": "#1a3a7a",
    "Colombia": "#fcd116", "Croatia": "#e8112d", "Curaçao": "#002b7f",
    "Czech Republic": "#d7141a", "DR Congo": "#007fff", "Ecuador": "#ffd100",
    "Egypt": "#c8102e", "England": "#cf081f", "France": "#21304f",
    "Germany": "#d4af37", "Ghana": "#006b3f", "Haiti": "#d21034", "Iran": "#239f40",
    "Iraq": "#1a8a3c", "Ivory Coast": "#f77f00", "Japan": "#bc002d",
    "Jordan": "#ce1126", "Mexico": "#0a7d3b", "Morocco": "#c1272d",
    "Netherlands": "#ff6900", "New Zealand": "#1a1a1a", "Norway": "#ba0c2f",
    "Panama": "#005293", "Paraguay": "#d52b1e", "Portugal": "#a01419",
    "Qatar": "#8a1538", "Saudi Arabia": "#0a6e3c", "Scotland": "#0a3161",
    "Senegal": "#0a843f", "South Africa": "#007749", "South Korea": "#0047a0",
    "Spain": "#c60b1e", "Sweden": "#0057b7", "Switzerland": "#d52b1e",
    "Tunisia": "#e70013", "Turkey": "#e30a17", "United States": "#0a3161",
    "Uruguay": "#5ba3d0", "Uzbekistan": "#0099b5",
}

def flag(team_name: str) -> str:
    return ISO.get(team_name, "")

def accent(team_name: str) -> str:
    return ACCENT.get(team_name, "#8aa395")
