"""Drapeaux (emoji) des sélections nationales pour l'habillage de l'interface.

On utilise les emoji drapeaux : ils s'affichent partout sans télécharger
d'images. Les équipes non listées reçoivent un drapeau neutre.
"""

FLAGS = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Canada": "🇨🇦", "Cape Verde": "🇨🇻", "Colombia": "🇨🇴", "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼", "Czech Republic": "🇨🇿", "DR Congo": "🇨🇩", "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷", "Germany": "🇩🇪",
    "Ghana": "🇬🇭", "Haiti": "🇭🇹", "Iran": "🇮🇷", "Iraq": "🇮🇶",
    "Ivory Coast": "🇨🇮", "Japan": "🇯🇵", "Jordan": "🇯🇴", "Mexico": "🇲🇽",
    "Morocco": "🇲🇦", "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Norway": "🇳🇴",
    "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Portugal": "🇵🇹", "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳",
    "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Spain": "🇪🇸", "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷", "United States": "🇺🇸",
    "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
}

# Couleur d'accent par équipe (pour les bandes), dérivée du drapeau dominant.
ACCENT = {
    "Algeria": "#0a8f3c", "Argentina": "#6cabdd", "Australia": "#f4c500",
    "Austria": "#ed2939", "Belgium": "#e30613", "Bosnia and Herzegovina": "#004494",
    "Brazil": "#ffdf00", "Canada": "#d52b1e", "Cape Verde": "#1a3a7a",
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
    "Spain": "#c60b1e", "Sweden": "#fecb00", "Switzerland": "#d52b1e",
    "Tunisia": "#e70013", "Turkey": "#e30a17", "United States": "#0a3161",
    "Uruguay": "#5ba3d0", "Uzbekistan": "#0099b5",
}


def flag(team_name: str) -> str:
    return FLAGS.get(team_name, "🏳️")


def accent(team_name: str) -> str:
    return ACCENT.get(team_name, "#8aa395")
