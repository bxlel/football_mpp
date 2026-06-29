"""Couleurs de maillot (domicile) des sélections nationales.

Chaque équipe a une couleur principale et une secondaire, utilisées pour
l'habillage broadcast de l'interface. Couleurs approximatives des maillots
domicile. Les équipes non listées reçoivent une couleur neutre.
"""

JERSEY_COLORS = {
    "Algeria":        ("#0a8f3c", "#ffffff"),
    "Argentina":      ("#6cabdd", "#ffffff"),
    "Australia":      ("#f4c500", "#0b7a3b"),
    "Austria":        ("#ed2939", "#ffffff"),
    "Belgium":        ("#e30613", "#000000"),
    "Bosnia and Herzegovina": ("#004494", "#f7d917"),
    "Brazil":         ("#ffdf00", "#0a843f"),
    "Canada":         ("#d52b1e", "#ffffff"),
    "Cape Verde":     ("#1a3a7a", "#e30613"),
    "Colombia":       ("#fcd116", "#003893"),
    "Croatia":        ("#e8112d", "#ffffff"),
    "Curaçao":        ("#002b7f", "#f9d616"),
    "Czech Republic": ("#d7141a", "#11457e"),
    "DR Congo":       ("#007fff", "#f7d618"),
    "Ecuador":        ("#ffd100", "#0072c6"),
    "Egypt":          ("#c8102e", "#ffffff"),
    "England":        ("#ffffff", "#cf081f"),
    "France":         ("#21304f", "#ed2939"),
    "Germany":        ("#ffffff", "#1a1a1a"),
    "Ghana":          ("#ffffff", "#006b3f"),
    "Haiti":          ("#1a3a7a", "#d21034"),
    "Iran":           ("#ffffff", "#239f40"),
    "Iraq":           ("#1a8a3c", "#ffffff"),
    "Ivory Coast":    ("#f77f00", "#0a843f"),
    "Japan":          ("#0a1f6b", "#ffffff"),
    "Jordan":         ("#ce1126", "#ffffff"),
    "Mexico":         ("#0a7d3b", "#ffffff"),
    "Morocco":        ("#c1272d", "#0a6e3c"),
    "Netherlands":    ("#ff6900", "#ffffff"),
    "New Zealand":    ("#ffffff", "#1a1a1a"),
    "Norway":         ("#ba0c2f", "#00205b"),
    "Panama":         ("#d21034", "#005293"),
    "Paraguay":       ("#d52b1e", "#0038a8"),
    "Portugal":       ("#a01419", "#0a6e3c"),
    "Qatar":          ("#8a1538", "#ffffff"),
    "Saudi Arabia":   ("#0a6e3c", "#ffffff"),
    "Scotland":       ("#0a3161", "#ffffff"),
    "Senegal":        ("#0a843f", "#fcd116"),
    "South Africa":   ("#007749", "#ffb81c"),
    "South Korea":    ("#c8102e", "#0047a0"),
    "Spain":          ("#c60b1e", "#ffc400"),
    "Sweden":         ("#005b99", "#fecb00"),
    "Switzerland":    ("#d52b1e", "#ffffff"),
    "Tunisia":        ("#e70013", "#ffffff"),
    "Turkey":         ("#e30a17", "#ffffff"),
    "United States":  ("#0a3161", "#b31942"),
    "Uruguay":        ("#5ba3d0", "#1a1a1a"),
    "Uzbekistan":     ("#0099b5", "#ffffff"),
}

DEFAULT_COLORS = ("#7a7a8c", "#ffffff")


def jersey(team_name: str):
    """Renvoie (primaire, secondaire) pour une équipe, ou la couleur neutre."""
    return JERSEY_COLORS.get(team_name, DEFAULT_COLORS)
