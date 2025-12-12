from datetime import date, datetime
from option_bot_spreads.core.options_pricing import OptionContract, compute_surface_features_for_expiration

spot = 468.00
expiration = date(2025, 12, 26)

# Simulated chain (this proves your math is good)
contracts = [
    OptionContract("C1", 465, expiration, "C", 4.20, 4.30),
    OptionContract("C2", 470, expiration, "C", 1.50, 1.60),
    OptionContract("P1", 465, expiration, "P", 1.20, 1.30),
    OptionContract("P2", 470, expiration, "P", 3.10, 3.20),
]

features = compute_surface_features_for_expiration(
    contracts,
    spot=spot,
    expiration=expiration,
    as_of_dt=datetime.utcnow(),
)

print("FEATURES:", features)
