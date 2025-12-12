import logging
from option_bot_spreads.backtesting.simulator import RegimeSwitchingSimulator

logging.basicConfig(level=logging.INFO)

print("Starting simulator test...")

sim = RegimeSwitchingSimulator(seed=42)
df = sim.simulate_days(n_days=2000, start_price=500.0, start_regime="MID_VOL")

print("Generated DF Head:")
print(df.head())

print("Generated DF Tail:")
print(df.tail())

print("Done.")
