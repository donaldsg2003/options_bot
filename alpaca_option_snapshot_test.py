#!/usr/bin/env python3

import os
from datetime import datetime, timezone, timedelta
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockLatestQuoteRequest
from dotenv import load_dotenv
import requests
import pandas as pd

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

load_dotenv()

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

def get_api_keys():
    """Try keyring first, then fall back to env."""
    api_key = ""
    secret_key = ""

    if KEYRING_AVAILABLE:
        try:
            api_key = keyring.get_password("alpaca", "API_KEY") or ""
            secret_key = keyring.get_password("alpaca", "SECRET_KEY") or ""
        except Exception:
            pass

    if not api_key:
        api_key = os.getenv("APCA_API_KEY_ID")
    if not secret_key:
        secret_key = os.getenv("APCA_SECRET_KEY")

    return api_key, secret_key


API_KEY, SECRET_KEY = get_api_keys()

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("Missing Alpaca API credentials")



# 📡 Endpoint for SPY option chain snapshots
BASE_URL = "https://data.alpaca.markets/v1beta1/options/snapshots/SPY"

headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

# 🚀 Request option chain data
response = requests.get(BASE_URL, headers=headers)
data = response.json()

# 🛠️ Extract bid/ask/mid prices
rows = []
for option_symbol, snapshot in data.get("snapshots", {}).items():
    quote = snapshot.get("latestQuote", {})
    bid = quote.get("bid")
    ask = quote.get("ask")
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None
    
    rows.append({
        "Option": option_symbol,
        "Bid": bid,
        "Ask": ask,
        "Mid": mid
    })

# 📊 Convert to DataFrame for easy viewing
df = pd.DataFrame(rows)
print(df.head(20))  # Show first 20 rows
