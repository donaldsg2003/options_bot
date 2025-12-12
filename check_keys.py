# ======================================================
# check_keys.py — Verify Alpaca API keys from OS keyring
# ======================================================

import keyring
import logging
from alpaca.trading.client import TradingClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def verify_keyring_keys() -> bool:
    """Load keys from keyring and verify Alpaca connectivity."""
    api_key = keyring.get_password("alpaca", "API_KEY")
    secret_key = keyring.get_password("alpaca", "SECRET_KEY")

    if not api_key or not secret_key:
        logging.critical(" Missing keys in keyring (service='alpaca').")
        logging.info("Run `python setup_keys.py` to initialize them.")
        return False

    try:
        client = TradingClient(api_key, secret_key)
        account = client.get_account()
        account_number = getattr(account, "account_number", "UNKNOWN")
        equity = float(getattr(account, "equity", 0.0))
        mode = "Paper" if getattr(account, "paper_trading", False) else "Live"

        logging.info(
            "Alpaca credentials valid: %s | Equity=$%.2f | %s Mode",
            account_number,
            equity,
            mode,
        )
        return True

    except Exception as e:
        logging.error(" Alpaca key verification failed: %s", e)
        return False


def main():
    print("\n Running Alpaca keyring verification...\n")
    ok = verify_keyring_keys()
    if ok:
        print("\n Keyring verification passed — credentials valid and usable.\n")
        exit(0)
    else:
        print("\n Verification failed — please re-run `setup_keys.py`.\n")
        exit(1)


if __name__ == "__main__":
    main()
