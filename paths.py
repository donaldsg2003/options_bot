# paths.py
"""
Centralized path configuration for the entire bot.
All database and file paths defined here.
"""

from pathlib import Path

# =====================================================
# Base Directories
# =====================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
BACKUPS_DIR = PROJECT_ROOT / "backups"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
ML_DB_PATH = DATA_DIR / "ml_training_data.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
BACKUPS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# =====================================================
# Database Paths
# =====================================================
# Trading bot database (your live trades)
TRADING_DB = DATA_DIR / "options_bot.db"

# Market data database (for backtesting)
MARKET_DATA_DB = DATA_DIR / "market_data.db"

# ML training database (optional - for later)
ML_TRAINING_DB = DATA_DIR / "ml_training_data.db"

# =====================================================
# State Files
# =====================================================
SESSION_STATE = DATA_DIR / "session_state.json"

# =====================================================
# Log Files
# =====================================================
SYSTEM_LOG = LOGS_DIR / "system.log"
TRADES_LOG = LOGS_DIR / "trades.log"
DATA_LOG = LOGS_DIR / "data.log"
MARKET_DATA_LOG = LOGS_DIR / "market_data.log"

# =====================================================
# CSV Trade Journals
# =====================================================
def get_trade_journal_path(month_year: str) -> Path:
    """
    Get path for monthly trade journal CSV.
    
    Args:
        month_year: Format like "2025_jan"
    
    Returns:
        Path to CSV file in data/ directory
    """
    return DATA_DIR / f"options_bot_{month_year}.csv"

# =====================================================
# Configuration Files
# =====================================================
ENV_FILE = CONFIG_DIR / ".env"

# =====================================================
# Backup Paths
# =====================================================
def get_backup_path(db_name: str, timestamp: str) -> Path:
    """
    Get backup path for a database.
    
    Args:
        db_name: Name of database (e.g., "options_bot")
        timestamp: Timestamp string (e.g., "2025-11-25")
    
    Returns:
        Path to backup file
    """
    return BACKUPS_DIR / f"{db_name}_{timestamp}.db"

# =====================================================
# Helper Functions
# =====================================================
def get_all_databases() -> list[Path]:
    """Get list of all database paths."""
    return [
        TRADING_DB,
        MARKET_DATA_DB,
        ML_TRAINING_DB
    ]


def verify_structure():
    """Verify all required directories exist."""
    required_dirs = [
        DATA_DIR,
        LOGS_DIR,
        BACKUPS_DIR,
        CONFIG_DIR
    ]
    
    for directory in required_dirs:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        else:
            print(f"✓ Exists: {directory}")


if __name__ == "__main__":
    print("\n📂 FOLDER STRUCTURE")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"\nDirectories:")
    print(f"  data/      {DATA_DIR}")
    print(f"  logs/      {LOGS_DIR}")
    print(f"  backups/   {BACKUPS_DIR}")
    print(f"  config/    {CONFIG_DIR}")
    print(f"\nDatabases:")
    print(f"  Trading:   {TRADING_DB}")
    print(f"  Market:    {MARKET_DATA_DB}")
    print(f"  ML:        {ML_TRAINING_DB}")
    print("="*60)
    print("\nVerifying structure...")
    verify_structure()
    print("="*60)
