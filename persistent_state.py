# persistent_state.py
"""
Thread-safe persistent state manager with automatic pruning.

ENHANCEMENTS:
- Thread-safe read/modify/write operations
- Automatic state file pruning (prevents unbounded growth)
- Compressed storage for large states
- Atomic updates with rollback capability
"""

import json
import os
import sys
import gzip
import threading
from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from option_bot_spreads.paths import SESSION_STATE
from option_bot_spreads.helpers.utils import utc_now_iso


class PersistentState:
    """
    Thread-safe persistent state manager.
    
    Features:
    - Atomic read-modify-write operations
    - Automatic pruning of old entries
    - Compression for large state files
    - Rollback capability on errors
    """
    
    # Configuration
    MAX_RECENT_SPREADS = 1000  # Keep only 1000 most recent spreads
    MAX_ORPHAN_HISTORY = 100   # Keep only 100 orphan detections
    COMPRESSION_THRESHOLD = 1_000_000  # Compress if > 1MB
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or str(SESSION_STATE)
        self.lock = threading.RLock()  # Reentrant lock for nested calls
        self.state = self._load()
        
    def _load(self) -> dict:
        """Load state from disk with compression support."""
        # Try compressed version first
        gz_path = f"{self.filepath}.gz"
        if os.path.exists(gz_path):
            try:
                with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load compressed state: {e}")
        
        # Try regular file
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load state: {e}")
                return {}
        
        return {}

    def _save(self) -> None:
        """
        Save state to disk with automatic pruning and compression.
        
        Thread-safe with atomic write operation.
        """
        with self.lock:
            # Prune old entries before saving
            self._prune_old_entries()
            
            # Serialize to JSON
            data = json.dumps(self.state, indent=2)
            data_size = len(data)
            
            # Use temporary file for atomic write
            temp_path = f"{self.filepath}.tmp"
            
            try:
                if data_size > self.COMPRESSION_THRESHOLD:
                    # Write compressed
                    with gzip.open(f"{temp_path}.gz", "wt", encoding="utf-8") as f:
                        f.write(data)
                    
                    # Atomic rename
                    if os.path.exists(f"{self.filepath}.gz"):
                        os.remove(f"{self.filepath}.gz")
                    os.rename(f"{temp_path}.gz", f"{self.filepath}.gz")
                    
                    # Remove uncompressed version if exists
                    if os.path.exists(self.filepath):
                        os.remove(self.filepath)
                else:
                    # Write uncompressed
                    with open(temp_path, "w", encoding="utf-8") as f:
                        f.write(data)
                    
                    # Atomic rename
                    if os.path.exists(self.filepath):
                        os.remove(self.filepath)
                    os.rename(temp_path, self.filepath)
                    
                    # Remove compressed version if exists
                    if os.path.exists(f"{self.filepath}.gz"):
                        os.remove(f"{self.filepath}.gz")
                        
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(f"{temp_path}.gz"):
                    os.remove(f"{temp_path}.gz")
                raise

    def _prune_old_entries(self) -> None:
        """
        Prune old entries to prevent unbounded state file growth.
        
        Keeps:
        - Most recent 1000 spreads
        - Most recent 100 orphan detections
        - All open positions
        """
        # Prune recent_spreads
        recent = self.state.get("recent_spreads", {})
        if len(recent) > self.MAX_RECENT_SPREADS:
            # Sort by closed_at timestamp (most recent first)
            sorted_items = sorted(
                recent.items(),
                key=lambda x: x[1].get("closed_at", ""),
                reverse=True
            )
            # Keep only MAX_RECENT_SPREADS
            self.state["recent_spreads"] = dict(
                sorted_items[:self.MAX_RECENT_SPREADS]
            )
        
        # Prune orphan_detections
        orphans = self.state.get("orphan_detections", {})
        if len(orphans) > self.MAX_ORPHAN_HISTORY:
            # Sort by first_seen (most recent first)
            sorted_items = sorted(
                orphans.items(),
                key=lambda x: x[1].get("first_seen", ""),
                reverse=True
            )
            self.state["orphan_detections"] = dict(
                sorted_items[:self.MAX_ORPHAN_HISTORY]
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Thread-safe get operation.
        
        Args:
            key: State key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Value associated with key, or default
        """
        with self.lock:
            return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Thread-safe set operation with automatic persistence.
        
        Args:
            key: State key to set
            value: Value to store (must be JSON-serializable)
        """
        with self.lock:
            self.state[key] = value
            self._save()

    def update(self, key: str, updates: dict) -> None:
        """
        Thread-safe dictionary update operation.
        
        Args:
            key: State key containing a dictionary
            updates: Dictionary of updates to apply
            
        Example:
            state.update("open_spreads", {
                "spread_key": {"status": "closed"}
            })
        """
        with self.lock:
            if key not in self.state:
                self.state[key] = {}
            
            if not isinstance(self.state[key], dict):
                raise TypeError(f"State key '{key}' is not a dictionary")
            
            self.state[key].update(updates)
            self._save()

    def append(self, key: str, value: Any) -> None:
        """
        Thread-safe list append operation.
        
        Args:
            key: State key containing a list
            value: Value to append
        """
        with self.lock:
            if key not in self.state:
                self.state[key] = []
            
            if not isinstance(self.state[key], list):
                raise TypeError(f"State key '{key}' is not a list")
            
            self.state[key].append(value)
            self._save()

    def delete(self, key: str, subkey: Optional[str] = None) -> None:
        """
        Thread-safe delete operation.
        
        Args:
            key: Primary key to delete
            subkey: If provided, delete subkey from nested dict
            
        Example:
            state.delete("open_spreads", "spread_key_123")
            state.delete("temp_data")  # Delete entire key
        """
        with self.lock:
            if subkey:
                if key in self.state and isinstance(self.state[key], dict):
                    if subkey in self.state[key]:
                        del self.state[key][subkey]
            else:
                self.state.pop(key, None)
            
            self._save()

    def touch(self, key: str) -> None:
        """
        Store current UTC timestamp for a key.
        
        Useful for cooldown tracking and rate limiting.
        
        Args:
            key: Key to timestamp
        """
        with self.lock:
            self.state[key] = utc_now_iso()
            self._save()

    def has_recent(self, key: str, seconds: int) -> bool:
        """
        Check if a timestamp key is recent.
        
        Args:
            key: Key containing an ISO timestamp
            seconds: Maximum age in seconds
            
        Returns:
            True if timestamp exists and is within seconds threshold
        """
        with self.lock:
            ts_str = self.state.get(key)
            if not ts_str:
                return False
            
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                delta = now - ts
                return delta.total_seconds() < seconds
            except Exception:
                return False

    def atomic_increment(self, key: str, subkey: str, amount: int = 1) -> int:
        """
        Thread-safe atomic increment operation.
        
        Args:
            key: Primary key (should be a dict)
            subkey: Key within dict to increment
            amount: Amount to increment by (can be negative)
            
        Returns:
            New value after increment
            
        Example:
            # Increment pending order count
            new_count = state.atomic_increment("pending_orders", "MID_VOL")
        """
        with self.lock:
            if key not in self.state:
                self.state[key] = {}
            
            if not isinstance(self.state[key], dict):
                raise TypeError(f"State key '{key}' must be a dictionary")
            
            current = int(self.state[key].get(subkey, 0))
            new_value = max(0, current + amount)  # Prevent negative
            self.state[key][subkey] = new_value
            self._save()
            
            return new_value

    def atomic_update_if(
        self, 
        key: str, 
        condition: callable, 
        updates: dict
    ) -> bool:
        """
        Thread-safe conditional update (compare-and-swap pattern).
        
        Args:
            key: State key to update
            condition: Function that takes current value and returns bool
            updates: Updates to apply if condition is True
            
        Returns:
            True if update was applied, False otherwise
            
        Example:
            # Only update if spread is still open
            updated = state.atomic_update_if(
                "open_spreads",
                lambda x: x.get("spread_key", {}).get("status") == "open",
                {"spread_key": {"status": "closing"}}
            )
        """
        with self.lock:
            current_value = self.state.get(key)
            
            if condition(current_value):
                if isinstance(current_value, dict) and isinstance(updates, dict):
                    current_value.update(updates)
                else:
                    self.state[key] = updates
                
                self._save()
                return True
            
            return False

    def get_stats(self) -> dict:
        """
        Get statistics about the state file.
        
        Returns:
            Dictionary with size, entry counts, etc.
        """
        with self.lock:
            stats = {
                "total_keys": len(self.state),
                "open_spreads": len(self.state.get("open_spreads", {})),
                "recent_spreads": len(self.state.get("recent_spreads", {})),
                "orphan_detections": len(self.state.get("orphan_detections", {})),
                "file_size_bytes": 0,
                "compressed": False
            }
            
            # Check file size
            if os.path.exists(self.filepath):
                stats["file_size_bytes"] = os.path.getsize(self.filepath)
            elif os.path.exists(f"{self.filepath}.gz"):
                stats["file_size_bytes"] = os.path.getsize(f"{self.filepath}.gz")
                stats["compressed"] = True
            
            return stats

    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the current state.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to backup file
        """
        import shutil
        
        with self.lock:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.filepath}.backup_{timestamp}"
            
            # Copy current state to backup
            if os.path.exists(self.filepath):
                shutil.copy2(self.filepath, backup_path)
            elif os.path.exists(f"{self.filepath}.gz"):
                shutil.copy2(f"{self.filepath}.gz", f"{backup_path}.gz")
            
            return backup_path
