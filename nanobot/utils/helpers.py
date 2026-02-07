"""Utility functions for nanobot."""

import os
import atexit
from pathlib import Path
from datetime import datetime


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """Get the nanobot data directory (~/.nanobot)."""
    return ensure_dir(Path.home() / ".nanobot")


def get_workspace_path(workspace: str | None = None) -> Path:
    """
    Get the workspace path.
    
    Args:
        workspace: Optional workspace path. Defaults to ~/.nanobot/workspace.
    
    Returns:
        Expanded and ensured workspace path.
    """
    if workspace:
        path = Path(workspace).expanduser()
    else:
        path = Path.home() / ".nanobot" / "workspace"
    return ensure_dir(path)


def get_sessions_path() -> Path:
    """Get the sessions storage directory."""
    return ensure_dir(get_data_path() / "sessions")


def get_memory_path(workspace: Path | None = None) -> Path:
    """Get the memory directory within the workspace."""
    ws = workspace or get_workspace_path()
    return ensure_dir(ws / "memory")


def get_skills_path(workspace: Path | None = None) -> Path:
    """Get the skills directory within the workspace."""
    ws = workspace or get_workspace_path()
    return ensure_dir(ws / "skills")


def today_date() -> str:
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def truncate_string(s: str, max_len: int = 100, suffix: str = "...") -> str:
    """Truncate a string to max length, adding suffix if truncated."""
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)] + suffix


def safe_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    # Replace unsafe characters
    unsafe = '<>:"/\\|?*'
    for char in unsafe:
        name = name.replace(char, "_")
    return name.strip()


def parse_session_key(key: str) -> tuple[str, str]:
    """
    Parse a session key into channel and chat_id.

    Args:
        key: Session key in format "channel:chat_id"

    Returns:
        Tuple of (channel, chat_id)
    """
    parts = key.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid session key: {key}")
    return parts[0], parts[1]


# ============================================================================
# Single Instance Lock
# ============================================================================

_pid_file_path: Path | None = None


def acquire_instance_lock(name: str = "gateway") -> bool:
    """
    Acquire a single-instance lock using a PID file.

    Args:
        name: Name of the instance (used for the lock file name).

    Returns:
        True if lock was acquired, False if another instance is running.

    Raises:
        RuntimeError: If lock cannot be created due to filesystem issues.
    """
    global _pid_file_path

    lock_dir = get_data_path()
    _pid_file_path = lock_dir / f"{name}.pid"

    # Check if lock file exists
    if _pid_file_path.exists():
        try:
            existing_pid = _pid_file_path.read_text().strip()
            if existing_pid:
                # Check if the process is actually running
                try:
                    os.kill(int(existing_pid), 0)  # Signal 0 just checks if process exists
                    # Process is running
                    return False
                except (OSError, ValueError):
                    # Process is not running or invalid PID, clean up stale lock
                    _pid_file_path.unlink(missing_ok=True)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Cannot read lock file {_pid_file_path}: {e}")

    # Write our PID
    try:
        _pid_file_path.write_text(str(os.getpid()))
        # Register cleanup on exit
        atexit.register(release_instance_lock)
        return True
    except (OSError, IOError) as e:
        raise RuntimeError(f"Cannot create lock file {_pid_file_path}: {e}")


def release_instance_lock() -> None:
    """Release the instance lock by removing the PID file."""
    global _pid_file_path
    if _pid_file_path and _pid_file_path.exists():
        try:
            _pid_file_path.unlink()
        except (OSError, IOError):
            pass  # Best effort cleanup


def is_instance_running(name: str = "gateway") -> bool:
    """
    Check if an instance is already running.

    Args:
        name: Name of the instance.

    Returns:
        True if an instance is running, False otherwise.
    """
    lock_file = get_data_path() / f"{name}.pid"
    if not lock_file.exists():
        return False

    try:
        existing_pid = lock_file.read_text().strip()
        if existing_pid:
            os.kill(int(existing_pid), 0)  # Check if process exists
            return True
    except (OSError, ValueError):
        # Process not running or invalid PID
        lock_file.unlink(missing_ok=True)

    return False
