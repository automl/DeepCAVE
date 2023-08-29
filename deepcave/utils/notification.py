#  noqa: D400
"""
# Notification

This module provides utilities for creating, updating and resetting notifications.

## Contents
    - reset: Reset the notification.
    - update: Update the notification
    - get_latest: Retrieve and reset the latest notification.
"""
from typing import Optional, Tuple


class Notification:
    """Can create, update and reset a notification."""

    def __init__(self) -> None:  # noqa: D107
        self._update_required = False
        self._message: Optional[str] = None
        self._color: Optional[str] = None

    def reset(self) -> None:
        """Reset the notification."""
        self._update_required = False
        self._message = None
        self._color = None

    def update(self, message: str, color: str = "danger") -> None:
        """Update the notification."""
        self._update_required = True
        self._message = message
        self._color = color

    def get_latest(self) -> Optional[Tuple[str, str]]:
        """Retrieve and reset the latest notification."""
        if self._update_required and self._message is not None and self._color is not None:
            result = (self._message, self._color)
            self.reset()

            return result

        return None
