from typing import Optional, Tuple


class Notification:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._update_required = False
        self._message = None
        self._color = None

    def update(self, message: str, color: str = "danger") -> None:
        self._update_required = True
        self._message = message
        self._color = color

    def get_latest(self) -> Optional[Tuple[str, str]]:
        if self._update_required:
            result = (self._message, self._color)
            self.reset()

            return result

        return None
