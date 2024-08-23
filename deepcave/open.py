# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# Open

This module opens the browser after a given delay.

## Constants
    DELAY: int
"""

import sys
import webbrowser
from threading import Timer

DELAY = 4


if __name__ == "__main__":
    if "--address" in sys.argv:
        address = sys.argv[sys.argv.index("--address") + 1]

    if "--port" in sys.argv:
        port = sys.argv[sys.argv.index("--port") + 1]

    def open_browser() -> None:
        """Open the link in the browser."""
        webbrowser.open_new(f"http://{address}:{port}")

    Timer(DELAY, open_browser).start()
