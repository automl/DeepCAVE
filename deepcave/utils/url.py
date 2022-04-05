import urllib
import json
from typing import Dict, Any, Optional


def create_url(pathname: str, inputs: Dict[str, Any]) -> str:
    string = json.dumps(inputs)
    converted_string = urllib.parse.quote(string)

    return f"{pathname}/?inputs={converted_string}"


def parse_url(pathname: str) -> Optional[Dict[str, Any]]:
    url = urllib.parse.urlparse(pathname)
    query = urllib.parse.parse_qs(url.query)

    if "inputs" in query:
        string = query["inputs"][0]

        return json.loads(string)

    return None
