"""WHO ICD-11 API client (https://icd.who.int/icdapi).

Used to attach standard ICD-11 codes (and official titles/definitions) to the
conditions in the knowledge base, and to serve live coding searches.

Authentication uses OAuth2 client credentials registered at
https://icd.who.int/icdapi (free) — set ICD_CLIENT_ID / ICD_CLIENT_SECRET.
Without credentials every method degrades gracefully: the module never raises
for "not configured", callers check `configured` first.
"""

import time

import httpx

TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
API_BASE = "https://id.who.int/icd"
REQUEST_TIMEOUT = 30.0


class ICDClient:
    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        api_version: str = "v2",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_version = api_version
        self._token: str | None = None
        self._token_expiry = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    # -- internals -----------------------------------------------------------

    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "icdapi_access",
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        self._token_expiry = time.time() + int(payload.get("expires_in", 3600))
        return self._token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token()}",
            "Accept": "application/json",
            "API-Version": self.api_version,
            "Accept-Language": "en",
        }

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        response = httpx.get(
            url, params=params, headers=self._headers(), timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    # -- public API ----------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Free-text search over ICD-11 MMS; returns [{code, title, uri}]."""
        if not self.configured:
            raise RuntimeError("ICD-11 API credentials not configured.")
        data = self._get_json(
            f"{API_BASE}/entity/search",
            {"q": query, "useFlexSearch": "true", "flatResult": "true"},
        )
        results = []
        for entity in (data.get("destinationEntities") or [])[:limit]:
            uri = entity.get("id") or ""
            detail = {}
            if uri:
                try:
                    detail = self._get_json(uri)
                except httpx.HTTPError:
                    detail = {}
            results.append(
                {
                    "code": detail.get("code") or "",
                    "title": _localized(entity.get("title")) or _localized(detail.get("title")),
                    "uri": uri,
                }
            )
        return [r for r in results if r["code"] or r["title"]]

    def lookup(self, disease_name: str) -> dict | None:
        """First best match for a plain disease name, or None."""
        try:
            matches = self.search(disease_name, limit=1)
        except (httpx.HTTPError, RuntimeError, KeyError):
            return None
        return matches[0] if matches else None


def _localized(value) -> str:
    """ICD titles arrive as localized value objects like {"@value": "..."}."""
    if isinstance(value, dict):
        return str(value.get("@value", "")).strip()
    return str(value).strip() if value else ""
