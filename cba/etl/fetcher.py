import os

import httpx
from dotenv import load_dotenv

from cba.auth import get_token


class CWAETLFetcher:
    def __init__(self):
        self.base_url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
        self.api_key = get_token()

    def headers(self):
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": self.api_key,  # 自動加上
        }

    def fetch_data(self, api_endpoint: str, location_id: str):
        url = f"{self.base_url}/{api_endpoint}"
        params = {
            "locationId": location_id
        }

        response = httpx.get(url, params=params, headers=self.headers())
        response.raise_for_status()
        return response.json()
