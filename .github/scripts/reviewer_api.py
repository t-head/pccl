# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.

from functools import cache
import json
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def valid_user_id(user_id):
    return type(user_id) is int and user_id > 0


def valid_login(login):
    return isinstance(login, str) and re.fullmatch(
        r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", login)


class GitHubError(RuntimeError):
    def __init__(self, method, path, status):
        self.status = status
        super().__init__(f"GitHub {method} {path} returned {status}")


class GitHub:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not token:
            raise ValueError("GitHub repository and token are required")
        self.repository, self.token = repository, token

    def request(self, method, path, data=None, *, repository=True):
        endpoint = f"/repos/{self.repository}{path}" if repository else path
        request = Request(
            f"https://api.github.com{endpoint}",
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Authorization": f"Bearer {self.token}",
                     "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json",
                     "X-GitHub-Api-Version": "2022-11-28"}, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            raise GitHubError(method, path, error.code) from None

    @cache
    def login(self, user_id):
        """Resolve a stable identity only for invitations and display; cache per run."""
        user = self.request("GET", f"/user/{user_id}", repository=False)
        if user["id"] != user_id or user["type"] != "User" or not valid_login(user["login"]):
            raise ValueError(f"GitHub user {user_id} is not an eligible human account")
        return user["login"]

    def pages(self, path):
        result = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 31):
            items = self.request("GET", f"{path}{separator}per_page=100&page={page}")
            if not isinstance(items, list):
                raise ValueError("Expected a complete GitHub list")
            result.extend(items)
            if len(items) < 100:
                return result
        raise ValueError("GitHub pagination limit reached; request reviewers manually")
