from typing import Any, Optional

import gitlab
from gitlab import Gitlab, GitlabAuthenticationError
from pydantic import BaseModel, Field  # type: ignore

from app.sources.client.iclient import IClient


class GitLabResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:  # type: ignore
        return self.model_dump()


class GitLabClientViaToken:
    def __init__(
        self,
        token: str,
        url: Optional[str] = None,
        timeout: Optional[float] = None,
        api_version: Optional[str] = "4",
        retry_transient_errors: Optional[bool] = None,
        max_retries: Optional[int] = None,
        obey_rate_limit: Optional[bool] = None,
    ) -> None:
        self.token = token
        self.url = url or "https://gitlab.com"
        self.timeout = timeout
        self.api_version = api_version
        self.retry_transient_errors = retry_transient_errors
        self.max_retries = max_retries
        self.obey_rate_limit = obey_rate_limit

        self._sdk: Optional[Gitlab] = None

    def create_client(self) -> Gitlab:
        kwargs: dict[str, Any] = {
            "url": self.url,
            "private_token": self.token,
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.api_version is not None:
            kwargs["api_version"] = self.api_version
        if self.retry_transient_errors is not None:
            kwargs["retry_transient_errors"] = self.retry_transient_errors
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        if self.obey_rate_limit is not None:
            kwargs["obey_rate_limit"] = self.obey_rate_limit

        self._sdk = gitlab.Gitlab(**kwargs)
        try:
            self._sdk.auth()  # validate the credentials early
        except GitlabAuthenticationError as e:
            raise RuntimeError("GitLab authentication failed") from e
        except Exception as e:
            raise RuntimeError("Error initializing GitLab client") from e

        return self._sdk

    def get_sdk(self) -> Gitlab:
        if self._sdk is None:
            # lazy init if not yet created
            return self.create_client()
        return self._sdk

    def get_base_url(self) -> str:
        return self.url


class GitLabConfig(BaseModel):
    token: str = Field(..., description="GitLab private token")
    url: Optional[str] = Field(default="https://gitlab.com", description="GitLab instance URL")
    timeout: Optional[float] = None
    api_version: Optional[str] = Field(default="4", description="GitLab API version")
    retry_transient_errors: Optional[bool] = None
    max_retries: Optional[int] = None
    obey_rate_limit: Optional[bool] = None

    def create_client(self) -> GitLabClientViaToken:
        return GitLabClientViaToken(
            token=self.token,
            url=self.url,
            timeout=self.timeout,
            api_version=self.api_version,
            retry_transient_errors=self.retry_transient_errors,
            max_retries=self.max_retries,
            obey_rate_limit=self.obey_rate_limit,
        )


class GitLabClient(IClient):
    def __init__(self, client: GitLabClientViaToken) -> None:
        self.client = client

    def get_client(self) -> GitLabClientViaToken:
        return self.client

    def get_sdk(self) -> Gitlab:
        return self.client.get_sdk()

    @classmethod
    def build_with_config(
        cls,
        config: GitLabConfig,
    ) -> "GitLabClient":
        client = config.create_client()
        client.get_sdk()
        return cls(client)
