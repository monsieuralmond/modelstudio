from __future__ import annotations

from .config import Settings
from .models import VesslConfigRequest, VesslConfigStatus


class VesslConfigService:
    def __init__(self, settings: Settings) -> None:
        self._access_token = settings.vessl_access_token
        self._organization_name = settings.vessl_default_organization
        self._project_name = settings.vessl_default_project

    def set_config(self, payload: VesslConfigRequest) -> VesslConfigStatus:
        self._access_token = payload.access_token.strip()
        self._organization_name = payload.organization_name.strip()
        self._project_name = payload.project_name.strip()
        return self.status()

    def status(self) -> VesslConfigStatus:
        return VesslConfigStatus(
            configured=bool(self._access_token),
            organization_name=self._organization_name,
            project_name=self._project_name,
        )
