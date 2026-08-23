"""Consistent API error response helpers."""

from fastapi import HTTPException


class NotFoundError(HTTPException):
    def __init__(self, resource: str, resource_id: str) -> None:
        super().__init__(
            status_code=404,
            detail={"code": "not_found", "message": f"{resource} '{resource_id}' was not found"},
        )
