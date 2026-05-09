"""Pydantic Valve helpers shared by tool plugins."""

from typing import Any, Type

from pydantic import BaseModel


def coerce_user_valves(raw_valves: Any, valves_cls: Type[BaseModel]) -> BaseModel:
    """Normalize raw user valves into the target valves class.

    Open WebUI hands ``raw_valves`` over from filter context, where it can
    arrive as the target class itself, a different ``BaseModel`` subclass
    (when the user-valve schema has drifted between plugin versions), a raw
    dict, or anything else. Always return a fresh ``valves_cls`` instance so
    callers can rely on the field set being current.
    """
    if isinstance(raw_valves, valves_cls):
        return raw_valves
    if isinstance(raw_valves, BaseModel):
        try:
            data = raw_valves.model_dump()
        except Exception:
            data = {}
        return valves_cls.model_validate(data)
    if isinstance(raw_valves, dict):
        return valves_cls.model_validate(raw_valves)
    return valves_cls.model_validate({})
