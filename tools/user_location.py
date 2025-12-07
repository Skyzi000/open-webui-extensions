"""
title: User Location
author: https://github.com/skyzi000
version: 0.1.0
license: MIT

Get user's current location using the browser's Geolocation API.
This tool uses __event_call__ with type "execute" to run JavaScript
in the user's browser and retrieve their location coordinates.

Note: The user must grant location permission in their browser.
"""

import asyncio
import json
from typing import Any, Callable

from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):  # type: ignore
        self.event_emitter = event_emitter

    async def emit(
        self, description="Unknown state", status="in_progress", done=False
    ):
        """
        Send a status event to the event emitter.

        :param description: Event description
        :param status: Event status
        :param done: Whether the event is complete
        """
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    """
    User Location

    Use this tool to get the user's current geographical location using
    the browser's Geolocation API. Returns latitude, longitude, and accuracy.

    Important notes:
    - Requires user permission: The browser will prompt the user to allow
      location access. If denied, the tool will return an error.
    - Accuracy varies: GPS-enabled devices provide higher accuracy than
      IP-based location.
    - Use cases: Weather information, local search, timezone detection,
      location-based recommendations.
    """

    class Valves(BaseModel):
        TIMEOUT: int = Field(
            default=30,
            description="Geolocation request timeout in seconds (default: 30).",
        )
        ENABLE_HIGH_ACCURACY: bool = Field(
            default=True,
            description="Request high accuracy location (may use more battery on mobile).",
        )
        MAXIMUM_AGE: int = Field(
            default=60000,
            description="Maximum age of cached position in milliseconds (default: 60000 = 1 minute).",
        )
        pass

    def __init__(self):
        """Initialize the user location tool."""
        self.valves = self.Valves()

    async def get_current_location(
        self,
        __user__: dict = None,  # type: ignore
        __event_emitter__: Callable[[dict], Any] = None,  # type: ignore
        __event_call__: Callable[[dict], Any] = None,  # type: ignore
    ) -> str:
        """
        Get the user's current geographical location using the browser's
        Geolocation API.

        This method executes JavaScript in the user's browser to access
        navigator.geolocation.getCurrentPosition().

        The user's browser will show a permission dialog if location access
        hasn't been granted yet. The user must approve to receive location data.

        Returns a JSON object with:
        - latitude: Decimal degrees (positive = North, negative = South)
        - longitude: Decimal degrees (positive = East, negative = West)
        - accuracy: Accuracy in meters
        - altitude: Altitude in meters (null if unavailable)
        - altitudeAccuracy: Altitude accuracy in meters (null if unavailable)
        - heading: Direction of travel in degrees (null if unavailable)
        - speed: Speed in meters per second (null if unavailable)

        Error cases:
        - PERMISSION_DENIED: User denied location access
        - POSITION_UNAVAILABLE: Location information unavailable
        - TIMEOUT: Request timed out
        - GEOLOCATION_NOT_SUPPORTED: Browser doesn't support Geolocation API
        - EVENT_CALL_NOT_AVAILABLE: __event_call__ is not available

        :return: JSON string with location data or error information
        """
        emitter = EventEmitter(__event_emitter__)

        # Check if __event_call__ is available
        if not __event_call__:
            error_msg = "__event_call__が利用できません。この機能はブラウザ経由でのみ使用できます。"
            await emitter.emit(
                description=error_msg,
                status="event_call_not_available",
                done=True,
            )
            return json.dumps(
                {
                    "error": "EVENT_CALL_NOT_AVAILABLE",
                    "message": error_msg,
                },
                ensure_ascii=False,
            )

        await emitter.emit(
            description="ブラウザから位置情報を取得中...",
            status="getting_location",
            done=False,
        )

        # JavaScript code to execute in the browser
        js_code = f"""
        try {{
            if (!navigator.geolocation) {{
                return {{
                    error: "GEOLOCATION_NOT_SUPPORTED",
                    message: "このブラウザは位置情報APIをサポートしていません。"
                }};
            }}

            const position = await new Promise((resolve, reject) => {{
                navigator.geolocation.getCurrentPosition(
                    resolve,
                    reject,
                    {{
                        enableHighAccuracy: {str(self.valves.ENABLE_HIGH_ACCURACY).lower()},
                        timeout: {self.valves.TIMEOUT * 1000},
                        maximumAge: {self.valves.MAXIMUM_AGE}
                    }}
                );
            }});

            return {{
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                accuracy: position.coords.accuracy,
                altitude: position.coords.altitude,
                altitudeAccuracy: position.coords.altitudeAccuracy,
                heading: position.coords.heading,
                speed: position.coords.speed
            }};
        }} catch (error) {{
            const errorMessages = {{
                1: {{ code: "PERMISSION_DENIED", message: "ユーザーが位置情報へのアクセスを拒否しました。" }},
                2: {{ code: "POSITION_UNAVAILABLE", message: "位置情報を取得できませんでした。" }},
                3: {{ code: "TIMEOUT", message: "位置情報の取得がタイムアウトしました。" }}
            }};

            const errorInfo = errorMessages[error.code] || {{
                code: "UNKNOWN_ERROR",
                message: error.message || "不明なエラーが発生しました。"
            }};

            return {{
                error: errorInfo.code,
                message: errorInfo.message
            }};
        }}
        """

        try:
            # Execute JavaScript in the browser with timeout
            result_task = __event_call__(
                {
                    "type": "execute",
                    "data": {
                        "code": js_code,
                    },
                }
            )

            try:
                result = await asyncio.wait_for(
                    result_task, timeout=self.valves.TIMEOUT + 5
                )
            except asyncio.TimeoutError:
                timeout_msg = f"位置情報の取得が{self.valves.TIMEOUT}秒以内に完了しませんでした。"
                await emitter.emit(
                    description=timeout_msg,
                    status="timeout",
                    done=True,
                )
                return json.dumps(
                    {
                        "error": "TIMEOUT",
                        "message": timeout_msg,
                    },
                    ensure_ascii=False,
                )

            # Check if result contains an error
            if result and isinstance(result, dict) and "error" in result:
                await emitter.emit(
                    description=result.get("message", "位置情報の取得に失敗しました。"),
                    status=result.get("error", "error").lower(),
                    done=True,
                )
                return json.dumps(result, ensure_ascii=False)

            # Success
            if result and isinstance(result, dict) and "latitude" in result:
                await emitter.emit(
                    description=f"位置情報を取得しました (精度: {result.get('accuracy', 'N/A')}m)",
                    status="success",
                    done=True,
                )
                return json.dumps(result, ensure_ascii=False)

            # Unexpected result format
            unexpected_msg = "予期しない形式のレスポンスを受け取りました。"
            await emitter.emit(
                description=unexpected_msg,
                status="unexpected_response",
                done=True,
            )
            return json.dumps(
                {
                    "error": "UNEXPECTED_RESPONSE",
                    "message": unexpected_msg,
                    "raw_result": str(result),
                },
                ensure_ascii=False,
            )

        except Exception as e:
            error_msg = f"位置情報の取得中にエラーが発生しました: {str(e)}"
            await emitter.emit(
                description=error_msg,
                status="error",
                done=True,
            )
            return json.dumps(
                {
                    "error": "EXECUTION_ERROR",
                    "message": error_msg,
                },
                ensure_ascii=False,
            )
