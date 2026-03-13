"""
MCP Server — Lakewood University Campus Tools
==============================================

Exposes three tools over stdio MCP transport:
  - campus_info  : look up the campus guide knowledge base
  - get_events   : fetch upcoming events from the mock API
  - book_room    : reserve a study / meeting room via the mock API

Run standalone (for testing):
    python mcp_server.py

Normally spawned automatically as a subprocess by app_mcp.py.

Requires the mock API for get_events / book_room:
    uvicorn mock_api:app --port 8000
"""

from __future__ import annotations

import json
import os

import requests
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mcp = FastMCP("campus-assistant")

CAMPUS_GUIDE_PATH = os.path.join(os.path.dirname(__file__), "campus_guide.txt")
MOCK_API_BASE = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def campus_info(query: str) -> str:
    """
    Search the Lakewood University campus guide for information about campus
    hours, buildings, dining, student services, transportation, policies, and
    other campus facilities.

    Args:
        query: The topic or question to look up, e.g. 'library hours',
               'dining options', 'parking'
    """
    with open(CAMPUS_GUIDE_PATH, encoding="utf-8") as fh:
        return fh.read()


@mcp.tool()
def get_events(category: str = "") -> str:
    """
    Retrieve upcoming campus events from the live events API.
    Optionally filter by category.
    Available categories: career, academic, arts, wellness, tech.

    Args:
        category: Optional event category to filter by: career, academic,
                  arts, wellness, or tech. Leave empty for all events.
    """
    params: dict = {}
    if category:
        params["category"] = category
    try:
        resp = requests.get(f"{MOCK_API_BASE}/events", params=params, timeout=5)
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except requests.ConnectionError:
        return (
            "Error: Could not connect to the mock API. "
            "Make sure it is running with: uvicorn mock_api:app --port 8000"
        )
    except requests.HTTPError as exc:
        return f"Error from events API: {exc}"


@mcp.tool()
def book_room(
    room_type: str,
    date: str,
    start_time: str,
    duration_hours: int,
    name: str,
    purpose: str = "",
) -> str:
    """
    Reserve a campus study or meeting room.
    Valid room types: hartwell (2-8 people), innovation_hub (4-12 people),
    student_union (up to 20 people).
    Bookings are 1-3 hours; up to 7 days in advance.

    Args:
        room_type: The type of room to book: 'hartwell', 'innovation_hub',
                   or 'student_union'
        date: Booking date in ISO format, e.g. '2026-04-05'
        start_time: Start time in 24-hour format, e.g. '14:00'
        duration_hours: Duration in whole hours: 1, 2, or 3
        name: Full name of the person making the booking
        purpose: Optional brief description of the booking purpose
    """
    payload: dict = {
        "room_type": room_type,
        "date": date,
        "start_time": start_time,
        "duration_hours": duration_hours,
        "name": name,
    }
    if purpose:
        payload["purpose"] = purpose
    try:
        resp = requests.post(f"{MOCK_API_BASE}/book-room", json=payload, timeout=5)
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except requests.ConnectionError:
        return (
            "Error: Could not connect to the mock API. "
            "Make sure it is running with: uvicorn mock_api:app --port 8000"
        )
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return f"Booking failed: {detail}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
