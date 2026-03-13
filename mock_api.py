"""
Mock FastAPI server for the Campus Assistant demo.

Run with:
    uvicorn mock_api:app --port 8000

Endpoints:
    GET  /events        – return upcoming campus events
    POST /book-room     – reserve a campus study room
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Lakewood University Mock API", version="1.0.0")

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

EVENTS: list[dict] = [
    {
        "id": "evt-001",
        "title": "Career Fair",
        "date": "2026-04-02",
        "time": "10:00 AM – 3:00 PM",
        "location": "Alpine Student Centre Ballroom",
        "description": "Meet recruiters from 80+ companies. Bring your resume and dress in business attire.",
        "category": "career",
    },
    {
        "id": "evt-002",
        "title": "Research Symposium",
        "date": "2026-04-10",
        "time": "9:00 AM – 5:00 PM",
        "location": "Ridgeline Science Complex",
        "description": "Student and faculty research presentations across all disciplines.",
        "category": "academic",
    },
    {
        "id": "evt-003",
        "title": "Spring Concert",
        "date": "2026-04-18",
        "time": "7:00 PM",
        "location": "Morton Arts Centre Main Stage",
        "description": "Annual spring concert featuring the University Symphony and guest artists.",
        "category": "arts",
    },
    {
        "id": "evt-004",
        "title": "Finals Week Wellness Fair",
        "date": "2026-05-01",
        "time": "11:00 AM – 2:00 PM",
        "location": "The Commons Patio",
        "description": "De-stress with free snacks, massages, therapy dogs, and wellness resources.",
        "category": "wellness",
    },
    {
        "id": "evt-005",
        "title": "Convocation Ceremony",
        "date": "2026-05-15",
        "time": "10:00 AM",
        "location": "Foothills Stadium",
        "description": "University-wide convocation ceremony for graduating students.",
        "category": "academic",
    },
]

# Track bookings in-memory (resets on server restart)
_bookings: list[dict] = []
_booking_counter = 0


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class BookRoomRequest(BaseModel):
    room_type: str  # e.g. "hartwell", "innovation_hub", "student_union"
    date: str       # ISO date string, e.g. "2026-04-05"
    start_time: str  # e.g. "14:00"
    duration_hours: int  # 1, 2, or 3
    name: str        # booker's name
    purpose: Optional[str] = None


class BookRoomResponse(BaseModel):
    success: bool
    booking_id: str
    message: str
    details: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOM_CAPACITY: dict[str, str] = {
    "hartwell": "2–8 person capacity",
    "innovation_hub": "4–12 person capacity",
    "student_union": "up to 20 person capacity",
}

ROOM_DISPLAY: dict[str, str] = {
    "hartwell": "Hartwell Hall Study Rooms",
    "innovation_hub": "Innovation Hub Collaboration Rooms",
    "student_union": "Student Centre Meeting Rooms",
}


def _is_valid_room(room_type: str) -> bool:
    return room_type.lower() in ROOM_CAPACITY


def _is_valid_duration(hours: int) -> bool:
    return 1 <= hours <= 3


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/events")
def get_events(category: Optional[str] = None) -> dict:
    """Return upcoming campus events, optionally filtered by category."""
    results = EVENTS
    if category:
        results = [e for e in EVENTS if e["category"].lower() == category.lower()]
    return {"events": results, "count": len(results)}


@app.post("/book-room", response_model=BookRoomResponse)
def book_room(req: BookRoomRequest) -> BookRoomResponse:
    """Reserve a campus study / meeting room."""
    global _booking_counter

    room_key = req.room_type.lower().replace(" ", "_").replace("-", "_")
    if not _is_valid_room(room_key):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown room type '{req.room_type}'. "
                   f"Valid options: {', '.join(ROOM_CAPACITY.keys())}",
        )

    if not _is_valid_duration(req.duration_hours):
        raise HTTPException(
            status_code=400,
            detail="duration_hours must be 1, 2, or 3.",
        )

    # Simple conflict check: same room + same date + overlapping start time
    for existing in _bookings:
        if (
            existing["room_key"] == room_key
            and existing["date"] == req.date
            and existing["start_time"] == req.start_time
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"That time slot is already booked. "
                    f"Try a different time or room."
                ),
            )

    _booking_counter += 1
    booking_id = f"BK-{_booking_counter:04d}"
    booking = {
        "booking_id": booking_id,
        "room_key": room_key,
        "room_name": ROOM_DISPLAY[room_key],
        "date": req.date,
        "start_time": req.start_time,
        "duration_hours": req.duration_hours,
        "name": req.name,
        "purpose": req.purpose or "General study",
        "capacity": ROOM_CAPACITY[room_key],
        "booked_at": datetime.utcnow().isoformat() + "Z",
    }
    _bookings.append(booking)

    return BookRoomResponse(
        success=True,
        booking_id=booking_id,
        message=(
            f"Room booked! Your confirmation ID is {booking_id}. "
            f"{ROOM_DISPLAY[room_key]} reserved for {req.name} on {req.date} "
            f"at {req.start_time} for {req.duration_hours} hour(s)."
        ),
        details=booking,
    )


@app.get("/bookings")
def list_bookings() -> dict:
    """Return all current bookings (useful for demo/debugging)."""
    return {"bookings": _bookings, "count": len(_bookings)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
