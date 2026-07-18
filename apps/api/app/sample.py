from __future__ import annotations

from math import sin


def _keyframes(index: int, team: str, duration: float = 12.0) -> list[dict]:
    frames: list[dict] = []
    side = -1 if team == "aurora" else 1
    base_x = side * (9 + (index % 4) * 8)
    base_z = -23 + (index % 6) * 9
    for step in range(7):
        t = step * 2.0
        attack = (t / duration) * 23 * side
        frames.append(
            {
                "t": t,
                "x": round(base_x + attack + sin(step + index) * 1.6, 2),
                "z": round(base_z + sin(step * 0.75 + index * 0.4) * 3.2, 2),
                "confidence": round(0.9 + 0.08 * sin(index + step), 2),
            }
        )
    return frames


def make_demo_scene(scene_id: str = "moment-01", title: str = "The impossible passing lane") -> dict:
    home_names = ["N. Vale", "M. Kovac", "I. Sato", "L. Costa", "A. Noor", "J. Silva"]
    away_names = ["T. Frost", "C. Mensah", "R. Marin", "E. Park", "D. Arno", "S. Diallo"]
    tracks = []
    for team, names, color in (
        ("aurora", home_names, "#ff5f4a"),
        ("atlas", away_names, "#6ee7f2"),
    ):
        for index, name in enumerate(names):
            tracks.append(
                {
                    "id": f"{team}-{index + 1}",
                    "label": name,
                    "teamId": team,
                    "color": color,
                    "number": index + 7,
                    "externalPlayerId": None,
                    "keyframes": _keyframes(index, team),
                }
            )

    ball = []
    points = [(-23, -8, 0.22), (-15, -4, 0.22), (-5, 3, 0.35), (8, 8, 2.6), (22, 4, 0.3), (34, -2, 0.22), (45, -1, 0.22)]
    for step, (x, z, y) in enumerate(points):
        ball.append({"t": step * 2.0, "x": x, "y": y, "z": z, "confidence": 0.96})

    payload = {
        "pitch": {"length": 105, "width": 68},
        "teams": [
            {"id": "aurora", "name": "Aurora", "color": "#ff5f4a", "externalTeamId": None},
            {"id": "atlas", "name": "Atlas", "color": "#6ee7f2", "externalTeamId": None},
        ],
        "tracks": tracks,
        "ball": {"keyframes": ball},
        "eventBindings": [],
        "cameraCuts": [
            {"t": 0, "preset": "broadcast"},
            {"t": 4.2, "preset": "orbit"},
            {"t": 8.3, "preset": "tactical"},
        ],
    }
    return {"id": scene_id, "title": title, "version": 1, "duration": 12.0, "payload": payload}


def make_video_scene(
    scene_id: str,
    title: str,
    duration: float,
    video_asset: dict,
) -> dict:
    payload = {
        "pitch": {"length": 105, "width": 68},
        "videoAsset": video_asset,
        "teams": [
            {"id": "home", "name": "Home", "color": "#ff5f4a", "externalTeamId": None},
            {"id": "away", "name": "Away", "color": "#6ee7f2", "externalTeamId": None},
        ],
        "tracks": [],
        "ball": {"keyframes": []},
        "eventBindings": [],
        "cameraCuts": [{"t": 0, "preset": "broadcast"}],
    }
    return {
        "id": scene_id,
        "title": title,
        "version": 1,
        "duration": max(0.1, round(duration, 3)),
        "payload": payload,
    }
