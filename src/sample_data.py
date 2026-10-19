"""Artificial campus posts for a reproducible demo, not collected student data."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

SUBJECTS = {
    "study": [
        "Powell library has no seats left during midterms",
        "Finals week study group looking for a quiet room",
        "Anyone else stressed about exams and the problem set",
        "Best late night study spots around campus",
    ],
    "food": [
        "Dining hall dinner review the pasta is great today",
        "De Neve breakfast line is longer than my lecture",
        "Best cheap lunch near Westwood after class",
        "Bruin Plate menu finally has something good",
    ],
    "sports": [
        "Game day in the student section let us go Bruins",
        "Basketball tickets student section is sold out",
        "Football crowd at the Rose Bowl was amazing",
        "Anyone watching the Bruins game this weekend",
    ],
    "housing": [
        "Apartment rent in Westwood is getting out of hand",
        "Roommate search for next quarter near campus",
        "Dorm laundry machines are broken again",
        "Housing waitlist updates anyone hear back",
    ],
    "transport": [
        "Parking permit costs more than my textbooks",
        "Bruin bus route is late again this morning",
        "Bike racks near engineering are full",
        "Commuter train and campus shuttle timing question",
    ],
    "clubs": [
        "Club fair on campus today looking for new friends",
        "Running club meeting at the track this evening",
        "Hackathon project team looking for a designer",
        "Music club open mic night this Friday",
    ],
}
CALENDAR = [
    {"name": "Demo spring term", "start": "2026-03-30", "end": "2026-06-12"},
    {"name": "Demo fall term", "start": "2026-09-21", "end": "2026-12-11"},
]


def generate_sample(path=None, seed=41):
    rng = np.random.default_rng(seed)
    records = []
    start = pd.Timestamp("2026-03-30", tz="America/Los_Angeles")
    subjects = list(SUBJECTS)
    additions = [
        "any tips?",
        "is this just me?",
        "asking for a friend",
        "what do you think?",
        "update from today",
        "honest question",
    ]
    for day in range(203):
        for i in range(int(rng.integers(5, 11))):
            weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            if day % 77 > 55:
                weights[0] = 3.5
            if day > 189:
                weights[2] = 6
            if 175 < day < 190:
                weights[5] = 3
            subject = rng.choice(subjects, p=weights / weights.sum())
            base = rng.choice(SUBJECTS[subject])
            text = f"{base}. {rng.choice(additions)}"
            hour = int(rng.choice([8, 10, 12, 15, 18, 20, 22]))
            stamp = start + pd.Timedelta(
                days=day, hours=hour, minutes=int(rng.integers(0, 60))
            )
            # A noisy artificial relationship, useful for exercising the pipeline only.
            signal = (
                0.55 * (hour >= 18)
                + 0.6 * ("is this just me" in text)
                + 0.4 * (subject == "sports" and day > 189)
            )
            votes = int(max(0, np.exp(rng.normal(2.4 + signal, 1.0)) - 1))
            records.append(
                {
                    "post_id": f"sample-{day:03d}-{i}",
                    "text": text,
                    "upvotes": votes,
                    "timestamp": stamp.tz_convert("UTC").isoformat(),
                    "school": "UCLA",
                    "platform": str(
                        rng.choice(["Reddit-style sample", "Yik Yak-style sample"])
                    ),
                }
            )
    result = {
        "provenance": "synthetic",
        "description": "Seeded fictional text and engagement counts. No Reddit or Yik Yak connection.",
        "calendar": CALENDAR,
        "posts": records,
    }
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(result, indent=2) + "\n")
    return result
