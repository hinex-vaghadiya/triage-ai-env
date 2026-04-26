"""
Hospital resource management for TriageAI.
Tracks beds, doctors, and operating room availability.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Hospital:
    """Manages hospital resources: beds, doctors, and operating room."""
    num_beds: int = 3
    num_doctors: int = 2
    has_or: bool = True

    # State
    beds: List[Optional[str]] = field(default_factory=list)  # patient_id or None
    doctors: List[Optional[str]] = field(default_factory=list)  # patient_id or None
    or_patient: Optional[str] = None
    or_cooldown: int = 0  # Steps until OR is free after surgery
    OR_SURGERY_DURATION: int = 3  # Steps a surgery takes

    def __post_init__(self):
        if not self.beds:
            self.beds = [None] * self.num_beds
        if not self.doctors:
            self.doctors = [None] * self.num_doctors

    def reset(self, num_beds: int = 3, num_doctors: int = 2):
        self.num_beds = num_beds
        self.num_doctors = num_doctors
        self.beds = [None] * num_beds
        self.doctors = [None] * num_doctors
        self.or_patient = None
        self.or_cooldown = 0

    def get_free_bed(self) -> Optional[int]:
        for i, b in enumerate(self.beds):
            if b is None:
                return i
        return None

    def get_free_doctor(self) -> Optional[int]:
        for i, d in enumerate(self.doctors):
            if d is None:
                return i
        return None

    def assign_bed(self, bed_id: int, patient_id: str) -> bool:
        if 0 <= bed_id < len(self.beds) and self.beds[bed_id] is None:
            self.beds[bed_id] = patient_id
            return True
        return False

    def free_bed(self, patient_id: str) -> bool:
        for i, b in enumerate(self.beds):
            if b == patient_id:
                self.beds[i] = None
                return True
        return False

    def assign_doctor(self, doctor_id: int, patient_id: str) -> bool:
        if 0 <= doctor_id < len(self.doctors) and self.doctors[doctor_id] is None:
            self.doctors[doctor_id] = patient_id
            return True
        return False

    def free_doctor_from_patient(self, patient_id: str):
        for i, d in enumerate(self.doctors):
            if d == patient_id:
                self.doctors[i] = None

    def start_surgery(self, patient_id: str) -> bool:
        if self.or_patient is not None or self.or_cooldown > 0:
            return False
        self.or_patient = patient_id
        self.or_cooldown = self.OR_SURGERY_DURATION
        return True

    def tick_or(self) -> Optional[str]:
        """Advance OR timer. Returns patient_id if surgery just completed."""
        if self.or_cooldown > 0:
            self.or_cooldown -= 1
            if self.or_cooldown == 0:
                completed = self.or_patient
                self.or_patient = None
                return completed
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beds": [
                {"bed_id": i, "patient_id": b, "occupied": b is not None}
                for i, b in enumerate(self.beds)
            ],
            "doctors": [
                {"doctor_id": i, "patient_id": d, "busy": d is not None}
                for i, d in enumerate(self.doctors)
            ],
            "or": {
                "available": self.or_patient is None and self.or_cooldown == 0,
                "patient_id": self.or_patient,
                "cooldown_steps": self.or_cooldown,
            },
        }
