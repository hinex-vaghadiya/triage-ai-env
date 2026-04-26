"""
Pydantic/dataclass models for TriageAI.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class TriageAction:
    """Action the agent can take in the ER."""
    action_type: str  # triage|assign_bed|assign_doctor|order_treatment|send_to_or|discharge|reassess|submit
    patient_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriageObservation:
    """What the agent sees after each step."""
    task_id: str = ""
    task_description: str = ""
    current_step: int = 0
    max_steps: int = 20
    hospital: Dict[str, Any] = field(default_factory=dict)
    waiting_patients: List[Dict] = field(default_factory=list)
    admitted_patients: List[Dict] = field(default_factory=list)
    resolved_patients: List[Dict] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    last_action_success: bool = True
    last_action_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriageState:
    """Full environment state (for debugging/display)."""
    episode_id: str = ""
    task_id: str = ""
    step_count: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    composite_score: float = 0.0
    survival_rate: float = 1.0
    deaths: int = 0
    patients_treated: int = 0
