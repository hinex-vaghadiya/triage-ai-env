"""
TriageAI Environment — OpenEnv-compliant ER Crisis Simulator.
Implements reset(), step(), and state property.
"""

import random
import uuid
import math
from typing import Optional, Dict, List, Any

try:
    from openenv.core.env_server.interfaces import Environment as _BaseEnv
except ImportError:
    # Fallback if openenv-core not installed locally
    class _BaseEnv:
        def __init__(self, **kwargs): pass

from .patients import Patient, generate_patients
from .hospital import Hospital


class Environment(_BaseEnv):
    """Base environment with state property so subclass isn't abstract."""
    @property
    def state(self) -> Dict[str, Any]:
        return {}


# Task configs for 3 difficulty levels (curriculum)
TASK_CONFIGS = {
    "task_easy": {
        "name": "Basic ER Triage",
        "description": (
            "A calm ER shift with 4 patients. 3 beds, 2 doctors, 1 OR. "
            "Patients have clear symptoms. Triage them, assign resources, and treat. "
            "Goal: Keep all patients alive and treat them appropriately."
        ),
        "num_patients": 4, "num_beds": 3, "num_doctors": 2,
        "max_steps": 20, "difficulty": "easy",
    },
    "task_medium": {
        "name": "Busy ER Night",
        "description": (
            "A busy night shift with 7 patients flooding in. 3 beds, 2 doctors, 1 OR. "
            "Some patients have ambiguous symptoms. One hidden critical case may fool you. "
            "Patients deteriorate faster. Goal: Prioritize correctly under pressure."
        ),
        "num_patients": 7, "num_beds": 3, "num_doctors": 2,
        "max_steps": 30, "difficulty": "medium",
    },
    "task_hard": {
        "name": "Mass Casualty Event",
        "description": (
            "A mass casualty event — 10 patients arrive simultaneously. Only 3 beds, "
            "2 doctors, 1 OR. Multiple critical cases competing for limited resources. "
            "Patients deteriorate rapidly. Every decision matters. Goal: Maximize survival."
        ),
        "num_patients": 10, "num_beds": 3, "num_doctors": 2,
        "max_steps": 40, "difficulty": "hard",
    },
}


class TriageEnvironment(Environment):
    """OpenEnv-compliant Emergency Room Triage environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._patients: List[Patient] = []
        self._hospital = Hospital()
        self._task_id = "task_easy"
        self._task_config: Dict = TASK_CONFIGS["task_easy"]
        self._step_count = 0
        self._max_steps = 20
        self._done = False
        self._episode_id = ""
        self._last_action_msg = ""
        self._last_action_success = True
        self._reward = 0.0
        self._cumulative_reward = 0.0
        self._actions_taken: List[Dict] = []
        self._deaths_this_episode: List[str] = []
        self._treatments_given: int = 0

    @property
    def state(self) -> Dict[str, Any]:
        scores = self._compute_scores()
        return {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "scores": scores,
        }

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if seed is None:
            seed = random.randint(0, 2**31)

        self._task_id = task_id or kwargs.get("task_id", "task_easy")
        if self._task_id not in TASK_CONFIGS:
            self._task_id = "task_easy"

        self._task_config = TASK_CONFIGS[self._task_id]
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._max_steps = self._task_config["max_steps"]
        self._done = False
        self._reward = 0.0
        self._cumulative_reward = 0.0
        self._last_action_msg = "ER is open. Patients are arriving. Begin triage and treatment."
        self._last_action_success = True
        self._actions_taken = []
        self._deaths_this_episode = []
        self._treatments_given = 0

        rng = random.Random(seed)
        self._patients = generate_patients(
            rng, self._task_config["num_patients"], self._task_config["difficulty"]
        )
        self._hospital.reset(
            self._task_config["num_beds"], self._task_config["num_doctors"]
        )

        return self._build_observation()

    def step(self, action: Dict[str, Any], timeout_s: Optional[float] = None,
             **kwargs) -> Dict[str, Any]:
        if self._done:
            return self._build_observation()

        self._step_count += 1
        step_reward = 0.0

        # Parse action
        if isinstance(action, dict):
            action_type = action.get("action_type", "")
            patient_id = action.get("patient_id")
            params = action.get("params", {})
        else:
            action_type = ""
            patient_id = None
            params = {}

        try:
            if action_type == "triage":
                step_reward = self._action_triage(patient_id)
            elif action_type == "assign_bed":
                step_reward = self._action_assign_bed(patient_id)
            elif action_type == "assign_doctor":
                step_reward = self._action_assign_doctor(patient_id, params)
            elif action_type == "order_treatment":
                step_reward = self._action_order_treatment(patient_id, params)
            elif action_type == "send_to_or":
                step_reward = self._action_send_to_or(patient_id)
            elif action_type == "discharge":
                step_reward = self._action_discharge(patient_id)
            elif action_type == "reassess":
                step_reward = self._action_reassess(patient_id)
            elif action_type == "submit":
                self._done = True
                self._last_action_msg = "Episode submitted for final grading."
                self._last_action_success = True
            else:
                self._last_action_success = False
                self._last_action_msg = f"Unknown action: {action_type}. Valid: triage, assign_bed, assign_doctor, order_treatment, send_to_or, discharge, reassess, submit"
                step_reward = -0.01
        except Exception as e:
            self._last_action_success = False
            self._last_action_msg = f"Action error: {str(e)}"
            step_reward = -0.01

        self._actions_taken.append({"step": self._step_count, "action": action_type, "patient": patient_id})

        # Advance time: deteriorate patients, tick OR
        surgery_done = self._hospital.tick_or()
        if surgery_done:
            p = self._get_patient(surgery_done)
            if p and p.status != "dead":
                p.status = "treated"
                p.treatment_given = "surgery"
                p.current_severity = min(5, p.current_severity + 2)
                self._treatments_given += 1
                self._hospital.free_doctor_from_patient(p.id)
                step_reward += 0.05

        # Deterioration check
        for p in self._patients:
            if p.status in ("waiting", "in_bed") and p.status != "dead":
                p.steps_waiting += 1
                p.steps_since_last_deterioration += 1
                if p.steps_since_last_deterioration >= p.time_to_deteriorate:
                    p.steps_since_last_deterioration = 0
                    died = p.deteriorate()
                    if died:
                        self._deaths_this_episode.append(p.id)
                        self._hospital.free_bed(p.id)
                        self._hospital.free_doctor_from_patient(p.id)
                        step_reward -= 0.10

        # Check end conditions
        all_resolved = all(
            p.status in ("treated", "discharged", "dead") for p in self._patients
        )
        if all_resolved or self._step_count >= self._max_steps:
            self._done = True

        # Compute step reward
        self._reward = step_reward
        self._cumulative_reward += step_reward

        return self._build_observation()

    def _get_patient(self, patient_id: str) -> Optional[Patient]:
        for p in self._patients:
            if p.id == patient_id:
                return p
        return None

    # =====================================================================
    # Actions
    # =====================================================================

    def _action_triage(self, patient_id: str) -> float:
        p = self._get_patient(patient_id)
        if not p:
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} not found."
            return -0.01

        if p.status == "dead":
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} is deceased. Cannot triage."
            return -0.02

        p.triage_count += 1
        if p.triage_count > 2:
            self._last_action_success = True
            self._last_action_msg = f"Patient {patient_id} already triaged {p.triage_count} times. Diminishing returns."
            return -0.01

        # Reveal triage level (estimated severity based on symptoms)
        # Add some noise: agent's triage may be off by ±1
        noise = random.choice([-1, 0, 0, 0, 1])
        p.triage_level = max(1, min(5, p.current_severity + noise))

        severity_hint = {1: "CRITICAL - Immediate", 2: "URGENT - Very Soon", 3: "SEMI-URGENT", 4: "LESS URGENT", 5: "NON-URGENT"}
        self._last_action_success = True
        self._last_action_msg = (
            f"Triaged {p.name} ({patient_id}): Level {p.triage_level} - "
            f"{severity_hint.get(p.triage_level, 'Unknown')}. "
            f"Symptoms: {p.symptoms}"
        )

        reward = 0.02 if p.current_severity <= 2 else 0.01
        return reward

    def _action_assign_bed(self, patient_id: str) -> float:
        p = self._get_patient(patient_id)
        if not p:
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} not found."
            return -0.01
        if p.status != "waiting":
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} is not waiting (status: {p.status})."
            return -0.01

        bed_id = self._hospital.get_free_bed()
        if bed_id is None:
            self._last_action_success = False
            self._last_action_msg = "No beds available. Consider discharging a stable patient."
            return -0.01

        self._hospital.assign_bed(bed_id, patient_id)
        p.assigned_bed = bed_id
        p.status = "in_bed"
        self._last_action_success = True
        self._last_action_msg = f"Assigned {p.name} ({patient_id}) to Bed {bed_id}."

        return 0.03 if p.current_severity <= 2 else 0.01

    def _action_assign_doctor(self, patient_id: str, params: Dict) -> float:
        p = self._get_patient(patient_id)
        if not p:
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} not found."
            return -0.01
        if p.status not in ("waiting", "in_bed"):
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} cannot be examined (status: {p.status})."
            return -0.01

        doctor_id = params.get("doctor_id")
        if doctor_id is None:
            doctor_id = self._hospital.get_free_doctor()
        else:
            doctor_id = int(doctor_id)

        if doctor_id is None:
            self._last_action_success = False
            self._last_action_msg = "No doctors available right now."
            return -0.01

        if not self._hospital.assign_doctor(doctor_id, patient_id):
            self._last_action_success = False
            self._last_action_msg = f"Doctor {doctor_id} is not available."
            return -0.01

        p.assigned_doctor = doctor_id
        p.examined = True
        self._last_action_success = True
        exam_notes = p._get_exam_notes()
        self._last_action_msg = f"Doctor {doctor_id} examining {p.name} ({patient_id}). {exam_notes}"

        return 0.02

    def _action_order_treatment(self, patient_id: str, params: Dict) -> float:
        p = self._get_patient(patient_id)
        if not p:
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} not found."
            return -0.01
        if p.status not in ("in_bed",):
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} must be in a bed to receive treatment."
            return -0.01

        treatment = params.get("treatment", "medication")
        valid = {"medication", "iv_fluids", "oxygen", "monitoring"}
        if treatment not in valid:
            self._last_action_success = False
            self._last_action_msg = f"Invalid treatment '{treatment}'. Valid: {valid}"
            return -0.01

        # Check if treatment matches
        correct = (p.required_treatment in ("medication", "observation"))
        p.treatment_given = treatment
        p.status = "treated"
        p.current_severity = min(5, p.current_severity + 1)
        self._treatments_given += 1
        self._hospital.free_doctor_from_patient(p.id)

        self._last_action_success = True
        self._last_action_msg = f"Administered {treatment} to {p.name} ({patient_id}). Patient stabilizing."

        return 0.05 if correct else 0.02

    def _action_send_to_or(self, patient_id: str) -> float:
        p = self._get_patient(patient_id)
        if not p:
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} not found."
            return -0.01
        if p.status != "in_bed":
            self._last_action_success = False
            self._last_action_msg = f"Patient must be in a bed before surgery."
            return -0.01

        if not self._hospital.start_surgery(patient_id):
            self._last_action_success = False
            self._last_action_msg = "Operating room not available. Surgery in progress or on cooldown."
            return -0.01

        p.status = "in_or"
        self._hospital.free_bed(patient_id)
        self._last_action_success = True
        self._last_action_msg = f"Sending {p.name} ({patient_id}) to Operating Room. Surgery will take {self._hospital.OR_SURGERY_DURATION} steps."

        correct = (p.required_treatment == "surgery")
        return 0.05 if correct else -0.02

    def _action_discharge(self, patient_id: str) -> float:
        p = self._get_patient(patient_id)
        if not p:
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} not found."
            return -0.01
        if p.status not in ("in_bed", "treated", "waiting"):
            self._last_action_success = False
            self._last_action_msg = f"Cannot discharge patient with status: {p.status}."
            return -0.01
        if p.current_severity <= 2:
            self._last_action_success = False
            self._last_action_msg = f"Cannot discharge {p.name} — patient is too unstable (severity {p.current_severity})."
            return -0.03

        self._hospital.free_bed(p.id)
        self._hospital.free_doctor_from_patient(p.id)
        p.status = "discharged"
        self._last_action_success = True
        self._last_action_msg = f"Discharged {p.name} ({patient_id}). Bed freed."

        correct = (p.required_treatment == "discharge")
        return 0.03 if correct else 0.01

    def _action_reassess(self, patient_id: str) -> float:
        p = self._get_patient(patient_id)
        if not p:
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} not found."
            return -0.01
        if p.status == "dead":
            self._last_action_success = False
            self._last_action_msg = f"Patient {patient_id} is deceased."
            return -0.02

        p._update_vitals_for_severity()
        severity_desc = {1: "CRITICAL", 2: "URGENT", 3: "CONCERNING", 4: "STABLE", 5: "GOOD"}
        self._last_action_success = True
        self._last_action_msg = (
            f"Reassessed {p.name} ({patient_id}): Status {severity_desc.get(p.current_severity, '?')}. "
            f"Vitals: HR={p.vitals.get('hr')}, BP={p.vitals.get('bp')}, SpO2={p.vitals.get('spo2')}%, "
            f"Temp={p.vitals.get('temp')}C, Consciousness={p.vitals.get('consciousness')}"
        )
        return 0.0

    # =====================================================================
    # Observation Builder
    # =====================================================================

    def _build_observation(self) -> Dict[str, Any]:
        waiting = []
        admitted = []
        resolved = []

        for p in self._patients:
            if p.status == "waiting":
                detail = "triaged" if p.triage_level else "basic"
                waiting.append(p.to_visible_dict(detail))
            elif p.status in ("in_bed", "in_or"):
                detail = "examined" if p.examined else ("triaged" if p.triage_level else "basic")
                admitted.append(p.to_visible_dict(detail))
            else:
                resolved.append({"id": p.id, "name": p.name, "status": p.status})

        alive = sum(1 for p in self._patients if p.status != "dead")
        dead = sum(1 for p in self._patients if p.status == "dead")
        treated = sum(1 for p in self._patients if p.status in ("treated", "discharged"))
        total = len(self._patients)

        scores = self._compute_scores()

        return {
            "observation": {
                "task_id": self._task_id,
                "task_description": self._task_config["description"],
                "current_step": self._step_count,
                "max_steps": self._max_steps,
                "hospital": self._hospital.to_dict(),
                "waiting_patients": waiting,
                "admitted_patients": admitted,
                "resolved_patients": resolved,
                "summary": {
                    "total_patients": total,
                    "alive": alive,
                    "dead": dead,
                    "treated": treated,
                    "waiting": len(waiting),
                    "in_beds": sum(1 for p in self._patients if p.status == "in_bed"),
                    "in_or": sum(1 for p in self._patients if p.status == "in_or"),
                },
                "last_action_success": self._last_action_success,
                "last_action_message": self._last_action_msg,
                "metadata": scores,
            },
            "reward": self._reward,
            "done": self._done,
        }

    def _compute_scores(self) -> Dict[str, float]:
        total = len(self._patients)
        if total == 0:
            return {"survival_rate": 1.0, "composite_score": 0.0}

        alive = sum(1 for p in self._patients if p.status != "dead")
        survival = alive / total

        # Triage accuracy
        triaged = [p for p in self._patients if p.triage_level is not None]
        if triaged:
            triage_acc = sum(
                1 for p in triaged if abs(p.triage_level - p.hidden_severity) <= 1
            ) / len(triaged)
        else:
            triage_acc = 0.0

        # Treatment quality
        treated = [p for p in self._patients if p.treatment_given is not None]
        if treated:
            correct_tx = sum(1 for p in treated if self._treatment_matches(p)) / len(treated)
        else:
            correct_tx = 0.0

        # Time efficiency for critical patients
        critical = [p for p in self._patients if p.hidden_severity <= 2]
        if critical:
            avg_wait = sum(p.steps_waiting for p in critical) / len(critical)
            time_eff = max(0, 1.0 - avg_wait / max(self._max_steps, 1))
        else:
            time_eff = 1.0

        # Resource utilization
        total_slots = self._hospital.num_beds * max(self._step_count, 1)
        used_steps = sum(
            p.steps_waiting for p in self._patients if p.status != "waiting" or p.assigned_bed is not None
        )
        resource_util = min(1.0, used_steps / max(total_slots, 1))

        composite = (
            0.35 * survival +
            0.20 * triage_acc +
            0.15 * correct_tx +
            0.15 * time_eff +
            0.10 * resource_util +
            0.05 * (1.0 if self._step_count > 3 else 0.0)
        )
        composite = max(0.0, min(1.0, composite))

        return {
            "survival_rate": round(survival, 4),
            "triage_accuracy": round(triage_acc, 4),
            "treatment_quality": round(correct_tx, 4),
            "time_efficiency": round(time_eff, 4),
            "resource_utilization": round(resource_util, 4),
            "composite_score": round(composite, 4),
            "episode_id": self._episode_id,
            "deaths": len(self._deaths_this_episode),
        }

    def _treatment_matches(self, patient: Patient) -> bool:
        req = patient.required_treatment
        given = patient.treatment_given
        if req == "surgery" and given == "surgery":
            return True
        if req == "medication" and given in ("medication", "iv_fluids", "oxygen"):
            return True
        if req == "observation" and given in ("monitoring", "observation", "medication"):
            return True
        if req == "discharge" and patient.status == "discharged":
            return True
        return False
