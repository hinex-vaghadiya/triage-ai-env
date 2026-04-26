"""
Expert Agent for TriageAI — Generates SFT Training Data
========================================================
Plays the environment using an optimal rule-based strategy
and saves (observation, action) pairs as JSONL for fine-tuning.
"""

import json
import random
import time
import requests
import os
from typing import List, Dict, Any, Optional

ENV_URL = os.getenv("ENV_URL", "https://hinex-07-triage-ai-env.hf.space")

SYSTEM_PROMPT = """You are an expert Emergency Room triage physician AI. You manage a busy ER with limited beds, doctors, and one operating room.

GOAL: Save as many patients as possible by triaging, assigning resources, and treating efficiently.

ACTIONS (respond with EXACTLY one JSON object, nothing else):
1. {"action_type": "triage", "patient_id": "P001"}
2. {"action_type": "assign_bed", "patient_id": "P001"}
3. {"action_type": "assign_doctor", "patient_id": "P001", "params": {"doctor_id": 0}}
4. {"action_type": "order_treatment", "patient_id": "P001", "params": {"treatment": "medication"}}
5. {"action_type": "send_to_or", "patient_id": "P001"}
6. {"action_type": "discharge", "patient_id": "P001"}
7. {"action_type": "reassess", "patient_id": "P001"}
8. {"action_type": "submit"}

RULES:
- Triage critical patients FIRST (low SpO2, high HR, altered consciousness)
- Assign beds to urgent patients before stable ones
- Discharge non-urgent patients to free beds for critical ones
- Use OR for surgical emergencies only
- Respond with ONLY the JSON action object"""


def env_reset(task_id="task_easy", seed=None):
    seed = seed or random.randint(0, 100000)
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action):
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def format_obs(obs_data):
    """Format observation into text prompt (same as training notebook)."""
    obs = obs_data.get("observation", obs_data)
    lines = [f"=== ER (Step {obs.get('current_step',0)}/{obs.get('max_steps',0)}) ==="]
    lines.append(f"Action result: {obs.get('last_action_message','')}")
    s = obs.get("summary", {})
    lines.append(f"Patients: {s.get('total_patients',0)} total | {s.get('alive',0)} alive | {s.get('dead',0)} dead | {s.get('treated',0)} treated")
    h = obs.get("hospital", {})
    beds = h.get("beds", [])
    lines.append(f"Beds: {sum(1 for b in beds if not b.get('occupied'))}/{len(beds)} free")
    docs = h.get("doctors", [])
    lines.append(f"Doctors: {sum(1 for d in docs if not d.get('busy'))}/{len(docs)} available")
    or_info = h.get("or", {})
    or_str = "Available" if or_info.get("available") else f"In use (cooldown {or_info.get('cooldown_steps',0)})"
    lines.append(f"OR: {or_str}")
    for p in obs.get("waiting_patients", []):
        v = p.get("vitals", {})
        tl = f" [Triage: {p['triage_level']}]" if p.get("triage_level") else ""
        lines.append(f"[WAIT] {p['id']} {p.get('name','')}: {p.get('chief_complaint','')} | HR={v.get('hr')} SpO2={v.get('spo2')}%{tl}")
        lines.append(f"  Symptoms: {p.get('symptoms','')}")
    for p in obs.get("admitted_patients", []):
        v = p.get("vitals", {})
        tl = f" [Triage: {p.get('triage_level','')}]" if p.get("triage_level") else ""
        notes = f" | Notes: {p['clinical_notes']}" if p.get("clinical_notes") else ""
        lines.append(f"[BED] {p['id']} {p.get('name','')}: {p.get('chief_complaint','')} | HR={v.get('hr')} SpO2={v.get('spo2')}%{tl}{notes}")
    return "\n".join(lines)


def get_severity_from_vitals(p: dict) -> int:
    """Estimate severity from visible vitals."""
    v = p.get("vitals", {})
    hr = v.get("hr", 80)
    spo2 = v.get("spo2", 98)
    consciousness = v.get("consciousness", "alert")

    if consciousness in ("unresponsive",):
        return 1
    if consciousness in ("confused",):
        return 2
    if spo2 < 85:
        return 1
    if spo2 < 92:
        return 2
    if hr > 130:
        return 1
    if hr > 110:
        return 2
    if spo2 < 95:
        return 3
    if hr > 95:
        return 3
    return 5


def needs_surgery(p: dict) -> bool:
    """Guess from clinical notes whether patient needs surgery."""
    notes = p.get("clinical_notes", "")
    complaint = p.get("chief_complaint", "").lower()
    symptoms = p.get("symptoms", "").lower()

    surgery_keywords = ["laceration", "fracture", "hemorrhage", "hematoma",
                        "dissection", "torsion", "obstruction", "appendicitis",
                        "surgery", "surgical", "bleeding", "arterial",
                        "immediate intervention", "visible muscle"]

    text = f"{notes} {complaint} {symptoms}".lower()
    return any(kw in text for kw in surgery_keywords)


def expert_choose_action(obs_data: dict) -> dict:
    """
    Expert ER protocol:
    1. Triage untriaged patients (critical-looking ones first)
    2. Assign beds to critical triaged patients
    3. Assign doctors to examine admitted patients
    4. Treat: surgery patients → OR, medical → medication
    5. Discharge stable patients to free beds
    6. Submit when all resolved
    """
    obs = obs_data.get("observation", obs_data)
    waiting = obs.get("waiting_patients", [])
    admitted = obs.get("admitted_patients", [])
    hospital = obs.get("hospital", {})
    summary = obs.get("summary", {})

    free_beds = sum(1 for b in hospital.get("beds", []) if not b.get("occupied"))
    free_docs = sum(1 for d in hospital.get("doctors", []) if not d.get("busy"))
    or_available = hospital.get("or", {}).get("available", False)

    # --- Priority 1: Triage untriaged patients (critical vitals first) ---
    untriaged = [p for p in waiting if not p.get("triage_level")]
    if untriaged:
        # Sort by estimated severity (most critical first)
        untriaged.sort(key=lambda p: get_severity_from_vitals(p))
        return {"action_type": "triage", "patient_id": untriaged[0]["id"]}

    # --- Priority 2: Assign beds to critical waiting patients ---
    if free_beds > 0:
        triaged_waiting = sorted(waiting, key=lambda p: p.get("triage_level", 5))
        for p in triaged_waiting:
            tl = p.get("triage_level", 5)
            if tl <= 3:  # Critical/urgent/semi-urgent get beds first
                return {"action_type": "assign_bed", "patient_id": p["id"]}

    # --- Priority 3: Assign doctors to unexamined admitted patients ---
    if free_docs > 0:
        for p in admitted:
            if not p.get("examined") and not p.get("clinical_notes"):
                return {"action_type": "assign_doctor", "patient_id": p["id"], "params": {}}

    # --- Priority 4: Treat examined patients ---
    for p in admitted:
        if p.get("clinical_notes") or p.get("examined"):
            # Check if patient needs surgery
            if needs_surgery(p) and or_available:
                return {"action_type": "send_to_or", "patient_id": p["id"]}
            # Otherwise give medication
            return {"action_type": "order_treatment", "patient_id": p["id"],
                    "params": {"treatment": "medication"}}

    # --- Priority 5: Discharge stable patients to free beds ---
    for p in admitted:
        tl = p.get("triage_level", 5)
        if tl >= 4:  # Stable enough to discharge
            return {"action_type": "discharge", "patient_id": p["id"]}

    # --- Priority 6: Assign beds to remaining waiting patients ---
    if free_beds > 0 and waiting:
        # Assign bed to whoever is waiting, sorted by severity
        sorted_waiting = sorted(waiting, key=lambda p: p.get("triage_level", 5))
        return {"action_type": "assign_bed", "patient_id": sorted_waiting[0]["id"]}

    # --- Priority 7: Discharge stable waiting patients who don't need beds ---
    for p in waiting:
        tl = p.get("triage_level", 5)
        if tl >= 4:
            return {"action_type": "discharge", "patient_id": p["id"]}

    # --- Fallback: Submit ---
    return {"action_type": "submit"}


def run_expert_episode(task_id: str, seed: int, max_steps: int = 40) -> List[dict]:
    """Run one expert episode. Returns list of SFT training examples."""
    examples = []
    obs = env_reset(task_id=task_id, seed=seed)
    done = False
    step = 0

    while not done and step < max_steps:
        obs_text = format_obs(obs)
        action = expert_choose_action(obs)
        action_json = json.dumps(action)

        # Record the training example
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
                {"role": "assistant", "content": action_json},
            ]
        })

        obs = env_step(action)
        done = obs.get("done", False)
        step += 1

    # Get final scores
    meta = obs.get("observation", {}).get("metadata", {})
    score = meta.get("composite_score", 0)
    survival = meta.get("survival_rate", 0)

    return examples, score, survival


def generate_dataset(output_file: str = "expert_sft_data.jsonl",
                     episodes_per_task: int = 80):
    """Generate full SFT dataset from expert agent."""
    tasks = [
        ("task_easy", episodes_per_task),
        ("task_medium", episodes_per_task),
        ("task_hard", episodes_per_task // 2),
    ]

    all_examples = []
    total_scores = []
    total_survivals = []

    print("=" * 60)
    print("Generating Expert SFT Data for TriageAI")
    print("=" * 60)

    for task_id, num_eps in tasks:
        task_scores = []
        task_survivals = []
        print(f"\n--- {task_id} ({num_eps} episodes) ---")

        for ep in range(num_eps):
            seed = random.randint(0, 100000)
            try:
                examples, score, survival = run_expert_episode(task_id, seed)

                # Only keep episodes where the expert did well
                if score > 0.4:  # quality threshold
                    all_examples.extend(examples)
                    task_scores.append(score)
                    task_survivals.append(survival)

                if (ep + 1) % 20 == 0:
                    avg_s = sum(task_scores[-20:]) / max(1, len(task_scores[-20:]))
                    avg_v = sum(task_survivals[-20:]) / max(1, len(task_survivals[-20:]))
                    print(f"  Ep {ep+1}/{num_eps}: examples={len(all_examples)} "
                          f"avg_score={avg_s:.3f} avg_survival={avg_v:.3f}")

            except Exception as e:
                print(f"  Ep {ep+1} error: {e}")
                time.sleep(1)
                continue

        total_scores.extend(task_scores)
        total_survivals.extend(task_survivals)

    # Save dataset
    with open(output_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Dataset generated!")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Saved to: {output_file}")
    print(f"  Avg expert score: {sum(total_scores)/max(1,len(total_scores)):.3f}")
    print(f"  Avg expert survival: {sum(total_survivals)/max(1,len(total_survivals)):.3f}")
    print(f"{'=' * 60}")

    return all_examples


if __name__ == "__main__":
    generate_dataset()
