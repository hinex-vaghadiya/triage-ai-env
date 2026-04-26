"""
TriageAI Baseline Inference Script
Runs an LLM agent against the ER Triage environment.
Outputs [START]/[STEP]/[END] structured logs to stdout.
"""

import os
import sys
import json
import time
import requests

from openai import OpenAI

# =====================================================================
# Environment Variables (OpenEnv compliant)
# =====================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

SPACE_ID = os.getenv("SPACE_ID", "")
ENV_URL = os.getenv("ENV_URL", "")

if not ENV_URL:
    if SPACE_ID:
        owner, name = SPACE_ID.split("/", 1)
        ENV_URL = f"https://{owner}-{name}.hf.space"
    else:
        ENV_URL = "http://localhost:7860"

client = OpenAI(base_url=API_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", HF_TOKEN or "dummy"))

TASK_IDS = ["task_easy", "task_medium", "task_hard"]

# =====================================================================
# System Prompt
# =====================================================================
SYSTEM_PROMPT = """You are an expert Emergency Room triage physician AI. You are managing a busy ER with limited beds, doctors, and one operating room.

Your goal: SAVE AS MANY PATIENTS AS POSSIBLE by triaging, assigning resources, and treating patients efficiently.

Available actions (respond with EXACTLY one JSON object):

1. {"action_type": "triage", "patient_id": "P001"} — Assess patient severity
2. {"action_type": "assign_bed", "patient_id": "P001"} — Assign patient to a bed
3. {"action_type": "assign_doctor", "patient_id": "P001", "params": {"doctor_id": 0}} — Send doctor to examine
4. {"action_type": "order_treatment", "patient_id": "P001", "params": {"treatment": "medication"}} — Treat (medication|iv_fluids|oxygen|monitoring)
5. {"action_type": "send_to_or", "patient_id": "P001"} — Emergency surgery (patient must be in bed)
6. {"action_type": "discharge", "patient_id": "P001"} — Discharge stable patients to free beds
7. {"action_type": "reassess", "patient_id": "P001"} — Re-check patient vitals
8. {"action_type": "submit"} — End episode for final grading

CRITICAL RULES:
- Triage critical patients FIRST (low SpO2, high HR, altered consciousness)
- Assign beds to urgent patients before stable ones
- Don't waste time reassessing stable patients
- Discharge non-urgent patients to free beds for critical ones
- Use the OR for surgical emergencies only
- Respond with ONLY the JSON action, nothing else
"""


# =====================================================================
# Environment Interaction
# =====================================================================
def env_reset(task_id="task_easy", seed=42):
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action):
    resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


def format_observation(obs_data):
    obs = obs_data.get("observation", obs_data)
    lines = [f"=== ER STATUS (Step {obs.get('current_step', 0)}/{obs.get('max_steps', 0)}) ==="]
    lines.append(f"Last action: {obs.get('last_action_message', '')}")

    summary = obs.get("summary", {})
    lines.append(f"\nPatients: {summary.get('total_patients', 0)} total | "
                 f"{summary.get('alive', 0)} alive | {summary.get('dead', 0)} dead | "
                 f"{summary.get('treated', 0)} treated | {summary.get('waiting', 0)} waiting")

    hosp = obs.get("hospital", {})
    beds = hosp.get("beds", [])
    lines.append(f"\nBeds: {sum(1 for b in beds if not b.get('occupied'))}/{len(beds)} free")
    docs = hosp.get("doctors", [])
    lines.append(f"Doctors: {sum(1 for d in docs if not d.get('busy'))}/{len(docs)} available")
    or_info = hosp.get("or", {})
    lines.append(f"OR: {'Available' if or_info.get('available') else f'In use (cooldown: {or_info.get(\"cooldown_steps\", 0)})'}")

    for p in obs.get("waiting_patients", []):
        lines.append(f"\n[WAITING] {p['id']} {p['name']} (Age {p['age']}): {p.get('chief_complaint', '')}")
        lines.append(f"  Symptoms: {p.get('symptoms', '')}")
        v = p.get("vitals", {})
        lines.append(f"  Vitals: HR={v.get('hr')}, BP={v.get('bp')}, SpO2={v.get('spo2')}%")
        if p.get("triage_level"):
            lines.append(f"  Triage Level: {p['triage_level']}")

    for p in obs.get("admitted_patients", []):
        lines.append(f"\n[ADMITTED] {p['id']} {p['name']} (Age {p['age']}): {p.get('chief_complaint', '')}")
        v = p.get("vitals", {})
        lines.append(f"  Vitals: HR={v.get('hr')}, BP={v.get('bp')}, SpO2={v.get('spo2')}%")
        if p.get("clinical_notes"):
            lines.append(f"  Notes: {p['clinical_notes']}")

    return "\n".join(lines)


def parse_action(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"action_type": "submit"}


# =====================================================================
# Main Loop
# =====================================================================
def run_task(task_id, seed=42, max_retries=3):
    print(f"[START] task={task_id} seed={seed} model={MODEL_NAME}", flush=True)
    start = time.time()

    try:
        obs = env_reset(task_id=task_id, seed=seed)
    except Exception as e:
        print(f"[END] task={task_id} score=0.0 total_reward=0.0 steps=0 error=reset_failed", flush=True)
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    step_count = 0
    total_reward = 0.0
    done = False
    final_score = 0.0

    while not done:
        obs_text = format_observation(obs)
        messages.append({"role": "user", "content": obs_text})

        action_dict = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME, messages=messages, temperature=0.1, max_tokens=300,
                )
                action_dict = parse_action(resp.choices[0].message.content.strip())
                messages.append({"role": "assistant", "content": resp.choices[0].message.content.strip()})
                break
            except Exception as e:
                print(f"  LLM error attempt {attempt+1}: {e}", file=sys.stderr)
                if attempt == max_retries - 1:
                    action_dict = {"action_type": "submit"}

        step_count += 1
        obs = env_step(action_dict)

        reward = obs.get("reward", 0.0)
        done = obs.get("done", False)
        total_reward += reward if reward else 0.0
        score = obs.get("observation", {}).get("metadata", {}).get("composite_score", 0.0)

        print(f"[STEP] task={task_id} step={step_count} action={action_dict.get('action_type', '?')} reward={reward} score={score} done={done}", flush=True)

        if done:
            final_score = score
            break

        if len(messages) > 20:
            messages = messages[:2] + messages[-10:]

    print(f"[END] task={task_id} score={final_score} total_reward={round(total_reward, 4)} steps={step_count}", flush=True)


def main():
    print("TriageAI Inference Starting...", file=sys.stderr)
    for task_id in TASK_IDS:
        for seed in [42, 123]:
            try:
                run_task(task_id, seed=seed)
            except Exception as e:
                print(f"  Task {task_id} failed: {e}", file=sys.stderr)
    print("TriageAI Inference Complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
