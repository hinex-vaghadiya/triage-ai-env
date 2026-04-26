"""Quick format test against live Space - single task."""
import sys, json, requests

ENV = "https://hinex-07-triage-ai-env.hf.space"
task = "task_easy"

print(f"[START] task={task} seed=42 model=rule-based", flush=True)

obs = requests.post(f"{ENV}/reset", json={"task_id": task, "seed": 42}, timeout=30).json()
step = 0
done = False
tr = 0.0

while not done and step < 25:
    ob = obs.get("observation", obs)
    w = ob.get("waiting_patients", [])
    a = ob.get("admitted_patients", [])

    if w and not w[0].get("triage_level"):
        act = {"action_type": "triage", "patient_id": w[0]["id"]}
    elif w:
        act = {"action_type": "assign_bed", "patient_id": w[0]["id"]}
    elif a and not a[0].get("examined"):
        act = {"action_type": "assign_doctor", "patient_id": a[0]["id"], "params": {}}
    elif a:
        act = {"action_type": "order_treatment", "patient_id": a[0]["id"], "params": {"treatment": "medication"}}
    else:
        act = {"action_type": "submit"}

    obs = requests.post(f"{ENV}/step", json=act, timeout=30).json()
    step += 1
    r = obs.get("reward", 0.0)
    done = obs.get("done", False)
    tr += r if r else 0
    sc = obs.get("observation", {}).get("metadata", {}).get("composite_score", 0.0)
    atype = act.get("action_type", "?")

    print(f"[STEP] task={task} step={step} action={atype} reward={r} score={sc} done={done}", flush=True)

fs = obs.get("observation", {}).get("metadata", {}).get("composite_score", 0.0)
print(f"[END] task={task} score={fs} total_reward={round(tr, 4)} steps={step}", flush=True)
