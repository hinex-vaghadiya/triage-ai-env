"""
Validate inference output format against the live TriageAI Space.
Uses a rule-based agent (no LLM needed) to test [START]/[STEP]/[END] format.
This simulates what the hackathon validator will do.
"""

import sys
import re
import subprocess

# Run inference with a rule-based agent and capture stdout
SCRIPT = r'''
import sys, json, time, requests

ENV_URL = "https://hinex-07-triage-ai-env.hf.space"
TASKS = ["task_easy", "task_medium", "task_hard"]

def env_reset(task_id, seed=42):
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action):
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def smart_agent(obs):
    """Rule-based agent that makes reasonable ER decisions."""
    ob = obs.get("observation", obs)
    waiting = ob.get("waiting_patients", [])
    admitted = ob.get("admitted_patients", [])
    
    # Strategy: triage all -> assign beds -> treat -> discharge
    for p in waiting:
        if not p.get("triage_level"):
            return {"action_type": "triage", "patient_id": p["id"]}
    
    for p in waiting:
        return {"action_type": "assign_bed", "patient_id": p["id"]}
    
    for p in admitted:
        if not p.get("examined"):
            return {"action_type": "assign_doctor", "patient_id": p["id"], "params": {}}
    
    for p in admitted:
        return {"action_type": "order_treatment", "patient_id": p["id"], "params": {"treatment": "medication"}}
    
    return {"action_type": "submit"}

for task_id in TASKS:
    for seed in [42]:
        print(f"[START] task={task_id} seed={seed} model=rule-based", flush=True)
        try:
            obs = env_reset(task_id, seed)
            step = 0
            done = False
            total_reward = 0.0
            
            while not done and step < 25:
                action = smart_agent(obs)
                obs = env_step(action)
                step += 1
                reward = obs.get("reward", 0.0)
                done = obs.get("done", False)
                total_reward += reward if reward else 0.0
                score = obs.get("observation", {}).get("metadata", {}).get("composite_score", 0.0)
                
                print(f"[STEP] task={task_id} step={step} action={action.get('action_type','?')} reward={reward} score={score} done={done}", flush=True)
            
            final_score = obs.get("observation", {}).get("metadata", {}).get("composite_score", 0.0)
            print(f"[END] task={task_id} score={final_score} total_reward={round(total_reward, 4)} steps={step}", flush=True)
        except Exception as e:
            print(f"[END] task={task_id} score=0.0 total_reward=0.0 steps=0 error={e}", flush=True)
'''

print("=" * 60)
print("TriageAI Format Validator")
print("=" * 60)
print("\nRunning inference against live Space...\n")

result = subprocess.run(
    [sys.executable, "-c", SCRIPT],
    capture_output=True, text=True, timeout=120,
    env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
)

stdout = result.stdout
stderr = result.stderr

if stderr:
    print(f"[stderr] {stderr[:500]}")

print("--- RAW STDOUT ---")
print(stdout)
print("--- END STDOUT ---\n")

# === VALIDATE FORMAT (same as hackathon validator) ===
lines = stdout.strip().split("\n")
starts = [l for l in lines if l.strip().startswith("[START]")]
steps = [l for l in lines if l.strip().startswith("[STEP]")]
ends = [l for l in lines if l.strip().startswith("[END]")]

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

checks = []

# Check 1: [START] blocks found
c1 = len(starts) > 0
checks.append(("Found [START] blocks", c1, f"{len(starts)} found"))
print(f"  {'PASS' if c1 else 'FAIL'}: Found [START] blocks ({len(starts)})")

# Check 2: [STEP] blocks found
c2 = len(steps) > 0
checks.append(("Found [STEP] blocks", c2, f"{len(steps)} found"))
print(f"  {'PASS' if c2 else 'FAIL'}: Found [STEP] blocks ({len(steps)})")

# Check 3: [END] blocks found
c3 = len(ends) > 0
checks.append(("Found [END] blocks", c3, f"{len(ends)} found"))
print(f"  {'PASS' if c3 else 'FAIL'}: Found [END] blocks ({len(ends)})")

# Check 4: Matching START/END count
c4 = len(starts) == len(ends)
checks.append(("START/END count match", c4, f"{len(starts)} vs {len(ends)}"))
print(f"  {'PASS' if c4 else 'FAIL'}: START/END count match ({len(starts)} vs {len(ends)})")

# Check 5: [START] has task= field
c5 = all("task=" in s for s in starts)
checks.append(("[START] has task= field", c5, ""))
print(f"  {'PASS' if c5 else 'FAIL'}: [START] has task= field")

# Check 6: [STEP] has step= and reward= fields
c6 = all("step=" in s and "reward=" in s for s in steps)
checks.append(("[STEP] has step= and reward=", c6, ""))
print(f"  {'PASS' if c6 else 'FAIL'}: [STEP] has step= and reward= fields")

# Check 7: [END] has score= field
c7 = all("score=" in s for s in ends)
checks.append(("[END] has score= field", c7, ""))
print(f"  {'PASS' if c7 else 'FAIL'}: [END] has score= field")

# Check 8: No [START]/[STEP]/[END] on stderr
c8 = "[START]" not in (stderr or "") and "[STEP]" not in (stderr or "") and "[END]" not in (stderr or "")
checks.append(("Tags only on stdout", c8, ""))
print(f"  {'PASS' if c8 else 'FAIL'}: Tags only on stdout (not stderr)")

# Check 9: Scores are numeric
scores_valid = True
for e in ends:
    match = re.search(r"score=([0-9.]+)", e)
    if match:
        val = float(match.group(1))
        if not (0.0 <= val <= 1.0):
            scores_valid = False
    else:
        scores_valid = False
c9 = scores_valid
checks.append(("Scores in [0,1] range", c9, ""))
print(f"  {'PASS' if c9 else 'FAIL'}: Scores in valid [0,1] range")

all_pass = all(c[1] for c in checks)
print(f"\n{'=' * 60}")
if all_pass:
    print("ALL CHECKS PASSED - Ready for submission!")
else:
    failed = [c[0] for c in checks if not c[1]]
    print(f"FAILED CHECKS: {', '.join(failed)}")
print(f"{'=' * 60}")
