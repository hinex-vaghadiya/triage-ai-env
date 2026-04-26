"""
Local test for TriageAI environment — no server needed.
Tests reset/step/state cycle, reward boundaries, and determinism.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finale.server.environment import TriageEnvironment


def test_basic_cycle():
    """Test reset → step → done cycle."""
    env = TriageEnvironment()
    print("=== Test: Basic Cycle (task_easy) ===")

    obs = env.reset(seed=42, task_id="task_easy")
    assert "observation" in obs
    assert obs["done"] == False
    ob = obs["observation"]
    print(f"  Reset OK. Patients: {ob['summary']['total_patients']}, "
          f"Waiting: {ob['summary']['waiting']}")
    assert ob["summary"]["total_patients"] == 4

    # Triage first patient
    patients = ob["waiting_patients"]
    pid = patients[0]["id"]
    obs2 = env.step({"action_type": "triage", "patient_id": pid})
    assert obs2["observation"]["last_action_success"] == True
    print(f"  Triage {pid} OK: {obs2['observation']['last_action_message'][:80]}...")

    # Assign bed
    obs3 = env.step({"action_type": "assign_bed", "patient_id": pid})
    assert obs3["observation"]["last_action_success"] == True
    print(f"  Assign bed OK: {obs3['observation']['last_action_message']}")

    # Submit
    obs4 = env.step({"action_type": "submit"})
    assert obs4["done"] == True
    score = obs4["observation"]["metadata"]["composite_score"]
    print(f"  Submit OK. Final score: {score}")
    assert 0.0 <= score <= 1.0
    print("  ✓ PASSED\n")


def test_all_difficulties():
    """Test all 3 difficulty levels."""
    env = TriageEnvironment()
    for task_id, expected_patients in [("task_easy", 4), ("task_medium", 7), ("task_hard", 10)]:
        print(f"=== Test: {task_id} ===")
        obs = env.reset(seed=42, task_id=task_id)
        ob = obs["observation"]
        print(f"  Patients: {ob['summary']['total_patients']}, Max steps: {ob['max_steps']}")
        assert ob["summary"]["total_patients"] == expected_patients
        print("  ✓ PASSED\n")


def test_determinism():
    """Same seed = same scenario."""
    env = TriageEnvironment()
    print("=== Test: Determinism ===")
    obs1 = env.reset(seed=42, task_id="task_easy")
    obs2 = env.reset(seed=42, task_id="task_easy")
    p1 = [p["id"] for p in obs1["observation"]["waiting_patients"]]
    p2 = [p["id"] for p in obs2["observation"]["waiting_patients"]]
    assert p1 == p2, f"Non-deterministic: {p1} != {p2}"
    print(f"  Seed 42 patients match: {p1}")
    print("  ✓ PASSED\n")


def test_reward_boundaries():
    """Reward components always 0.0-1.0."""
    env = TriageEnvironment()
    print("=== Test: Reward Boundaries ===")
    obs = env.reset(seed=42, task_id="task_medium")
    for i in range(10):
        obs = env.step({"action_type": "reassess", "patient_id": "P001"})
    scores = obs["observation"]["metadata"]
    for key in ["survival_rate", "triage_accuracy", "treatment_quality", "composite_score"]:
        val = scores.get(key, 0)
        assert 0.0 <= val <= 1.0, f"{key}={val} out of bounds!"
        print(f"  {key}: {val}")
    print("  ✓ PASSED\n")


def test_invalid_actions():
    """Invalid actions should fail gracefully."""
    env = TriageEnvironment()
    print("=== Test: Invalid Actions ===")
    env.reset(seed=42, task_id="task_easy")

    # Unknown action
    obs = env.step({"action_type": "fly_helicopter"})
    assert obs["observation"]["last_action_success"] == False
    print(f"  Unknown action: {obs['observation']['last_action_message'][:60]}")

    # Invalid patient
    obs = env.step({"action_type": "triage", "patient_id": "P999"})
    assert obs["observation"]["last_action_success"] == False
    print(f"  Invalid patient: {obs['observation']['last_action_message'][:60]}")

    print("  ✓ PASSED\n")


def test_full_episode():
    """Run a complete episode with smart actions."""
    env = TriageEnvironment()
    print("=== Test: Full Episode (Smart Agent) ===")
    obs = env.reset(seed=42, task_id="task_easy")

    steps = 0
    while not obs.get("done", False) and steps < 20:
        ob = obs["observation"]
        waiting = ob.get("waiting_patients", [])
        admitted = ob.get("admitted_patients", [])

        # Simple strategy: triage all → assign beds → treat → discharge
        if waiting:
            p = waiting[0]
            if not p.get("triage_level"):
                obs = env.step({"action_type": "triage", "patient_id": p["id"]})
            else:
                obs = env.step({"action_type": "assign_bed", "patient_id": p["id"]})
        elif admitted:
            p = admitted[0]
            if not p.get("examined"):
                obs = env.step({"action_type": "assign_doctor", "patient_id": p["id"], "params": {}})
            else:
                obs = env.step({"action_type": "order_treatment", "patient_id": p["id"], "params": {"treatment": "medication"}})
        else:
            obs = env.step({"action_type": "submit"})

        steps += 1
        reward = obs.get("reward", 0)
        print(f"  Step {steps}: reward={reward}")

    score = obs["observation"]["metadata"]["composite_score"]
    survival = obs["observation"]["metadata"]["survival_rate"]
    print(f"  Final: score={score}, survival={survival}, steps={steps}")
    assert score > 0.0
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("TriageAI Local Environment Tests")
    print("=" * 60 + "\n")

    test_basic_cycle()
    test_all_difficulties()
    test_determinism()
    test_reward_boundaries()
    test_invalid_actions()
    test_full_episode()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
