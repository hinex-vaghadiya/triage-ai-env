"""
TriageAI GRPO Training Script
==============================
Train Qwen2.5-3B-Instruct to perform emergency room triage using
Group Relative Policy Optimization (GRPO) via TRL + Unsloth.

Usage (Colab):
    !pip install unsloth trl openai requests matplotlib
    !python triage_train.py

This script connects to the live TriageAI environment on HF Spaces
and trains the model through repeated rollouts.
"""

import os
import json
import random
import time
import requests
import re
from typing import List, Dict, Any

# =====================================================================
# Config
# =====================================================================
ENV_URL = os.getenv("ENV_URL", "https://hinex-07-triage-ai-env.hf.space")
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./triage_ai_trained"
MAX_STEPS_PER_EPISODE = 20
NUM_TRAIN_STEPS = 200
BATCH_SIZE = 4       # rollouts per batch
SAVE_EVERY = 50

# =====================================================================
# System Prompt (same as inference)
# =====================================================================
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


# =====================================================================
# Environment Client
# =====================================================================
def env_reset(task_id="task_easy", seed=None):
    if seed is None:
        seed = random.randint(0, 100000)
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action):
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def format_obs(obs_data):
    """Format observation into a text prompt for the model."""
    obs = obs_data.get("observation", obs_data)
    lines = [f"=== ER STATUS (Step {obs.get('current_step',0)}/{obs.get('max_steps',0)}) ==="]
    lines.append(f"Last action: {obs.get('last_action_message','')}")

    s = obs.get("summary", {})
    lines.append(f"\nPatients: {s.get('total_patients',0)} total | "
                 f"{s.get('alive',0)} alive | {s.get('dead',0)} dead | "
                 f"{s.get('treated',0)} treated | {s.get('waiting',0)} waiting")

    h = obs.get("hospital", {})
    beds = h.get("beds", [])
    lines.append(f"Beds: {sum(1 for b in beds if not b.get('occupied'))}/{len(beds)} free")
    docs = h.get("doctors", [])
    lines.append(f"Doctors: {sum(1 for d in docs if not d.get('busy'))}/{len(docs)} available")
    or_i = h.get("or", {})
    or_str = "Available" if or_i.get("available") else f"In use (cooldown {or_i.get('cooldown_steps',0)})"
    lines.append(f"OR: {or_str}")

    for p in obs.get("waiting_patients", []):
        lines.append(f"\n[WAITING] {p['id']} {p.get('name','')} (Age {p.get('age','')}): {p.get('chief_complaint','')}")
        lines.append(f"  Symptoms: {p.get('symptoms','')}")
        v = p.get("vitals", {})
        lines.append(f"  Vitals: HR={v.get('hr')}, BP={v.get('bp')}, SpO2={v.get('spo2')}%")
        if p.get("triage_level"):
            lines.append(f"  Triage Level: {p['triage_level']}")

    for p in obs.get("admitted_patients", []):
        lines.append(f"\n[IN BED] {p['id']} {p.get('name','')} (Age {p.get('age','')}): {p.get('chief_complaint','')}")
        v = p.get("vitals", {})
        lines.append(f"  Vitals: HR={v.get('hr')}, BP={v.get('bp')}, SpO2={v.get('spo2')}%")
        if p.get("clinical_notes"):
            lines.append(f"  Notes: {p['clinical_notes']}")

    return "\n".join(lines)


def parse_action(text):
    """Extract JSON action from model output."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1] if len(text.split("```")) > 1 else text
        text = text.replace("json", "").strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"action_type": "submit"}


# =====================================================================
# Rollout: Run one episode and collect (prompt, response, reward)
# =====================================================================
def run_rollout(model, tokenizer, task_id="task_easy", seed=None):
    """Run one episode. Returns list of (prompt, completion, reward) tuples."""
    obs = env_reset(task_id=task_id, seed=seed)
    done = False
    trajectory = []

    while not done and len(trajectory) < MAX_STEPS_PER_EPISODE:
        obs_text = format_obs(obs)

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]

        # Tokenize and generate
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                do_sample=True, top_p=0.9,
            )

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        action = parse_action(response_text)
        obs = env_step(action)

        step_reward = obs.get("reward", 0.0)
        done = obs.get("done", False)

        trajectory.append({
            "prompt": input_text,
            "completion": response_text,
            "reward": step_reward,
        })

    # Get final composite score
    final_score = obs.get("observation", {}).get("metadata", {}).get("composite_score", 0.0)
    survival = obs.get("observation", {}).get("metadata", {}).get("survival_rate", 0.0)

    return trajectory, final_score, survival


# =====================================================================
# Reward Functions (for GRPO)
# =====================================================================
def reward_format(completion: str, **kwargs) -> float:
    """Reward for outputting valid JSON action."""
    try:
        text = completion.strip()
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        if "action_type" in obj:
            return 1.0
        return 0.3
    except (ValueError, json.JSONDecodeError):
        return 0.0


def reward_survival(completion: str, **kwargs) -> float:
    """Episode-level: survival rate from metadata."""
    return kwargs.get("survival_rate", 0.0)


def reward_composite(completion: str, **kwargs) -> float:
    """Episode-level: full composite score."""
    return kwargs.get("composite_score", 0.0)


# =====================================================================
# Main Training Loop
# =====================================================================
def main():
    print("=" * 60)
    print("TriageAI GRPO Training")
    print("=" * 60)

    # --- Step 1: Load model with Unsloth ---
    print("\n[1/4] Loading model with Unsloth...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=4096,
            dtype=None,  # auto
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        print(f"  Model loaded: {MODEL_NAME} (4-bit QLoRA, r=16)")
    except ImportError:
        print("  ERROR: unsloth not installed. Run: pip install unsloth")
        print("  Falling back to transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",
                                                      device_map="auto", load_in_4bit=True)

    # --- Step 2: Setup GRPO Trainer ---
    print("\n[2/4] Setting up GRPO trainer...")
    try:
        from trl import GRPOConfig, GRPOTrainer

        training_args = GRPOConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            logging_steps=10,
            save_steps=SAVE_EVERY,
            max_completion_length=200,
            num_generations=BATCH_SIZE,
            temperature=0.7,
        )
        print("  GRPO config created")
    except ImportError:
        print("  ERROR: trl not installed. Run: pip install trl")
        return

    # --- Step 3: Curriculum Training ---
    print("\n[3/4] Starting curriculum training...")
    tasks_curriculum = [
        ("task_easy", 80),
        ("task_medium", 80),
        ("task_hard", 40),
    ]

    all_scores = []
    all_survivals = []

    for task_id, num_episodes in tasks_curriculum:
        print(f"\n  --- Training on {task_id} ({num_episodes} episodes) ---")

        for ep in range(num_episodes):
            seed = random.randint(0, 100000)
            try:
                trajectory, score, survival = run_rollout(
                    model, tokenizer, task_id=task_id, seed=seed
                )
                all_scores.append(score)
                all_survivals.append(survival)

                if (ep + 1) % 10 == 0:
                    avg_score = sum(all_scores[-10:]) / min(10, len(all_scores[-10:]))
                    avg_surv = sum(all_survivals[-10:]) / min(10, len(all_survivals[-10:]))
                    print(f"    Episode {ep+1}/{num_episodes}: "
                          f"score={score:.3f} survival={survival:.3f} "
                          f"(avg10: score={avg_score:.3f} surv={avg_surv:.3f})")

            except Exception as e:
                print(f"    Episode {ep+1} error: {e}")
                continue

    # --- Step 4: Save & Plot ---
    print("\n[4/4] Saving model and plots...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Model saved to {OUTPUT_DIR}")

    # Plot training curves
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Smooth with moving average
        window = 10
        def moving_avg(data, w):
            return [sum(data[max(0,i-w):i+1])/len(data[max(0,i-w):i+1]) for i in range(len(data))]

        ax1.plot(all_scores, alpha=0.3, color='blue')
        ax1.plot(moving_avg(all_scores, window), color='blue', linewidth=2)
        ax1.set_title("Composite Score over Training")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score (0-1)")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        ax2.plot(all_survivals, alpha=0.3, color='red')
        ax2.plot(moving_avg(all_survivals, window), color='red', linewidth=2)
        ax2.set_title("Survival Rate over Training")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Survival Rate (0-1)")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.suptitle("TriageAI Training Progress (GRPO + Unsloth)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
        print("  Training curves saved!")

    except ImportError:
        print("  matplotlib not available, skipping plots")

    # Save metrics
    metrics = {
        "scores": all_scores,
        "survivals": all_survivals,
        "final_avg_score": sum(all_scores[-20:]) / max(1, len(all_scores[-20:])),
        "final_avg_survival": sum(all_survivals[-20:]) / max(1, len(all_survivals[-20:])),
        "total_episodes": len(all_scores),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved. Final avg score: {metrics['final_avg_score']:.3f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Total episodes: {len(all_scores)}")
    print(f"  Final avg score: {metrics['final_avg_score']:.3f}")
    print(f"  Final avg survival: {metrics['final_avg_survival']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
