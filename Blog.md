# We Gave an LLM a Hospital. It Killed Everyone.

*Here's how we taught it to stop.*

---

Last week, we asked Qwen2.5-3B a simple question: *"A patient presents with tearing chest pain radiating to the back, BP 190/110. What is the likely diagnosis?"*

It nailed it. Stanford-level answer. Aortic dissection, type A vs B classification, treatment protocol, the works.

So we thought — what if we gave it an actual ER to run?

We set up a basic scenario. Four patients arrive at the emergency department. Three beds. Two doctors. One operating room. The model gets vitals, symptoms, and has to decide: who gets triaged first, who gets a bed, who goes to surgery, and who gets sent home.

It was a bloodbath.

The model assigned the only open bed to a 22-year-old with a mild headache. Meanwhile, Patient P003 — a 67-year-old with SpO2 at 82% and an altered level of consciousness — sat in the waiting room for six turns. By the time the model got around to him, he was dead.

Three out of four patients died. Not because the model didn't know medicine. It knew plenty. But knowing what an aortic dissection *is* and knowing *what to do when three people are dying at once and you only have one surgeon* — those are completely different skills.

That's the gap we set out to close.

---

## What We Built

**TriageAI** is an emergency room simulator built on top of OpenEnv. You can think of it as a gym for teaching LLMs how to manage chaos.

The setup is deliberately stressful. Patients arrive with raw symptoms and vitals — no neat diagnosis labels, no severity scores. Heart rate 140, blood oxygen 82%, consciousness level "confused." The model has to look at those numbers and *figure out* who is about to die.

Then comes the hard part. There are never enough beds. The operating room has a cooldown after each surgery. Doctors get tied up with examinations. And every single turn, untreated patients get worse. Wait too long, and they die.

The model has eight actions it can take: triage a patient, assign a bed, call a doctor, order treatment, send to surgery, discharge someone stable, reassess, or submit its final answer. Every one of these is a trade-off. Give a bed to Patient A and Patient B can't have it. Send someone to surgery and the OR is locked for three turns. The environment forces the model to think about consequences — something LLMs are famously bad at.

We designed the reward function to be un-gameable. It's not a single number — it's five components blended together. Survival rate matters most (35% weight), but you also get scored on whether you triaged correctly (20%), whether you matched the right treatment to the right patient (15%), how quickly you helped the critical cases (15%), and whether you actually used your resources efficiently (10%). An agent that just discharges everyone immediately gets a terrible score. One that assigns beds randomly and hopes for the best also fails. You have to actually learn clinical prioritization.

We built three difficulty levels. Easy mode is four patients, which is manageable. Medium throws seven at you, including one whose symptoms are deliberately misleading. Hard mode simulates a mass casualty event: ten patients, three beds, and multiple people crashing at the same time. We've watched GPT-4 struggle with hard mode.

---

## How We Trained It

Our first instinct was pure RL. We tried GRPO — just let the model play the environment over and over and learn from the reward signal. It barely learned anything. The problem is that a 3B model taking random actions in a complex environment almost always ends with everyone dead, so the reward signal is just "0, 0, 0, 0" for hundreds of episodes. There's nothing to learn from.

So we took a different approach.

**Step 1:** We wrote a rule-based expert agent. This is not a neural network — just a handcrafted decision tree. Triage the sickest patient first. Assign beds by severity. Hold the OR for actual surgical emergencies. Discharge stable patients to free beds. Simple rules, but they reflect how a competent triage nurse actually thinks.

**Step 2:** We ran this expert across 100 episodes of the environment (40 easy, 40 medium, 20 hard) and collected about 1,200 examples of "here's what the ER looks like right now, here's what you should do."

**Step 3:** We loaded Qwen2.5-3B-Instruct through Unsloth with 4-bit quantization and QLoRA (rank 16, targeting all attention and MLP layers) and fine-tuned it on these expert demonstrations using TRL's SFTTrainer for 5 epochs. The whole thing ran in about 15 minutes on a single T4.

One thing worth noting: the training script talks directly to the live environment on Hugging Face Spaces. Every single training example was generated from real environment interactions — not from a static JSON file or a pre-built dataset.

---

## What Changed

The numbers tell the story:

| Task | Metric | Before (Baseline Qwen 3B) | After (SFT-Trained) | Change | % Improve |
|---|---|---|---|---|---|
| **task_easy** | Score | 0.448 | 0.656 | +0.208 | **46.38%** |
| **task_easy** | Survival | 0.250 | 0.583 | +0.333 | **133.33%** |
| **task_medium** | Score | 0.518 | 0.638 | +0.120 | **23.16%** |
| **task_medium** | Survival | 0.286 | 0.429 | +0.143 | **50.02%** |
| **task_hard** | Score | 0.535 | 0.608 | +0.073 | **13.72%** |
| **task_hard** | Survival | 0.200 | 0.267 | +0.067 | **33.33%** |

On the easy task, the survival rate jumped from a dismal 25% to nearly 60% — a massive 133% relative improvement. The model also showed consistent, undeniable improvements in survival and composite score across all difficulty levels.

But the numbers don't capture the behavioral shift.

Before training, the model would get stuck in loops. It would triage Patient 1, then triage Patient 2, then triage Patient 3, then triage Patient 4 — and by the time it finished assessing everyone, the first patient had deteriorated and died. It was doing the "safe" thing (gathering information) instead of the *right* thing (acting on incomplete information).

After training, it learned to interleave. Triage the most obviously sick patient, immediately assign them a bed, get a doctor over there, then move on to the next. It stopped trying to be thorough and started being *fast*, which is exactly what real triage requires.

It also learned resource management. Before training, it would waste the OR on non-surgical cases (locking it for three turns for no reason). After training, it held the OR until it found a patient who actually needed surgery. That's not medical knowledge — that's planning.

### The Plots

![Before vs After — Composite scores and survival rates improved across all difficulty levels](training_curves.png)

![Loss curve — Steady convergence from ~2.5 to below 1.0 over training](loss_curve.png)

![Post-training rewards across difficulty levels — Performance drops naturally from easy to hard, confirming the curriculum works](reward_curve.png)

---

## Why Should You Care?

Honestly, we don't think TriageAI is going to be used in actual hospitals. That's not the point.

The point is that there's a huge class of real-world problems that look exactly like this: *multiple competing priorities, not enough resources, a ticking clock, and incomplete information.* Logistics. Cloud infrastructure scaling. Disaster response. Air traffic control. Supply chain management during disruptions.

LLMs today can't handle any of these. They can write you a beautiful essay about supply chain optimization, but they cannot actually *do* it when the constraints are live and the clock is running.

We think environments like TriageAI — ones that force models to make hard trade-offs under pressure — are how we get there. We built the gym. Now anyone with a Hugging Face account can clone it, swap in their own model, and start training.

The environment is live, public, and completely free to use: [https://huggingface.co/spaces/hinex-07/triage-ai-env](https://huggingface.co/spaces/hinex-07/triage-ai-env)

If you want to try training your own model on it, the notebook is here: [training/triage_ai_grpo.ipynb](https://colab.research.google.com/drive/1zC5-DEDIiBHxBbhQJ4LNFa-iLvHIugl0?usp=sharing)

The full source code for the environment and the training scripts is available on GitHub: [https://github.com/hinex-vaghadiya/triage-ai-env](https://github.com/hinex-vaghadiya/triage-ai-env)

We'd love to see what a 7B or 14B model can do with more training budget. We bet it can hit 80%+ survival.

---

*Built by Team Block Dragon · OpenEnv Hackathon 2026*
