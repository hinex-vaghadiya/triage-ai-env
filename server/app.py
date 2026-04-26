"""
FastAPI server for TriageAI environment.
Exposes /reset, /step, /state, /health endpoints.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from .environment import TriageEnvironment


# Create environment instance
env = TriageEnvironment()

# Create FastAPI app
app = FastAPI(
    title="TriageAI — Emergency Room Crisis Simulator",
    description="An OpenEnv-compliant RL environment for training LLMs on medical triage under resource constraints.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Request/Response Models
# =====================================================================

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_type: str
    patient_id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "TriageAI"}


@app.get("/")
async def root():
    return {
        "environment": "TriageAI — Emergency Room Crisis Simulator",
        "version": "1.0.0",
        "theme": "Wild Card (Theme 5)",
        "description": "Train LLMs to perform emergency medical triage under resource constraints.",
        "endpoints": ["/reset", "/step", "/state", "/health"],
        "tasks": ["task_easy", "task_medium", "task_hard"],
        "actions": ["triage", "assign_bed", "assign_doctor", "order_treatment", "send_to_or", "discharge", "reassess", "submit"],
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    try:
        result = env.reset(
            seed=request.seed,
            task_id=request.task_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    try:
        action = {
            "action_type": request.action_type,
            "patient_id": request.patient_id,
            "params": request.params or {},
        }
        result = env.step(action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    try:
        scores = env._compute_scores()
        return {
            "episode_id": env._episode_id,
            "task_id": env._task_id,
            "step_count": env._step_count,
            "max_steps": env._max_steps,
            "done": env._done,
            "cumulative_reward": round(env._cumulative_reward, 4),
            "scores": scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
