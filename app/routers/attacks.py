from fastapi import APIRouter, BackgroundTasks
from app.models import MultiTurnAttackRequest
from app.services.multi_turn_service import run_multi_turn

router = APIRouter()

@router.post("/multi-turn")
async def multi_turn_attack(req: MultiTurnAttackRequest, bg: BackgroundTasks):
    run_id = await run_multi_turn(
        objectives=req.objectives,
        models=req.models,
        memory_labels=req.memory_labels,
        max_concurrency=req.max_concurrency
    )
    return {"run_id": run_id, "status": "started"}
