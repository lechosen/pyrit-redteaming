
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyrit.setup import initialize_pyrit_async

from app.routers import attacks, results


# ------------------------------------------------------------
# Lifespan Manager (modern replacement for @app.on_event)
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_pyrit_async(
        memory_db_type="SQLite",
        db_path="pyrit.db"
    )
    print("🚀 PyRIT initialized with SQLite local memory.")
    yield
    # Shutdown (optional)
    print("🛑 Shutting down PyRIT API service.")


# ------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------
app = FastAPI(
    title="PyRIT Multi‑Model Red Team API",
    description="RESTful wrapper converting PyRIT into multi-turn attack APIs.",
    version="1.0.0",
    lifespan=lifespan
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Adjust later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Health & Root Endpoints
# ------------------------------------------------------------
@app.get("/hello")
async def hello():
    return {"message": "Hello from AIMVG"}

@app.get("/status")
async def status():
    return {"status": "ok", "pyrit_memory": "sqlite"}

@app.get("/")
async def root():
    return {
        "message": "PyRIT Red Team API is running",
        "endpoints": [
            "/hello",
            "/status",
            "/docs",
            "/attacks/multi-turn",
            "/results/{run_id}"
        ]
    }


# ------------------------------------------------------------
# Routers: Attacks + Results
# ------------------------------------------------------------
app.include_router(attacks.router, prefix="/attacks")
app.include_router(results.router, prefix="/results")
