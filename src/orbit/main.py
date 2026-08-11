from fastapi import FastAPI

from orbit.api.routes.health import router as health_router

app = FastAPI(title="Orbit")

app.include_router(health_router)
