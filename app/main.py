from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(title="AI Code Analyst Agent")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to AI Code Analyst Agent"}
