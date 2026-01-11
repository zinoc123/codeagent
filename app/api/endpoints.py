from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from app.core.agent import run_agent
from app.utils.file_processing import extract_zip
from app.schemas.report import AnalysisReport
import shutil
import os

router = APIRouter()

@router.post("/analyze", response_model=AnalysisReport)
async def analyze_project(
    problem_description: str = Form(...),
    code_zip: UploadFile = File(...)
):
    # Validate file type
    if not code_zip.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed.")
    
    try:
        # Read zip content
        content = await code_zip.read()
        
        # Extract to temp dir
        temp_dir = extract_zip(content)
        
        try:
            # Run Agent
            report = await run_agent(problem_description, temp_dir)
            return report
        finally:
            # Cleanup temp dir
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
