from typing import List, Optional
from pydantic import BaseModel, Field

class Location(BaseModel):
    file: str
    function: Optional[str] = None
    lines: str

class FeatureAnalysis(BaseModel):
    feature_description: str
    implementation_location: List[Location]

class ExecutionResult(BaseModel):
    tests_passed: bool
    log: str

class FunctionalVerification(BaseModel):
    generated_test_code: str
    execution_result: ExecutionResult

class AnalysisReport(BaseModel):
    feature_analysis: List[FeatureAnalysis]
    execution_plan_suggestion: str
    functional_verification: Optional[FunctionalVerification] = None
