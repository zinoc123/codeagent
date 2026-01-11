import os
import subprocess
import shutil
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.schemas.report import AnalysisReport, FunctionalVerification, ExecutionResult
from app.utils.file_processing import read_codebase, get_file_structure, read_specific_files, load_documents_for_rag

# State Definition
class AgentState(TypedDict):
    problem_description: str
    code_path: str
    file_structure: str
    selected_files: List[str]
    rag_context: str
    analysis_report: Optional[AnalysisReport]
    test_code: Optional[str]
    test_result: Optional[ExecutionResult]

# LLM Setup
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0, base_url="http://localhost:11434")

# Prompts
PLAN_SYSTEM_PROMPT = """You are a senior software architect.
Given the file structure of a codebase and a problem description, identify which files are most likely to contain the relevant code to solve the problem.
Return a list of file paths.
"""

ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect.
Analyze the codebase to identify implementation locations for features described by the user.
You are provided with:
1. The problem description.
2. The content of specific files that were deemed relevant.
3. Relevant code snippets retrieved via semantic search (RAG).

Output a JSON report matching the AnalysisReport schema.
For each feature, specify file, function, and lines.
Also suggest an execution plan.
"""

TEST_GEN_SYSTEM_PROMPT = """You are a QA Engineer.
Based on the analysis report and the codebase, generate a standalone verification script.
If the project is Node.js, generate a JavaScript file using 'assert' module (or 'supertest' if available/appropriate).
If the project is Python, generate a Python script using 'unittest' or 'pytest'.
The script should:
1. Setup necessary environment.
2. Output "TESTS_PASSED" if all pass, or "TESTS_FAILED" with error details.
3. Be self-contained.

Return ONLY the code for the test file. Do not include markdown formatting like ```python or ```javascript.
"""

class FileSelection(BaseModel):
    files: List[str] = Field(description="List of file paths to examine")

# Nodes
def plan_node(state: AgentState):
    print("--- Planning (Hierarchical Selection) ---")
    file_structure = get_file_structure(state['code_path'])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLAN_SYSTEM_PROMPT),
        ("user", "Problem: {problem_description}\nFile Structure:\n{file_structure}")
    ])
    
    structured_llm = llm.with_structured_output(FileSelection)
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "problem_description": state['problem_description'],
        "file_structure": file_structure
    })
    
    return {
        "file_structure": file_structure,
        "selected_files": result.files
    }

def retrieval_node(state: AgentState):
    print("--- Retrieval (RAG) ---")
    # Load documents
    docs = load_documents_for_rag(state['code_path'])
    
    if not docs:
        return {"rag_context": "No documents found for RAG."}
        
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    
    # Create vector store
    # Note: In production, we wouldn't rebuild index every time.
    # embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    
    # Use Qdrant
    url = "http://localhost:6333"
    collection_name = "code_analysis"
    
    # Optional: Re-create collection to ensure clean state for this request
    # In a real multi-user system, we might use unique collection names or filters.
    try:
        client = QdrantClient(url=url)
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant at {url}. Falling back to in-memory if possible or failing. Error: {e}")
        # Fallback or strict failure? Assuming Docker is running as requested.
        pass

    vectorstore = QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=url,
        collection_name=collection_name
    )
    
    # Retrieve
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(state['problem_description'])
    
    rag_context = "\n\n".join([f"--- Snippet from {d.metadata['source']} ---\n{d.page_content}" for d in retrieved_docs])
    
    return {"rag_context": rag_context}

def analyze_node(state: AgentState):
    print("--- Analyzing Code ---")
    # Read content of selected files
    selected_content = read_specific_files(state['code_path'], state['selected_files'])
    
    # Combine context
    combined_context = f"""
    *** SELECTED FILES CONTENT ***
    {selected_content}
    
    *** RETRIEVED SNIPPETS (RAG) ***
    {state['rag_context']}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYSIS_SYSTEM_PROMPT),
        ("user", "Problem: {problem_description}\nContext:\n{context}")
    ])
    
    structured_llm = llm.with_structured_output(AnalysisReport)
    chain = prompt | structured_llm
    
    report = chain.invoke({
        "problem_description": state['problem_description'],
        "context": combined_context
    })
    
    return {"analysis_report": report}

def generate_test_node(state: AgentState):
    print("--- Generating Tests ---")
    report = state['analysis_report']
    
    # We pass the same context or maybe just the report?
    # For better test generation, let's pass the selected files content again if possible, 
    # but here we'll stick to the previous flow pattern.
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", TEST_GEN_SYSTEM_PROMPT),
        ("user", "Analysis: {report}\nProblem: {problem_description}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "report": report.model_dump_json(),
        "problem_description": state['problem_description']
    })
    
    code = result.content
    code = code.replace("```javascript", "").replace("```python", "").replace("```", "").strip()
    
    return {"test_code": code}

def run_test_node(state: AgentState):
    print("--- Running Tests ---")
    code_path = state['code_path']
    test_code = state['test_code']
    
    # Simple heuristic to detect if the code is JS or Python
    # If the code contains "require(" or "import ", we can guess.
    # But checking package.json is also a strong signal for the *project* type.
    # However, the *generated test code* language must match the runner.
    
    is_node_project = "package.json" in state['file_structure']
    
    # Check if the generated code looks like Python
    if "def " in test_code or "import unittest" in test_code or "import pytest" in test_code:
        test_filename = "verification_test.py"
        cmd = ["python", test_filename]
    else:
        test_filename = "verification_test.js"
        cmd = ["node", test_filename]
        # Force install supertest if missing
        if "require('supertest')" in test_code or 'require("supertest")' in test_code:
            try:
                subprocess.run(["npm", "install", "supertest", "--no-save"], cwd=code_path, check=False, capture_output=True, timeout=60)
            except:
                pass
        
    test_file_path = os.path.join(code_path, test_filename)
    
    with open(test_file_path, 'w') as f:
        f.write(test_code)
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=code_path, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        output = result.stdout + "\n" + result.stderr
        passed = result.returncode == 0
    except subprocess.TimeoutExpired:
        output = "Test execution timed out."
        passed = False
    except Exception as e:
        output = str(e)
        passed = False
        
    execution_result = ExecutionResult(tests_passed=passed, log=output)
    
    report = state['analysis_report']
    report.functional_verification = FunctionalVerification(
        generated_test_code=test_code,
        execution_result=execution_result
    )
    
    return {"test_result": execution_result, "analysis_report": report}

# Graph Construction
workflow = StateGraph(AgentState)

workflow.add_node("plan", plan_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("generate_test", generate_test_node)
workflow.add_node("run_test", run_test_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "retrieval")
workflow.add_edge("retrieval", "analyze")
workflow.add_edge("analyze", "generate_test")
workflow.add_edge("generate_test", "run_test")
workflow.add_edge("run_test", END)

app_graph = workflow.compile()

async def run_agent(problem_description: str, code_path: str) -> AnalysisReport:
    initial_state = {
        "problem_description": problem_description,
        "code_path": code_path,
        "file_structure": "",
        "selected_files": [],
        "rag_context": "",
        "analysis_report": None,
        "test_code": None,
        "test_result": None
    }
    
    result = await app_graph.ainvoke(initial_state)
    return result['analysis_report']
