import os
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document

IGNORE_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', 'env', '.idea', '.vscode', 'dist', 'build'}
IGNORE_EXTENSIONS = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.zip', '.tar', '.gz', '.pdf'}

def extract_zip(zip_file_bytes: bytes) -> str:
    """Extracts zip bytes to a temporary directory and returns the path."""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, 'project.zip')
    
    with open(zip_path, 'wb') as f:
        f.write(zip_file_bytes)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    os.remove(zip_path)
    return temp_dir

def read_codebase(root_path: str) -> str:
    """Reads all text files in the directory and returns a single string representation."""
    code_content = []
    
    for root, dirs, files in os.walk(root_path):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in IGNORE_EXTENSIONS:
                continue
                
            try:
                # Check file size (skip if > 100KB)
                if file_path.stat().st_size > 100 * 1024:
                    content = "<Skipped: File too large>"
                else:
                    # Try reading as text
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                rel_path = file_path.relative_to(root_path)
                code_content.append(f"--- File: {rel_path} ---\n{content}\n")
            except Exception as e:
                print(f"Skipping file {file_path}: {e}")
                
    return "\n".join(code_content)

def read_specific_files(root_path: str, file_paths: List[str]) -> str:
    """Reads specific files requested by the agent."""
    code_content = []
    for rel_path in file_paths:
        # Security check: prevent traversal
        if ".." in rel_path or rel_path.startswith("/"):
            continue
            
        full_path = os.path.join(root_path, rel_path)
        if not os.path.exists(full_path):
            continue
            
        try:
             with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
             code_content.append(f"--- File: {rel_path} ---\n{content}\n")
        except:
            pass
    return "\n".join(code_content)

def load_documents_for_rag(root_path: str) -> List[Document]:
    """Loads all code files as LangChain Documents for RAG."""
    docs = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in IGNORE_EXTENSIONS:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                rel_path = str(file_path.relative_to(root_path))
                docs.append(Document(page_content=content, metadata={"source": rel_path}))
            except:
                pass
    return docs

def get_file_structure(root_path: str) -> str:
    """Returns a tree-like string structure of the codebase."""
    structure = []
    root_path_obj = Path(root_path)
    
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        level = root.replace(root_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if Path(f).suffix.lower() not in IGNORE_EXTENSIONS:
                structure.append(f"{subindent}{f}")
                
    return "\n".join(structure)
