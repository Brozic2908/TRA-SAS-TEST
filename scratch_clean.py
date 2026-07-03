import os

files_to_remove = [
    r"c:\1_Workspace\5_Nam_5\Vibe code\RAG-LLM\scratch_inspect.py",
    r"c:\1_Workspace\5_Nam_5\Vibe code\RAG-LLM\ingest.py",
    r"c:\1_Workspace\5_Nam_5\Vibe code\RAG-LLM\extract_docx.py",
    r"c:\1_Workspace\5_Nam_5\Vibe code\RAG-LLM\check_data.ipynb"
]

for f in files_to_remove:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"Deleted root file: {f}")
        except Exception as e:
            print(f"Error deleting {f}: {e}")
