# Research Agent (Gemini free / PowerShell)

This repository contains a small research agent script that can use the free Google Gemini API
or a local Ollama model. It was prepared to run from PowerShell on Windows.

Files included
- requirements.txt — Python package dependencies
- research_agent.py — the main agent script

Quick PowerShell setup
1. Clone the repo
   git clone https://github.com/sameer23-max/research-agent-gemini-powershell.git
   cd research-agent-gemini-powershell

2. Create & activate a virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   If activation is blocked, run as Administrator:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

3. Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

4. Set environment variables (PowerShell)
   $env:GEMINI_API_KEY = "your_key_here"

   Optional: use a local Ollama model instead
   $env:PROVIDER = "ollama"
   $env:MODEL = "qwen2.5:3b"

5. Run the agent
   python research_agent.py

Notes
- The script will refuse to run without a GEMINI_API_KEY unless PROVIDER is set to "ollama".
- Ask the agent to save a report (it only writes reports when explicitly requested).
