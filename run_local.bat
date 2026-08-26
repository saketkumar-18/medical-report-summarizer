@echo off
REM Local dev server for MedSumm
cd /d %~dp0
set MEDSUMM_DEPLOYMENT=local
python -m uvicorn app.main:app --host 127.0.0.1 --port 8321 --reload
