#!/bin/sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir . --reload-exclude '(.*/|^)\..*' 2>&1 | tee /mnt/nfs/__Backups/SoHoAI--databases/SoHoAI-gateway.log
