#!/bin/sh
# Start the SoHoAI gateway.
# CC gateway-models.json cache writing removed 2026-05-22 (claude-code-* alias scheme removed).
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-include "SoHoAI-config.yaml" --reload-dir . --reload-exclude '(.*/|^)\..*' 2>&1 | tee /mnt/nfs/__Backups/SoHoAI--databases/SoHoAI-gateway.log
