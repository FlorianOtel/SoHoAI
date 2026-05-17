#!/bin/sh
# Write CC gateway model discovery cache before starting uvicorn.
# This is required because CC's internal $14() cache-writer (which calls GET /v1/models
# and writes ~/.claude/cache/gateway-models.json) requires ANTHROPIC_API_KEY or
# ANTHROPIC_AUTH_TOKEN in the process environment.  Setting ANTHROPIC_API_KEY in
# settings.json would override OAuth auth and switch billing to API tokens — see
# docs/Model-routing.md §4.5 for the full explanation.  We write the cache here
# from SoHoAI-config.yaml so that H14() in the CC binary can populate the /model picker.
~/Gin-AI/.Gin-AI-python-3.12/bin/python3 - << 'PYEOF'
import json, os, time, yaml
from pathlib import Path

with open('SoHoAI-config.yaml') as f:
    cfg = yaml.safe_load(f)

models = []
for m in cfg.get('model_list', []):
    name = m.get('model_name', '')
    # Only non-Anthropic models get claude-code-* aliases in the CC picker
    if name.startswith(('anthropic/', 'claude-')):
        continue
    suffix = name.split('/', 1)[-1]
    alias = f'claude-code-{suffix}'
    info = m.get('model_info', {})
    ctx_k = (info.get('context_window') or 0) // 1000
    provider = 'local' if name.startswith('local/') else 'Ollama Cloud'
    label = f'{" ".join(w.capitalize() for w in suffix.replace("-", " ").split())} ({provider}, {ctx_k}k ctx)'
    models.append({'id': alias, 'display_name': label})

base_url = 'http://192.168.1.93:8000'
cache = Path.home() / '.claude' / 'cache' / 'gateway-models.json'
cache.parent.mkdir(parents=True, exist_ok=True)
cache.write_text(json.dumps({
    'baseUrl': base_url,
    'fetchedAt': int(time.time() * 1000),
    'models': models,
}))
print(f'[start-sohoai] CC gateway cache written: {len(models)} models → {cache}')
PYEOF

uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-include "SoHoAI-config.yaml" --reload-dir . --reload-exclude '(.*/|^)\..*' 2>&1 | tee /mnt/nfs/__Backups/SoHoAI--databases/SoHoAI-gateway.log
