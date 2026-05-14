# SoHoAI

Project directory: `/home/florian/Gin-AI/projects/SoHoAI`

Distributed two-server AI orchestrator with:
- FastAPI gateway (Server 1: 192.168.1.93)
- LLM inference (Server 2: 192.168.1.95, RTX 5070, Gemma 4 E4B 7.52B Q8_0 via llama-server)
- Redis conversation cache + SQLite persistence
- KV cache slot management for llama-server
- Smart routing: local GPU → cloud fallback
- RAG pipeline: docling + bge-m3 (Ollama) + Qdrant vector store

Key files: main.py, config.yaml, router.py, conversation.py, kv_cache.py, chat_store.py

RAG package: rag_engine/ (schema, collection, embeddings, state, scanner, ingest, search)
RAG CLI: utils/rag_sync_nfs.py, rag_ingest_daemon.py, rag_status.py, rag_search_cli.py, rag_reset.py

Databases (SQLite under db_base_path in config.yaml; Qdrant active storage is local NVMe):
- /mnt/nfs/__Backups/SoHoAI--databases/sqlite/telemetry.db
- /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db
- /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/ (Qdrant snapshots on NAS)
- /var/lib/qdrant/storage (Qdrant active storage — local NVMe, Server 1)
