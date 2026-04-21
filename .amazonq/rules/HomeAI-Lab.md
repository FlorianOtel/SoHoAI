# HomeAI-Lab

Project directory: `/home/florian/Gin-AI/projects/HomeAI-Lab`

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

Databases (all under db_base_path in config.yaml):
- /mnt/nfs/__Backups/HomeAI-lab--databases/sqlite/chats.db
- /mnt/nfs/__Backups/HomeAI-lab--databases/sqlite/rag_state.db
- /mnt/nfs/__Backups/HomeAI-lab--databases/qdrant/
