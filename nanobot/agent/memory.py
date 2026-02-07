"""Memory system for persistent agent memory."""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Any

from nanobot.utils.helpers import ensure_dir, today_date


class MemoryStore:
    """
    Memory system for the agent.
    
    Supports daily notes (memory/YYYY-MM-DD.md) and long-term memory (MEMORY.md).
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
    
    def get_today_file(self) -> Path:
        """Get path to today's memory file."""
        return self.memory_dir / f"{today_date()}.md"
    
    def read_today(self) -> str:
        """Read today's memory notes."""
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""
    
    def append_today(self, content: str) -> None:
        """Append content to today's memory notes."""
        today_file = self.get_today_file()
        
        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            # Add header for new day
            header = f"# {today_date()}\n\n"
            content = header + content
        
        today_file.write_text(content, encoding="utf-8")
    
    def read_long_term(self) -> str:
        """Read long-term memory (MEMORY.md)."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""
    
    def write_long_term(self, content: str) -> None:
        """Write to long-term memory (MEMORY.md)."""
        self.memory_file.write_text(content, encoding="utf-8")
    
    def get_recent_memories(self, days: int = 7) -> str:
        """
        Get memories from the last N days.
        
        Args:
            days: Number of days to look back.
        
        Returns:
            Combined memory content.
        """
        from datetime import timedelta
        
        memories = []
        today = datetime.now().date()
        
        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"{date_str}.md"
            
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                memories.append(content)
        
        return "\n\n---\n\n".join(memories)
    
    def list_memory_files(self) -> list[Path]:
        """List all memory files sorted by date (newest first)."""
        if not self.memory_dir.exists():
            return []
        
        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)
    
    def get_memory_context(self) -> str:
        """
        Get memory context for the agent.
        
        Returns:
            Formatted memory context including long-term and recent memories.
        """
        parts = []
        
        # Long-term memory
        long_term = self.read_long_term()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)
        
        # Today's notes
        today = self.read_today()
        if today:
            parts.append("## Today's Notes\n" + today)
        
        return "\n\n".join(parts) if parts else ""


class MemUMemoryStore:
    """
    MemU-based memory system for advanced AI memory capabilities.

    Uses MemU's PostgreSQL + pgvector backend for:
    - Semantic similarity search (RAG)
    - Automatic memory categorization
    - Multi-modal memory support (conversation, document, image)
    - Long-term memory with reinforcement learning
    """

    def __init__(self, config: Any):
        """
        Initialize MemU memory store.

        Args:
            config: MemUConfig instance with connection and LLM settings.
        """
        self.config = config
        self._service = None
        self._loop = None

    def _get_service(self):
        """Lazy initialize the MemU MemoryService."""
        if self._service is not None:
            return self._service

        try:
            from memu.app.service import MemoryService
            from memu.app.settings import (
                DatabaseConfig,
                LLMConfig,
                LLMProfilesConfig,
                MemorizeConfig,
                RetrieveConfig,
                BlobConfig,
            )
        except ImportError:
            raise RuntimeError(
                "MemU is not installed. Install it with: pip install -e nanobot-ai[memu]"
            )

        # Build PostgreSQL connection string
        dsn = (
            f"postgresql://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )

        # Configure LLM profiles
        llm_profiles = {
            "default": {
                "provider": "openai",
                "base_url": self.config.base_url,
                "api_key": self.config.api_key,
                "chat_model": self.config.chat_model,
                "embed_model": self.config.embed_model,
                "client_backend": "sdk",
            },
            "embedding": {
                "provider": "openai",
                "base_url": self.config.base_url,
                "api_key": self.config.api_key,
                "embed_model": self.config.embed_model,
                "client_backend": "sdk",
            },
        }

        # Configure database
        database_config = {
            "metadata_store": {
                "provider": "postgres",
                "dsn": dsn,
            },
            "vector_index": {
                "provider": "pgvector",
                "dsn": dsn,
            },
        }

        # Configure blob storage
        resources_dir = Path(self.config.resources_dir).expanduser()
        resources_dir.mkdir(parents=True, exist_ok=True)

        blob_config = {
            "provider": "local",
            "resources_dir": str(resources_dir),
        }

        # Create service
        self._service = MemoryService(
            llm_profiles=llm_profiles,
            database_config=database_config,
            blob_config=blob_config,
        )

        return self._service

    async def memorize_conversation(
        self,
        conversation_text: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Store a conversation in memory.

        Args:
            conversation_text: The conversation text (JSON format preferred).
            user_id: Optional user ID for multi-user memory isolation.

        Returns:
            Result dict with created resources, items, and categories.
        """
        service = self._get_service()

        # Create a temporary file for the conversation
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Try to parse as JSON, otherwise treat as plain text
            try:
                json.loads(conversation_text)
                f.write(conversation_text)
            except json.JSONDecodeError:
                # Wrap as simple conversation
                simple_conv = json.dumps([{
                    "role": "user",
                    "content": conversation_text,
                }])
                f.write(simple_conv)
            temp_path = f.name

        try:
            user_scope = {"user_id": user_id} if user_id else None
            result = await service.memorize(
                resource_url=f"file://{temp_path}",
                modality="conversation",
                user=user_scope,
            )
            return result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def memorize_text(
        self,
        text: str,
        modality: str = "document",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Store text/document content in memory.

        Args:
            text: The text content to store.
            modality: Type of content (document, conversation, etc.).
            user_id: Optional user ID for multi-user memory isolation.

        Returns:
            Result dict with created resources, items, and categories.
        """
        service = self._get_service()

        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_path = f.name

        try:
            user_scope = {"user_id": user_id} if user_id else None
            result = await service.memorize(
                resource_url=f"file://{temp_path}",
                modality=modality,
                user=user_scope,
            )
            return result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        method: str = "rag",
    ) -> dict[str, Any]:
        """
        Retrieve relevant memories for a query.

        Args:
            query: The search query.
            user_id: Optional user ID for filtering.
            method: "rag" for vector search, "llm" for LLM-based ranking.

        Returns:
            Retrieved memories with categories, items, and resources.
        """
        service = self._get_service()

        queries = [{"role": "user", "content": {"text": query}}]
        where = {"user_id": user_id} if user_id else None

        result = await service.retrieve(
            queries=queries,
            where=where,
        )

        return result

    def get_memory_context(self, query: str | None = None, user_id: str | None = None) -> str:
        """
        Get memory context for the agent.

        Args:
            query: Optional query to retrieve relevant memories.
            user_id: Optional user ID for filtering.

        Returns:
            Formatted memory context as text.
        """
        if query:
            # Run async retrieve in sync context
            loop = asyncio.get_event_loop()
            try:
                result = loop.run_until_complete(
                    self.retrieve(query=query, user_id=user_id)
                )
            except RuntimeError:
                # No running loop, create one
                import asyncio
                result = asyncio.run(self.retrieve(query=query, user_id=user_id))

            # Format results
            parts = []
            if result.get("categories"):
                parts.append("## Relevant Memory Categories\n")
                for cat in result["categories"]:
                    name = cat.get("name", "Unknown")
                    summary = cat.get("summary", "")
                    parts.append(f"- **{name}**: {summary}")

            if result.get("items"):
                parts.append("\n## Relevant Memory Items\n")
                for item in result["items"]:
                    mem_type = item.get("memory_type", "knowledge")
                    summary = item.get("summary", "")
                    parts.append(f"- [{mem_type}] {summary}")

            return "\n".join(parts) if parts else ""

        # No query, return generic context
        return "## Memory\nMemU memory system is active. Ask me to retrieve specific information."

    def memorize_conversation_sync(
        self,
        conversation_text: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper for memorize_conversation."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Create a new event loop in a thread
            import concurrent.futures
            import threading

            result = [None]
            exception = [None]

            def run_in_new_loop():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result[0] = new_loop.run_until_complete(
                        self.memorize_conversation(conversation_text, user_id)
                    )
                    new_loop.close()
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join(timeout=30)

            if exception[0]:
                raise exception[0]
            return result[0]
        else:
            return loop.run_until_complete(
                self.memorize_conversation(conversation_text, user_id)
            )

    def retrieve_sync(
        self,
        query: str,
        user_id: str | None = None,
        method: str = "rag",
    ) -> dict[str, Any]:
        """Synchronous wrapper for retrieve."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Create a new event loop in a thread
            import concurrent.futures
            import threading

            result = [None]
            exception = [None]

            def run_in_new_loop():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result[0] = new_loop.run_until_complete(
                        self.retrieve(query, user_id, method)
                    )
                    new_loop.close()
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join(timeout=30)

            if exception[0]:
                raise exception[0]
            return result[0]
        else:
            return loop.run_until_complete(
                self.retrieve(query, user_id, method)
            )
