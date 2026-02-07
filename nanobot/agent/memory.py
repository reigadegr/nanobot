"""Memory system for persistent agent memory."""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, today_date


def _truncate_text(text: str, max_chars: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum character count while preserving word boundaries.

    Args:
        text: The text to truncate.
        max_chars: Maximum number of characters to keep.
        add_ellipsis: Whether to add "..." at the end if truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_chars:
        return text

    # Try to truncate at a word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:  # If we found a space in the last 20%
        truncated = truncated[:last_space]

    if add_ellipsis:
        truncated = truncated.rstrip() + "..."
    return truncated


class MemoryStore:
    """
    Memory system for the agent.

    Supports daily notes (memory/YYYY-MM-DD.md) and long-term memory (MEMORY.md).

    Token management:
    - Long-term memory is truncated to prevent token explosion
    - Daily notes are truncated to a configurable limit
    """

    # Default limits (can be overridden via config)
    DEFAULT_LONG_TERM_MAX_CHARS = 2000
    DEFAULT_DAILY_MAX_CHARS = 1000

    def __init__(self, workspace: Path, long_term_max_chars: int | None = None, daily_max_chars: int | None = None):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.long_term_max_chars = long_term_max_chars or self.DEFAULT_LONG_TERM_MAX_CHARS
        self.daily_max_chars = daily_max_chars or self.DEFAULT_DAILY_MAX_CHARS
    
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

        Token management: Content is truncated to prevent token explosion.

        Returns:
            Formatted memory context including long-term and recent memories.
        """
        parts = []

        # Long-term memory (with truncation)
        long_term = self.read_long_term()
        if long_term:
            truncated = _truncate_text(long_term, self.long_term_max_chars)
            parts.append("## Long-term Memory\n" + truncated)

        # Today's notes (with truncation)
        today = self.read_today()
        if today:
            truncated = _truncate_text(today, self.daily_max_chars)
            parts.append("## Today's Notes\n" + truncated)

        return "\n\n".join(parts) if parts else ""


class MemUMemoryStore:
    """
    MemU-based memory system for advanced AI memory capabilities.

    Uses MemU's PostgreSQL + pgvector backend for:
    - Semantic similarity search (RAG)
    - Automatic memory categorization
    - Multi-modal memory support (conversation, document, image)
    - Long-term memory with reinforcement learning

    Token management:
    - Top-K limits prevent token explosion from over-retrieval
    - Query-based retrieval ensures only relevant memories are loaded
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
        # Token management limits from config
        self.retrieve_config = getattr(config, 'retrieve', None)
        if self.retrieve_config is None:
            # Fallback to defaults if not configured
            from nanobot.config.schema import RetrieveLimitsConfig
            self.retrieve_config = RetrieveLimitsConfig()

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
                RetrieveCategoryConfig,
                RetrieveItemConfig,
                RetrieveResourceConfig,
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

        # Token management: Apply top_k limits from config (prevents token explosion)
        retrieve_config = RetrieveConfig(
            category=RetrieveCategoryConfig(
                enabled=True,
                top_k=self.retrieve_config.category_top_k,
            ),
            item=RetrieveItemConfig(
                enabled=True,
                top_k=self.retrieve_config.item_top_k,
            ),
            resource=RetrieveResourceConfig(
                enabled=True,
                top_k=self.retrieve_config.resource_top_k,
            ),
        )

        # Create service
        self._service = MemoryService(
            llm_profiles=llm_profiles,
            database_config=database_config,
            blob_config=blob_config,
            retrieve_config=retrieve_config,
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
        retrieve_start = time.time()
        service = self._get_service()

        queries = [{"role": "user", "content": {"text": query}}]
        where = {"user_id": user_id} if user_id else None

        result = await service.retrieve(
            queries=queries,
            where=where,
        )

        retrieve_time = time.time() - retrieve_start
        logger.info(f"[PERF] MemU.retrieve() took {retrieve_time:.2f}s (query: {query[:50]}...)")
        return result

    async def get_memory_context(
        self,
        query: str | None = None,
        user_id: str | None = None,
        max_chars: int = 3000,
    ) -> str:
        """
        Get memory context for the agent.

        Token management: Results are already limited by top_k config,
        and the final output is truncated to max_chars.

        Args:
            query: Optional query to retrieve relevant memories.
            user_id: Optional user ID for filtering.
            max_chars: Maximum characters for the formatted output.

        Returns:
            Formatted memory context as text.
        """
        if query and self.retrieve_config.use_query_retrieval:
            result = await self.retrieve(query=query, user_id=user_id)

            # Format results with truncation per item
            parts = []
            if result.get("categories"):
                parts.append("## Relevant Memory Categories\n")
                for cat in result["categories"]:
                    name = cat.get("name", "Unknown")
                    summary = cat.get("summary", "")
                    # Truncate each category summary to prevent bloat
                    summary = _truncate_text(summary, 200)
                    parts.append(f"- **{name}**: {summary}")

            if result.get("items"):
                parts.append("\n## Relevant Memory Items\n")
                for item in result["items"]:
                    mem_type = item.get("memory_type", "knowledge")
                    summary = item.get("summary", "")
                    # Truncate each item summary to prevent bloat
                    summary = _truncate_text(summary, 150)
                    parts.append(f"- [{mem_type}] {summary}")

            context = "\n".join(parts) if parts else ""
            # Final truncation to ensure total length limit
            return _truncate_text(context, max_chars, add_ellipsis=False)

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
