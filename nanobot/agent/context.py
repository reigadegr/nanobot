"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.memory import MemoryStore, MemUMemoryStore
from nanobot.agent.skills import SkillsLoader


def _extract_query_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """
    Extract a search query from the latest user message.

    This enables query-based memory retrieval for MemU to limit
    the memory context to only relevant information.

    Args:
        messages: List of message dictionaries.

    Returns:
        A query string extracted from the last user message, or None.
    """
    if not messages:
        return None

    # Get the last message from the user
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                # Use first 100 chars as query (enough for semantic search)
                return content[:100].strip()
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        return text[:100].strip()

    return None


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.

    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    def __init__(self, workspace: Path, memu_config: Any = None):
        """
        Initialize the context builder.

        Args:
            workspace: Path to the workspace directory.
            memu_config: Optional MemUConfig for MemU memory system.
        """
        self.workspace = workspace
        self.memu_config = memu_config
        self.skills = SkillsLoader(workspace)

        # Use MemU memory if configured and enabled
        if memu_config and getattr(memu_config, 'enabled', False):
            self.memory = MemUMemoryStore(memu_config)
            self.memory_mode = "memu"
        else:
            # Apply token limits from config if available
            long_term_max = None
            daily_max = None
            if memu_config and hasattr(memu_config, 'retrieve'):
                long_term_max = memu_config.retrieve.long_term_max_chars
                daily_max = memu_config.retrieve.daily_max_chars
            self.memory = MemoryStore(
                workspace,
                long_term_max_chars=long_term_max,
                daily_max_chars=daily_max,
            )
            self.memory_mode = "file"
    
    async def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.

        Args:
            skill_names: Optional list of skills to include.

        Returns:
            Complete system prompt.
        """
        start = time.time()
        parts = []

        # Core identity
        parts.append(self._get_identity())

        # Bootstrap files
        bootstrap_start = time.time()
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        bootstrap_time = time.time() - bootstrap_start
        if bootstrap_time > 0.1:
            logger.info(f"[PERF] Bootstrap files loaded in {bootstrap_time:.2f}s")

        # Memory context
        memory_start = time.time()
        memory = await self.memory.get_memory_context()
        memory_time = time.time() - memory_start
        logger.info(f"[PERF] Memory context retrieved in {memory_time:.2f}s (mode: {self.memory_mode})")
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        skills_start = time.time()
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        skills_time = time.time() - skills_start
        if skills_time > 0.1:
            logger.info(f"[PERF] Skills loaded in {skills_time:.2f}s")

        total_time = time.time() - start
        logger.info(f"[PERF] build_system_prompt total: {total_time:.2f}s")
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{now}

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Memory files: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
For normal conversation, just respond with text - do not call the message tool.

Always be helpful, accurate, and concise. When using tools, explain what you're doing.
When remembering something, write to {workspace_path}/memory/MEMORY.md"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    async def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Token management: For MemU, uses query-based retrieval to limit
        memory context to only relevant information.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        import time
        start = time.time()
        messages = []

        # System prompt - for MemU, use query-based retrieval
        system_prompt = await self.build_system_prompt(skill_names)

        # For MemU memory, get query-based context to limit tokens
        if self.memory_mode == "memu":
            query_retrieval_start = time.time()
            query = _extract_query_from_messages(history) or current_message[:100]
            memory_context = await self.memory.get_memory_context(query=query, user_id=chat_id)
            query_retrieval_time = time.time() - query_retrieval_start
            logger.info(f"[PERF] MemU query retrieval took {query_retrieval_time:.2f}s")
            if memory_context:
                # Replace the generic memory section with query-specific context
                if "# Memory" in system_prompt:
                    # Find and replace the Memory section
                    parts = system_prompt.split("# Memory")
                    if len(parts) > 1:
                        # Keep everything before # Memory, then add our query-specific memory
                        before = parts[0]
                        # Find where the memory section ends (next # or end)
                        after_parts = parts[1].split("\n\n#", 1)
                        if len(after_parts) > 1:
                            after = "\n\n#" + after_parts[1]
                        else:
                            after = ""
                        system_prompt = before + memory_context + after

        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        total_time = time.time() - start
        logger.info(f"[PERF] build_messages total: {total_time:.2f}s")
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        
        if tool_calls:
            msg["tool_calls"] = tool_calls
        
        messages.append(msg)
        return messages
