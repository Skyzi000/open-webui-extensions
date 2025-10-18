"""
title: Graphiti Memory Manage Tool
author: Skyzi000
description: Manage specific entities, relationships, or episodes in Graphiti knowledge graph memory.
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-extensions
version: 0.2
requirements: graphiti-core[falkordb]

Design:
- Main class: Tools
- Helper class: GraphitiHelper (handles initialization, not exposed to AI)
- Related components:
  - Graphiti: Knowledge graph memory system
  - FalkorDriver: FalkorDB backend driver for graph storage
  - OpenAIClient: OpenAI client with JSON structured output support
  - OpenAIGenericClient: Generic OpenAI-compatible client
  - OpenAIEmbedder: Embedding model for semantic search

Architecture:
- Search and Delete: Search for specific entities, edges, or episodes, then delete them via Cypher queries
- Episode Deletion: Uses Graphiti's remove_episode() method
- Node/Edge Deletion: Uses driver's execute_query() with Cypher DELETE statements
- UUID-based Deletion: Delete by UUID for precise control
- Batch Operations: Delete multiple items at once
- Group Isolation: Only delete from user's own memory space (respects group_id)

Related Filter:
- functions/filter/graphiti_memory.py: Main memory management filter
"""

import os
import re
import json
import copy
import asyncio
import contextvars
import hashlib
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable
from urllib.parse import quote

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge
from openai import AsyncOpenAI

# Context variable to store user-specific headers for each async request
# This ensures complete isolation between concurrent requests without locks
user_headers_context = contextvars.ContextVar('user_headers', default={})


class MultiUserOpenAIClient(OpenAIClient):
    """
    Custom OpenAI LLM client that retrieves user-specific headers from context variables.
    This allows a single Graphiti instance to safely handle concurrent requests from multiple users.
    
    Overrides self.client property to inject user headers dynamically without copying parent logic.
    This ensures automatic compatibility with future Graphiti updates.
    """
    
    def __init__(self, config: LLMConfig | None = None, cache: bool = False, **kwargs):
        if config is None:
            config = LLMConfig()
        
        # Store base client for dynamic header injection
        self._base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        
        # Initialize parent with our base client and any additional kwargs
        super().__init__(config, cache, self._base_client, **kwargs)
    
    @property
    def client(self) -> AsyncOpenAI:
        """Dynamically return client with user-specific headers from context"""
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client
    
    @client.setter
    def client(self, value: AsyncOpenAI):
        """Store base client for future header injection"""
        self._base_client = value


class MultiUserOpenAIGenericClient(OpenAIGenericClient):
    """
    Custom OpenAI-compatible generic LLM client that retrieves user-specific headers from context variables.
    
    Overrides self.client property to inject user headers dynamically without copying parent logic.
    This ensures automatic compatibility with future Graphiti updates.
    """
    
    def __init__(self, config: LLMConfig | None = None, cache: bool = False, **kwargs):
        if config is None:
            config = LLMConfig()
        
        # Store base client for dynamic header injection
        self._base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        
        # Initialize parent with our base client and any additional kwargs
        super().__init__(config, cache, self._base_client)
    
    @property
    def client(self) -> AsyncOpenAI:
        """Dynamically return client with user-specific headers from context"""
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client
    
    @client.setter
    def client(self, value: AsyncOpenAI):
        """Store base client for future header injection"""
        self._base_client = value


class MultiUserOpenAIEmbedder(OpenAIEmbedder):
    """
    Custom OpenAI embedder that retrieves user-specific headers from context variables.
    
    Overrides self.client property to inject user headers dynamically without copying parent logic.
    This ensures automatic compatibility with future Graphiti updates.
    """
    
    def __init__(
        self,
        config: OpenAIEmbedderConfig | None = None,
        client: AsyncOpenAI | None = None,
    ):
        if config is None:
            config = OpenAIEmbedderConfig()
        
        # Store base client for dynamic header injection
        if client is not None:
            self._base_client = client
        else:
            self._base_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        
        # Initialize parent with our base client
        super().__init__(config, self._base_client)
    
    @property
    def client(self) -> AsyncOpenAI:
        """Dynamically return client with user-specific headers from context"""
        headers = user_headers_context.get()
        if headers:
            return self._base_client.with_options(default_headers=headers)
        return self._base_client
    
    @client.setter
    def client(self, value: AsyncOpenAI):
        """Store base client for future header injection"""
        self._base_client = value


class GraphitiHelper:
    def __init__(self, tools_instance):
        self.tools = tools_instance
        self.graphiti = None
        self._last_config = None
    
    @property
    def valves(self):
        """Always get fresh valves from Tools instance."""
        return self.tools.valves
    
    @staticmethod
    def validate_message_format(content: str) -> tuple[bool, Optional[str]]:
        """
        Validate message format content.
        
        :param content: Content to validate
        :return: Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if ':' not in content:
            error_msg = (
                "❌ Error: For source='message', content must be in 'speaker: content' format.\n"
                "Example: 'user: Hello' or 'assistant: How can I help?'\n"
                "Each line should start with a speaker name followed by a colon."
            )
            return False, error_msg
        
        # Check if at least one line follows the format
        lines = content.strip().split('\n')
        valid_format = False
        for line in lines:
            if ':' in line and line.split(':', 1)[0].strip():
                valid_format = True
                break
        
        if not valid_format:
            error_msg = (
                "❌ Error: For source='message', at least one line must follow 'speaker: content' format.\n"
                "Provided content does not have any valid message lines.\n"
                "Example format:\n"
                "user: What's the weather?\n"
                "assistant: It's sunny today."
            )
            return False, error_msg
        
        return True, None
    
    def get_config_hash(self) -> str:
        """Generate configuration hash for change detection."""
        # Get all valve values as dict, excluding non-config fields
        valve_dict = self.valves.model_dump(
            exclude={
                'debug_print',  # Debugging settings don't affect initialization
                'group_id_format',  # Group ID format doesn't affect Graphiti init
                'confirmation_timeout',  # UI timeout doesn't affect Graphiti init
            }
        )
        # Sort keys for consistent hashing
        config_str = '|'.join(f"{k}={v}" for k, v in sorted(valve_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def config_changed(self) -> bool:
        """Check if configuration has changed."""
        current_hash = self.get_config_hash()
        if self._last_config != current_hash:
            if self._last_config is not None and self.valves.debug_print:
                print("Configuration changed, will reinitialize Graphiti")
            return True
        return False
    
    def initialize_graphiti(self):
        """Initialize Graphiti with configured settings."""
        if self.graphiti is not None and not self.config_changed():
            return
        
        if self.valves.debug_print:
            print("Initializing Graphiti for memory deletion...")
        
        # Disable telemetry if configured
        if not self.valves.graphiti_telemetry_enabled:
            os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
        
        # Set semaphore limit via environment variable
        os.environ['SEMAPHORE_LIMIT'] = str(self.valves.semaphore_limit)
        
        # Configure LLM client
        llm_config = LLMConfig(
            api_key=self.valves.api_key,
            model=self.valves.model,
            small_model=self.valves.small_model,
            base_url=self.valves.openai_api_url,
        )
        # Select LLM client based on configuration - use multi-user versions
        if self.valves.llm_client_type.lower() == "openai":
            llm_client = MultiUserOpenAIClient(config=llm_config)
            if self.valves.debug_print:
                print("Using Multi-User OpenAI client")
        elif self.valves.llm_client_type.lower() == "generic":
            llm_client = MultiUserOpenAIGenericClient(config=llm_config)
            if self.valves.debug_print:
                print("Using Multi-User OpenAI-compatible generic client")
        else:
            # Default to OpenAI client for unknown values
            llm_client = MultiUserOpenAIClient(config=llm_config)
            if self.valves.debug_print:
                print(f"Unknown client type '{self.valves.llm_client_type}', defaulting to Multi-User OpenAI client")
        # Initialize embedder
        embedder = MultiUserOpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=self.valves.api_key,
                base_url=self.valves.openai_api_url,
                embedding_model=self.valves.embedding_model,
                embedding_dim=self.valves.embedding_dim,
            )
        )
        
        # Initialize based on backend
        if self.valves.debug_print:
            print(f"Graph DB Backend: {self.valves.graph_db_backend}")
            print(f"Neo4j URI: {self.valves.neo4j_uri}")
            print(f"FalkorDB Host: {self.valves.falkordb_host}:{self.valves.falkordb_port}")
        
        falkor_driver = None
        if self.valves.graph_db_backend.lower() == "falkordb":
            if self.valves.debug_print:
                print("Initializing FalkorDB driver...")
            falkor_driver = FalkorDriver(
                host=self.valves.falkordb_host,
                port=self.valves.falkordb_port,
                username=self.valves.falkordb_username,
                password=self.valves.falkordb_password,
            )
        # Initialize Graphiti
        if falkor_driver:
            if self.valves.debug_print:
                print("Creating Graphiti instance with FalkorDB...")
            self.graphiti = Graphiti(
                graph_driver=falkor_driver,
                llm_client=llm_client,
                embedder=embedder,
                # OpenAIRerankerClient requires AsyncOpenAI client
                # Use _base_client from our custom multi-user client
                cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
            )
        elif self.valves.graph_db_backend.lower() == "neo4j":
            if self.valves.debug_print:
                print("Creating Graphiti instance with Neo4j...")
            self.graphiti = Graphiti(
                self.valves.neo4j_uri,
                self.valves.neo4j_user,
                self.valves.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                # OpenAIRerankerClient requires AsyncOpenAI client
                # Use _base_client from our custom multi-user client
                cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
            )
        else:
            raise ValueError(f"Unsupported graph database backend: {self.valves.graph_db_backend}. Supported backends are 'neo4j' and 'falkordb'.")
        
        self._last_config = self.get_config_hash()
        
        if self.valves.debug_print:
            print("Graphiti initialized successfully")
    
    async def ensure_graphiti_initialized(self) -> bool:
        """Ensure Graphiti is initialized, retry if needed."""
        if self.graphiti is None or self.config_changed():
            try:
                if self.valves.debug_print:
                    print("=== ensure_graphiti_initialized: Attempting initialization ===")
                self.initialize_graphiti()
                return True
            except Exception as e:
                print(f"Failed to initialize Graphiti: {e}")
                if self.valves.debug_print:
                    traceback.print_exc()
                return False
        return True
    
    def get_group_id(self, user: dict) -> Optional[str]:
        """
        Generate group_id from user information based on configured format.
        
        Args:
            user: User dictionary containing 'id', 'email', 'name'
            
        Returns:
            Generated group_id or None if group filtering is disabled
        """
        if self.valves.group_id_format.lower() == 'none':
            return None
        
        user_id = user.get('id', 'unknown')
        user_email = user.get('email', '')
        user_name = user.get('name', '')
        
        # Sanitize email and name
        sanitized_email = re.sub(r'[@.]', lambda m: '_at_' if m.group() == '@' else '_', user_email)
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', user_name)
        
        group_id = self.valves.group_id_format.format(
            user_id=user_id,
            user_email=sanitized_email,
            user_name=sanitized_name,
        )
        
        # Final sanitization
        group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', group_id)
        
        return group_id
    
    def is_japanese_preferred(self, user: dict) -> bool:
        """
        Check if user prefers Japanese language based on UserValves settings.
        
        Args:
            user: User dictionary containing 'valves' with message_language setting
            
        Returns:
            True if user prefers Japanese (ja), False otherwise (default: English)
        """
        user_valves = user.get("valves")
        if user_valves and hasattr(user_valves, 'message_language'):
            return user_valves.message_language.lower() == 'ja'
        return False
    
    async def delete_nodes_by_uuids(self, uuids: List[str], group_id: Optional[str] = None) -> int:
        """Delete nodes by UUIDs using EntityNode.delete_by_uuids()."""
        if not uuids or not self.graphiti:
            return 0
        
        if self.valves.debug_print:
            print(f"=== delete_nodes_by_uuids: Attempting to delete {len(uuids)} nodes ===")
            print(f"Group ID filter: {group_id}")
            print(f"UUIDs: {uuids}")
        
        try:
            # Use EntityNode.delete_by_uuids() static method
            await EntityNode.delete_by_uuids(self.graphiti.driver, uuids)
            
            if self.valves.debug_print:
                print(f"=== Successfully deleted {len(uuids)} nodes ===")
            
            return len(uuids)
        except Exception as e:
            print(f"Failed to delete nodes: {e}")
            if self.valves.debug_print:
                traceback.print_exc()
            return 0
    
    async def delete_edges_by_uuids(self, uuids: List[str], group_id: Optional[str] = None) -> int:
        """Delete edges by UUIDs using EntityEdge.delete_by_uuids()."""
        if not uuids or not self.graphiti:
            return 0
        
        if self.valves.debug_print:
            print(f"=== delete_edges_by_uuids: Attempting to delete {len(uuids)} edges ===")
            print(f"Group ID filter: {group_id}")
            print(f"UUIDs: {uuids}")
        
        try:
            # Use EntityEdge.delete_by_uuids() static method
            await EntityEdge.delete_by_uuids(self.graphiti.driver, uuids)
            
            if self.valves.debug_print:
                print(f"=== Successfully deleted {len(uuids)} edges ===")
            
            return len(uuids)
        except Exception as e:
            print(f"Failed to delete edges: {e}")
            if self.valves.debug_print:
                traceback.print_exc()
            return 0
    
    async def delete_episodes_by_uuids(self, uuids: List[str]) -> int:
        """Delete episodes by UUIDs using Graphiti.remove_episode()."""
        if not uuids or not self.graphiti:
            return 0
        
        if self.valves.debug_print:
            print(f"=== delete_episodes_by_uuids: Attempting to delete {len(uuids)} episodes ===")
            print(f"UUIDs: {uuids}")
        
        deleted_count = 0
        for uuid in uuids:
            try:
                if self.valves.debug_print:
                    print(f"Deleting episode with UUID: {uuid}")
                
                await self.graphiti.remove_episode(uuid)
                deleted_count += 1
                
                if self.valves.debug_print:
                    print(f"Successfully deleted episode {uuid}")
            except Exception as e:
                print(f"Failed to delete episode {uuid}: {e}")
                if self.valves.debug_print:
                    traceback.print_exc()
        
        if self.valves.debug_print:
            print(f"=== Total deleted: {deleted_count} episodes ===")
        
        return deleted_count
    
    async def show_confirmation_dialog(
        self,
        title: str,
        items: List[str],
        warning_message: str,
        timeout: int,
        __user__: dict = {},
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> tuple[bool, str]:
        """
        Show confirmation dialog and wait for user response.
        Helper method for Tools class - not exposed to AI.
        
        :param title: Dialog title
        :param items: List of items to display for confirmation
        :param warning_message: Warning message to show
        :param timeout: Timeout in seconds
        :param __user__: User information dictionary
        :param __event_call__: Event caller for confirmation dialog
        :return: Tuple of (confirmed, error_message) where:
                 - confirmed: True if user confirmed, False otherwise
                 - error_message: Empty string if confirmed, error message otherwise
        """
        if not __event_call__:
            return False, "❌ Error: Confirmation dialog is not available. Cannot perform deletion without user confirmation."
        
        preview_text = "  \n".join(items)
        
        # Get user's language preference from UserValves
        is_japanese = self.is_japanese_preferred(__user__)
        
        if is_japanese:
            confirmation_message = f"""以下の項目を削除しますか？  
  
{preview_text}  
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
{warning_message}  
⏰ {timeout}秒以内に選択しないと自動的にキャンセルされます。"""
        else:
            confirmation_message = f"""Do you want to delete the following items?  
  
{preview_text}  
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
{warning_message}  
⏰ Auto-cancel in {timeout} seconds if no selection is made."""
        
        try:
            confirmation_task = __event_call__(
                {
                    "type": "confirmation",
                    "data": {
                        "title": title,
                        "message": confirmation_message,
                    },
                }
            )
            
            try:
                result = await asyncio.wait_for(confirmation_task, timeout=timeout)
                if result:
                    return True, ""
                else:
                    return False, "🚫 User cancelled the operation"
            except asyncio.TimeoutError:
                return False, "⏰ Operation timed out - user did not respond within the time limit"
        except Exception:
            return False, "❌ Error: Failed to show confirmation dialog"

class Tools:
    class Valves(BaseModel):
        llm_client_type: str = Field(
            default="openai",
            description="Type of LLM client to use: 'openai' or 'generic'",
        )
        openai_api_url: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI compatible endpoint",
        )
        model: str = Field(
            default="gpt-5-mini",
            description="Model to use for memory processing",
        )
        small_model: str = Field(
            default="gpt-5-nano",
            description="Smaller model for memory processing in legacy mode",
        )
        embedding_model: str = Field(
            default="text-embedding-3-small",
            description="Model to use for embedding memories",
        )
        embedding_dim: int = Field(
            default=1536,
            description="Dimension of the embedding model",
        )
        api_key: str = Field(
            default="",
            description="API key for OpenAI compatible endpoint",
        )
        
        graph_db_backend: str = Field(
            default="neo4j",
            description="Graph database backend to use (e.g., 'neo4j', 'falkordb')",
        )
        
        neo4j_uri: str = Field(
            default="bolt://localhost:7687",
            description="Neo4j database connection URI",
        )
        neo4j_user: str = Field(
            default="neo4j",
            description="Neo4j database username",
        )
        neo4j_password: str = Field(
            default="password",
            description="Neo4j database password",
        )
        
        falkordb_host: str = Field(
            default="localhost",
            description="FalkorDB host address",
        )
        falkordb_port: int = Field(
            default=6379,
            description="FalkorDB port number",
        )
        falkordb_username: Optional[str] = Field(
            default=None,
            description="FalkorDB username (if applicable)",
        )
        falkordb_password: Optional[str] = Field(
            default=None,
            description="FalkorDB password (if applicable)",
        )
        
        graphiti_telemetry_enabled: bool = Field(
            default=False,
            description="Enable Graphiti telemetry",
        )
        
        semaphore_limit: int = Field(
            default=10,
            description="Maximum number of concurrent LLM operations",
        )
        
        group_id_format: str = Field(
            default="{user_id}",
            description="Format string for group_id. Available placeholders: {user_id}, {user_email}, {user_name}. Set to 'none' to disable group filtering.",
        )
        
        forward_user_info_headers: str = Field(
            default="default",
            description="Forward user information headers (User-Name, User-Id, User-Email, User-Role, Chat-Id) to OpenAI API. Options: 'default' (follow environment variable ENABLE_FORWARD_USER_INFO_HEADERS, defaults to false if not set), 'true' (always forward), 'false' (never forward).",
        )
        
        debug_print: bool = Field(
            default=False,
            description="Enable debug printing to console",
        )
        
        confirmation_timeout: int = Field(
            default=60,
            description="Timeout in seconds for confirmation dialogs",
        )
    
    class UserValves(BaseModel):
        message_language: str = Field(
            default="en",
            description="Language for confirmation dialog messages: 'en' (English) or 'ja' (Japanese)",
        )
    def __init__(self):
        self.valves = self.Valves()
        self.helper = GraphitiHelper(self)
        
        # Don't initialize here - Valves may not be loaded yet
        # Initialization happens lazily on first use via ensure_graphiti_initialized()
    
    def _get_user_info_headers(self, user: Optional[dict] = None, chat_id: Optional[str] = None) -> dict:
        """
        Build user information headers dictionary.
        
        Args:
            user: User dictionary containing 'id', 'email', 'name', 'role'
            chat_id: Current chat ID
            
        Returns:
            Dictionary of headers to send to OpenAI API
        """
        # Check Valves setting first
        valves_setting = self.valves.forward_user_info_headers.lower()
        
        if valves_setting == 'true':
            enable_forward = True
        elif valves_setting == 'false':
            enable_forward = False
        elif valves_setting == 'default':
            # Use environment variable (defaults to false if not set)
            env_setting = os.environ.get('ENABLE_FORWARD_USER_INFO_HEADERS', 'false').lower()
            enable_forward = env_setting == 'true'
        else:
            # Invalid value, default to false
            enable_forward = False
        
        if not enable_forward:
            return {}
        
        headers = {}
        if user:
            if user.get('name'):
                headers['X-OpenWebUI-User-Name'] = quote(str(user['name']), safe=" ")
            if user.get('id'):
                headers['X-OpenWebUI-User-Id'] = str(user['id'])
            if user.get('email'):
                headers['X-OpenWebUI-User-Email'] = str(user['email'])
            if user.get('role'):
                headers['X-OpenWebUI-User-Role'] = str(user['role'])
        
        if chat_id:
            headers['X-OpenWebUI-Chat-Id'] = str(chat_id)
        
        return headers
    
    async def add_memory(
        self,
        name: str,
        content: str,
        source: str = "text",
        source_description: str = "",
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Add a new memory episode to the knowledge graph.
        
        This tool creates a new episode and automatically extracts entities and relationships.
        The episode is processed to identify key information and integrate it into the knowledge graph.
        
        :param name: Name/title of the episode (e.g., "Meeting with client", "Product launch announcement")
        :param content: The content to store as memory. Can be text, conversation, or JSON data.
        :param source: Type of source content. Options are:
                      - "text": Plain text content (default) - Most reliable option
                      - "message": Conversation-style content in "speaker: content" format.
                                   REQUIRED FORMAT: Each line must be "speaker_name: message_content"
                                   The speaker name will be automatically extracted as an entity.
                                   Example: "user: Hello" or "assistant: How can I help?"
                      - "json": Structured JSON data (must be valid JSON string)
                                ⚠️ WARNING: Experimental feature with known compatibility issues.
                                May fail with ValidationError depending on the LLM endpoint configuration.
                                If you encounter errors, use "text" source type instead (works with JSON content too).
        :param source_description: Description of where this memory came from (e.g., "team meeting notes", "customer email")
        :return: Result message with episode details
        
        Examples:
        - add_memory(name="Client Meeting", content="Discussed Q1 targets with John", source="text", source_description="meeting notes")
        - add_memory(name="Customer Chat", content="user: What's the return policy?\nassistant: 30-day returns", source="message", source_description="support chat")
        - add_memory(name="Product Data", content='{"product": "Widget X", "price": 99.99}', source="json", source_description="inventory system")
        
        Important Notes:
        - For source="message": Content MUST follow "speaker: content" format for each line
        - For source="json": Content must be a valid JSON string (will be validated)
        
        Note: __user__ and __event_emitter__ are automatically injected by the system.
        """
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        
        source_type = None  # Initialize for error handling scope
        
        try:
            # Validate source type
            source_lower = source.lower()
            if source_lower == "text":
                source_type = EpisodeType.text
            elif source_lower == "message":
                source_type = EpisodeType.message
                # Validate message format using helper method
                is_valid, error_msg = GraphitiHelper.validate_message_format(content)
                if not is_valid:
                    return error_msg if error_msg else "❌ Error: Invalid message format"
            elif source_lower == "json":
                source_type = EpisodeType.json
                # Validate JSON format
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    return f"❌ Error: Invalid JSON content: {str(e)}\nPlease ensure the content is a valid JSON string."
            else:
                return f"❌ Error: Invalid source type '{source}'. Must be 'text', 'message', or 'json'"
            
            # Get user's group_id
            group_id = self.helper.get_group_id(__user__)
            if not group_id:
                return "❌ Error: Group ID is required. Please check your group_id_format configuration."
            
            if self.valves.debug_print:
                print(f"=== add_memory: Adding episode ===")
                print(f"Name: {name}")
                print(f"Source: {source_type}")
                print(f"Group ID: {group_id}")
                print(f"Content length: {len(content)} chars")
            
            # Add episode using Graphiti's add_episode method
            result = await self.helper.graphiti.add_episode(
                name=name,
                episode_body=content,
                source=source_type,
                source_description=source_description,
                reference_time=datetime.now(timezone.utc),
                group_id=group_id,
            )
            
            if self.valves.debug_print:
                print(f"=== Episode added successfully ===")
                print(f"Episode UUID: {result.episode.uuid}")
                print(f"Extracted {len(result.nodes)} entities")
                print(f"Extracted {len(result.edges)} relationships")
            
            # Build response message
            response = f"✅ Memory added successfully!\n\n"
            response += f"**Episode:** {name}\n"
            response += f"**UUID:** `{result.episode.uuid}`\n"
            response += f"**Source Type:** {source}\n"
            
            if result.nodes:
                response += f"\n**Extracted Entities ({len(result.nodes)}):**\n"
                for i, node in enumerate(result.nodes[:5], 1):  # Show first 5
                    response += f"{i}. {node.name}\n"
                if len(result.nodes) > 5:
                    response += f"   ... and {len(result.nodes) - 5} more\n"
            
            if result.edges:
                response += f"\n**Extracted Relationships ({len(result.edges)}):**\n"
                for i, edge in enumerate(result.edges[:5], 1):  # Show first 5
                    response += f"{i}. {edge.name}: {edge.fact}\n"
                if len(result.edges) > 5:
                    response += f"   ... and {len(result.edges) - 5} more\n"
            
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            error_str = str(e)
            
            # Provide more specific error messages for common issues
            if "ValidationError" in error_type or "validation error" in error_str.lower():
                if source_type == EpisodeType.message:
                    error_msg = f"❌ Error: LLM validation failed for message format.\n\n"
                    error_msg += f"The content does not follow the required 'speaker: content' format.\n"
                    error_msg += f"Each line must be formatted as: 'speaker_name: message_content'\n\n"
                    error_msg += f"Valid format examples:\n"
                    error_msg += f"  user: Hello, how are you?\n"
                    error_msg += f"  assistant: I'm doing well, thank you!\n\n"
                    error_msg += f"Technical details: {error_str}"
                elif source_type == EpisodeType.json:
                    error_msg = f"❌ Error: JSON type processing failed.\n\n"
                    error_msg += f"Graphiti's JSON extraction encountered a validation error.\n"
                    error_msg += f"This may be due to endpoint compatibility or JSON structure complexity.\n\n"
                    error_msg += f"✅ Possible solutions:\n"
                    error_msg += f"  1. USE TEXT TYPE: source='text' is more reliable across endpoints\n"
                    error_msg += f"     (Still extracts entities and relationships from JSON content)\n\n"
                    error_msg += f"  2. SIMPLIFY JSON: Flatten nested structures or reduce field count\n\n"
                    error_msg += f"  3. CONFIGURATION: The administrator may need to adjust llm_client_type in Valves\n\n"
                    if self.valves.debug_print:
                        error_msg += f"\nTechnical details: {error_str}"
                else:
                    error_msg = f"❌ Error: LLM validation failed.\n\n"
                    error_msg += f"The LLM response format was unexpected.\n"
                    error_msg += f"This may indicate endpoint compatibility issues.\n\n"
                    error_msg += f"The administrator may need to adjust llm_client_type or endpoint configuration in Valves.\n"
                    if self.valves.debug_print:
                        error_msg += f"\n\nTechnical details: {error_str}"
            elif "ConnectionError" in error_type or "timeout" in error_str.lower():
                error_msg = f"❌ Error: Connection error to LLM service.\n"
                error_msg += f"Network connection or endpoint configuration issue.\n"
                if self.valves.debug_print:
                    error_msg += f"\nDetails: {error_str}"
            elif "api_key" in error_str.lower() or "authentication" in error_str.lower() or "unauthorized" in error_str.lower():
                error_msg = f"❌ Error: Authentication failed.\n"
                error_msg += f"API key configuration issue in Valves.\n"
                if self.valves.debug_print:
                    error_msg += f"\nDetails: {error_str}"
            else:
                error_msg = f"❌ Error adding memory: {error_str}\n\n"
                error_msg += f"Possible solutions:\n"
                error_msg += f"  - Using 'text' source type may work better\n"
                error_msg += f"  - The administrator may need to verify endpoint configuration in Valves\n"
                if self.valves.debug_print:
                    error_msg += f"\n\nNote: Enable debug_print in Valves for detailed error information."
            
            if self.valves.debug_print:
                print(f"Exception type: {error_type}")
                traceback.print_exc()
            
            return error_msg
    
    async def search_entities(
        self,
        query: str,
        limit: int = 10,
        show_uuid: bool = False,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for entities by name or description without deleting them.
        
        This tool allows you to preview entities before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_entities.
        
        :param query: Search query to find entities (e.g., "John Smith", "Python programming")
        :param limit: Maximum number of entities to return (default: 10, max: 100)
        :param show_uuid: Whether to display UUID in search results (default: False). Set to True if you need to see UUIDs for debugging or manual deletion.
        :return: List of found entities with their details
        
        Note: __user__ and __event_emitter__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for entities
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract entity nodes
            entity_nodes = [node for node in search_results.nodes if hasattr(node, 'name')]
            
            if not entity_nodes:
                return f"ℹ️ No entities found matching '{query}'"
            
            total_count = len(entity_nodes)
            
            # Build result message
            result = f"🔍 Found {total_count} entities matching '{query}':\n\n"
            
            for i, node in enumerate(entity_nodes, 1):
                name = getattr(node, 'name', 'Unknown')
                summary = getattr(node, 'summary', 'No description')
                uuid = getattr(node, 'uuid', 'N/A')
                
                result += f"**{i}. {name}**\n"
                result += f"   Summary: {summary}\n"
                if show_uuid:
                    result += f"   UUID: `{uuid}`\n"
                result += "\n"
            
            result += f"💡 To delete these entities, use `search_and_delete_entities` with the same query and limit."
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching entities: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_facts(
        self,
        query: str,
        limit: int = 10,
        show_uuid: bool = False,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for facts (relationships) without deleting them.
        
        This tool allows you to preview relationships before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_facts.
        
        :param query: Search query to find relationships (e.g., "works at", "friends with")
        :param limit: Maximum number of facts to return (default: 10, max: 100)
        :param show_uuid: Whether to display UUID in search results (default: False). Set to True if you need to see UUIDs for debugging or manual deletion.
        :return: List of found facts with their details
        
        Note: __user__ and __event_emitter__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for facts
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract edges
            edges = search_results.edges
            
            if not edges:
                return f"ℹ️ No facts found matching '{query}'"
            
            total_count = len(edges)
            
            # Build result message
            result = f"🔍 Found {total_count} facts matching '{query}':\n\n"
            
            for i, edge in enumerate(edges, 1):
                fact_text = getattr(edge, 'fact', 'Unknown relationship')
                valid_at = getattr(edge, 'valid_at', 'unknown')
                invalid_at = getattr(edge, 'invalid_at', 'present')
                uuid = getattr(edge, 'uuid', 'N/A')
                
                result += f"**{i}. {fact_text}**\n"
                result += f"   Period: {valid_at} → {invalid_at}\n"
                if show_uuid:
                    result += f"   UUID: `{uuid}`\n"
                result += "\n"
            
            result += f"💡 To delete these facts, use `search_and_delete_facts` with the same query and limit."
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching facts: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_episodes(
        self,
        query: str,
        limit: int = 10,
        show_uuid: bool = False,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for episodes (conversation history) without deleting them.
        
        This tool allows you to preview episodes before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_episodes.
        
        :param query: Search query to find episodes (e.g., "conversation about Python")
        :param limit: Maximum number of episodes to return (default: 10, max: 100)
        :param show_uuid: Whether to display UUID in search results (default: False). Set to True if you need to see UUIDs for debugging or manual deletion.
        :return: List of found episodes with their details
        
        Note: __user__ and __event_emitter__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for episodes
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract episodes
            episodes = search_results.episodes
            
            if not episodes:
                return f"ℹ️ No episodes found matching '{query}'"
            
            total_count = len(episodes)
            
            # Build result message
            result = f"🔍 Found {total_count} episodes matching '{query}':\n\n"
            
            for i, episode in enumerate(episodes, 1):
                name = getattr(episode, 'name', 'Unknown episode')
                content = getattr(episode, 'content', '')
                created_at = getattr(episode, 'created_at', 'unknown')
                uuid = getattr(episode, 'uuid', 'N/A')
                
                # Truncate content for preview
                if len(content) > 150:
                    content_preview = content[:150] + "..."
                else:
                    content_preview = content
                
                result += f"**{i}. {name}**\n"
                result += f"   Content: {content_preview}\n"
                result += f"   Created: {created_at}\n"
                if show_uuid:
                    result += f"   UUID: `{uuid}`\n"
                result += "\n"
            
            result += f"💡 To delete these episodes, use `search_and_delete_episodes` with the same query and limit."
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching episodes: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_and_delete_entities(
        self,
        query: str,
        limit: int = 1,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for entities by name or description and delete them after user confirmation.
        
        This tool searches for entities (people, places, concepts) in your memory
        and allows you to delete them along with their relationships.
        
        IMPORTANT: This operation requires user confirmation and cannot be undone.
        
        :param query: Search query to find entities (e.g., "John Smith", "Python programming")
        :param limit: Maximum number of entities to return (default: 1, max: 100)
        :return: Result message with deleted entities count
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for entities using search_() which returns SearchResults
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract entity nodes
            entity_nodes = [node for node in search_results.nodes if hasattr(node, 'name')]
            
            if not entity_nodes:
                return f"ℹ️ No entities found matching '{query}'"
            
            # Show all entities for confirmation (no limit - user must see everything being deleted)
            total_count = len(entity_nodes)
            
            entity_list = []
            preview_items = []
            for i, node in enumerate(entity_nodes, 1):
                summary = getattr(node, 'summary', 'No description')
                if len(summary) > 80:
                    summary = summary[:80] + "..."
                entity_list.append(f"{i}. {node.name}: {summary}")
                preview_items.append(f"[{i}] {node.name}:  \n{summary}")
            
            # Get user's language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            # Show confirmation dialog
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title=f"エンティティの削除確認 ({total_count}件)" if is_japanese else f"Confirm Entity Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。関連する関係性も削除されます。" if is_japanese else "⚠️ This operation cannot be undone. Related relationships will also be deleted.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            # Show compact result message (list only first 10)
            result_list = entity_list[:10]
            result = f"🔍 Found {total_count} entities:\n" + "\n".join(result_list)
            if total_count > 10:
                result += f"\n... and {total_count - 10} more"
            
            # Delete entities using Cypher
            entity_uuids = [node.uuid for node in entity_nodes]
            deleted_count = await self.helper.delete_nodes_by_uuids(entity_uuids)
            
            result += f"\n\n✅ Deleted {deleted_count} entities and their relationships"
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching/deleting entities: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_and_delete_facts(
        self,
        query: str,
        limit: int = 1,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for facts (relationships/edges) and delete them after user confirmation.
        
        This tool searches for relationships between entities (e.g., "John works at Company X")
        and allows you to delete them.
        
        IMPORTANT: This operation requires user confirmation and cannot be undone.
        
        :param query: Search query to find relationships (e.g., "works at", "friends with")
        :param limit: Maximum number of facts to return (default: 1, max: 100)
        :return: Result message with deleted facts count
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for facts using search_() which returns SearchResults
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract edges (facts/relationships)
            edges = search_results.edges
            
            if not edges:
                return f"ℹ️ No facts found matching '{query}'"
            
            # Show all facts for confirmation (no limit - user must see everything being deleted)
            total_count = len(edges)
            
            # Get user's language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            fact_list = []
            preview_items = []
            period_label = "期間" if is_japanese else "Period"
            for i, edge in enumerate(edges, 1):
                fact_text = getattr(edge, 'fact', 'Unknown relationship')
                if len(fact_text) > 80:
                    fact_text = fact_text[:80] + "..."
                valid_at = getattr(edge, 'valid_at', 'unknown')
                invalid_at = getattr(edge, 'invalid_at', 'present')
                fact_list.append(f"{i}. {fact_text} ({valid_at} - {invalid_at})")
                preview_items.append(f"[{i}] {fact_text}  \n{period_label}: {valid_at} - {invalid_at}")
            
            # Show confirmation dialog
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title=f"関係性の削除確認 ({total_count}件)" if is_japanese else f"Confirm Fact Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。" if is_japanese else "⚠️ This operation cannot be undone.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            # Show compact result message (list only first 10)
            result_list = fact_list[:10]
            result = f"🔍 Found {total_count} facts:\n" + "\n".join(result_list)
            if total_count > 10:
                result += f"\n... and {total_count - 10} more"
            
            # Delete edges using Cypher
            edge_uuids = [edge.uuid for edge in edges]
            deleted_count = await self.helper.delete_edges_by_uuids(edge_uuids)
            
            result += f"\n\n✅ Deleted {deleted_count} facts"
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching/deleting facts: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def search_and_delete_episodes(
        self,
        query: str,
        limit: int = 1,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for episodes (conversation history) and delete them after user confirmation.
        
        This tool searches for past conversations and allows you to delete them
        along with the entities and relationships extracted from them.
        
        IMPORTANT: This operation requires user confirmation and cannot be undone.
        
        :param query: Search query to find episodes (e.g., "conversation about Python")
        :param limit: Maximum number of episodes to return (default: 1, max: 100)
        :return: Result message with deleted episodes count
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        # Validate and clamp limit
        limit = max(1, min(100, limit))
        
        try:
            group_id = self.helper.get_group_id(__user__)
            
            # Create a copy of config with custom limit
            search_config = copy.copy(COMBINED_HYBRID_SEARCH_RRF)
            search_config.limit = limit
            
            # Search for episodes using search_() which returns SearchResults
            search_results = await self.helper.graphiti.search_(
                query=query,
                group_ids=[group_id] if group_id else None,
                config=search_config,
            )
            
            # Extract episodes
            episodes = search_results.episodes
            
            if not episodes:
                return f"ℹ️ No episodes found matching '{query}'"
            
            # Show all episodes for confirmation (no limit - user must see everything being deleted)
            total_count = len(episodes)
            
            # Get user's language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            episode_list = []
            preview_items = []
            created_label = "作成日時" if is_japanese else "Created"
            for i, episode in enumerate(episodes, 1):
                name = getattr(episode, 'name', 'Unknown episode')
                content = getattr(episode, 'content', '')
                if len(content) > 80:
                    content_preview = content[:80] + "..."
                else:
                    content_preview = content
                created_at = getattr(episode, 'created_at', 'unknown')
                episode_list.append(f"{i}. {name}: {content_preview} (created: {created_at})")
                preview_items.append(f"[{i}] {name}  \n{content_preview}  \n{created_label}: {created_at}")
            
            # Show confirmation dialog
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title=f"エピソードの削除確認 ({total_count}件)" if is_japanese else f"Confirm Episode Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。関連するエンティティと関係性も削除されます。" if is_japanese else "⚠️ This operation cannot be undone. Related entities and relationships will also be deleted.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            # Show compact result message (list only first 10)
            result_list = episode_list[:10]
            result = f"🔍 Found {total_count} episodes:\n" + "\n".join(result_list)
            if total_count > 10:
                result += f"\n... and {total_count - 10} more"
            
            # Delete episodes using remove_episode
            episode_uuids = [episode.uuid for episode in episodes]
            deleted_count = await self.helper.delete_episodes_by_uuids(episode_uuids)
            
            result += f"\n\n✅ Deleted {deleted_count} episodes"
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching/deleting episodes: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def delete_by_uuids(
        self,
        node_uuids: str = "",
        edge_uuids: str = "",
        episode_uuids: str = "",
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Delete specific nodes, edges, or episodes by their UUIDs after user confirmation.
        
        This tool allows precise deletion when you know the exact UUID of items to delete.
        UUIDs can be found in debug output or by using search tools first.
        
        IMPORTANT: This operation requires user confirmation and cannot be undone.
        
        :param node_uuids: Comma-separated list of node UUIDs to delete
        :param edge_uuids: Comma-separated list of edge UUIDs to delete
        :param episode_uuids: Comma-separated list of episode UUIDs to delete
        :return: Result message with deletion status
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        
        try:
            # Get user's language preference first
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            # Prepare preview items with actual content from database
            preview_items = []
            
            # Fetch and display node information
            if node_uuids.strip():
                uuids = [uuid.strip() for uuid in node_uuids.split(',') if uuid.strip()]
                for i, uuid in enumerate(uuids, 1):
                    try:
                        # Fetch node details from database
                        nodes = await EntityNode.get_by_uuids(self.helper.graphiti.driver, [uuid])
                        if nodes:
                            node = nodes[0]
                            name = getattr(node, 'name', 'Unknown')
                            summary = getattr(node, 'summary', 'No description')
                            if len(summary) > 80:
                                summary = summary[:80] + "..."
                            summary_label = "概要" if is_japanese else "Summary"
                            preview_items.append(f"[Node {i}] {name}  \nUUID: {uuid}  \n{summary_label}: {summary}")
                        else:
                            not_found_msg = "見つかりません" if is_japanese else "Not found"
                            preview_items.append(f"[Node {i}] ⚠️ {not_found_msg}  \nUUID: {uuid}")
                    except Exception as e:
                        error_msg = "詳細取得エラー" if is_japanese else "Error fetching details"
                        preview_items.append(f"[Node {i}] ⚠️ {error_msg}  \nUUID: {uuid}")
                        if self.valves.debug_print:
                            print(f"Error fetching node {uuid}: {e}")
            
            # Fetch and display edge information
            if edge_uuids.strip():
                uuids = [uuid.strip() for uuid in edge_uuids.split(',') if uuid.strip()]
                for i, uuid in enumerate(uuids, 1):
                    try:
                        # Fetch edge details from database
                        edges = await EntityEdge.get_by_uuids(self.helper.graphiti.driver, [uuid])
                        if edges:
                            edge = edges[0]
                            fact = getattr(edge, 'fact', 'Unknown relationship')
                            if len(fact) > 80:
                                fact = fact[:80] + "..."
                            valid_at = getattr(edge, 'valid_at', 'unknown')
                            invalid_at = getattr(edge, 'invalid_at', 'present')
                            period_label = "期間" if is_japanese else "Period"
                            preview_items.append(f"[Edge {i}] {fact}  \nUUID: {uuid}  \n{period_label}: {valid_at} → {invalid_at}")
                        else:
                            not_found_msg = "見つかりません" if is_japanese else "Not found"
                            preview_items.append(f"[Edge {i}] ⚠️ {not_found_msg}  \nUUID: {uuid}")
                    except Exception as e:
                        error_msg = "詳細取得エラー" if is_japanese else "Error fetching details"
                        preview_items.append(f"[Edge {i}] ⚠️ {error_msg}  \nUUID: {uuid}")
                        if self.valves.debug_print:
                            print(f"Error fetching edge {uuid}: {e}")
            
            # Fetch and display episode information
            if episode_uuids.strip():
                uuids = [uuid.strip() for uuid in episode_uuids.split(',') if uuid.strip()]
                for i, uuid in enumerate(uuids, 1):
                    try:
                        # Fetch episode details from database
                        episodes = await EpisodicNode.get_by_uuids(self.helper.graphiti.driver, [uuid])
                        if episodes:
                            episode = episodes[0]
                            name = getattr(episode, 'name', 'Unknown episode')
                            content = getattr(episode, 'content', '')
                            if len(content) > 80:
                                content = content[:80] + "..."
                            created_at = getattr(episode, 'created_at', 'unknown')
                            content_label = "内容" if is_japanese else "Content"
                            created_label = "作成" if is_japanese else "Created"
                            preview_items.append(f"[Episode {i}] {name}  \nUUID: {uuid}  \n{content_label}: {content}  \n{created_label}: {created_at}")
                        else:
                            not_found_msg = "見つかりません" if is_japanese else "Not found"
                            preview_items.append(f"[Episode {i}] ⚠️ {not_found_msg}  \nUUID: {uuid}")
                    except Exception as e:
                        error_msg = "詳細取得エラー" if is_japanese else "Error fetching details"
                        preview_items.append(f"[Episode {i}] ⚠️ {error_msg}  \nUUID: {uuid}")
                        if self.valves.debug_print:
                            print(f"Error fetching episode {uuid}: {e}")
            
            if not preview_items:
                return "ℹ️ No UUIDs provided for deletion"
            
            # Show confirmation dialog
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title="UUID指定削除の確認" if is_japanese else "Confirm UUID-based Deletion",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。UUIDを直接指定しての削除は慎重に行ってください。" if is_japanese else "⚠️ This operation cannot be undone. Please be careful when deleting by UUID.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            results = []
            
            # Delete nodes
            if node_uuids.strip():
                uuids = [uuid.strip() for uuid in node_uuids.split(',') if uuid.strip()]
                try:
                    deleted_count = await self.helper.delete_nodes_by_uuids(uuids)
                    results.append(f"✅ Deleted {deleted_count} node(s)")
                except Exception as e:
                    results.append(f"❌ Failed to delete nodes: {str(e)}")
            
            # Delete edges
            if edge_uuids.strip():
                uuids = [uuid.strip() for uuid in edge_uuids.split(',') if uuid.strip()]
                try:
                    deleted_count = await self.helper.delete_edges_by_uuids(uuids)
                    results.append(f"✅ Deleted {deleted_count} edge(s)")
                except Exception as e:
                    results.append(f"❌ Failed to delete edges: {str(e)}")
            
            # Delete episodes
            if episode_uuids.strip():
                uuids = [uuid.strip() for uuid in episode_uuids.split(',') if uuid.strip()]
                try:
                    deleted_count = await self.helper.delete_episodes_by_uuids(uuids)
                    results.append(f"✅ Deleted {deleted_count} episode(s)")
                except Exception as e:
                    results.append(f"❌ Failed to delete episodes: {str(e)}")
            
            if not results:
                return "ℹ️ No UUIDs provided for deletion"
            
            return "\n".join(results)
            
        except Exception as e:
            error_msg = f"❌ Error deleting by UUIDs: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
    
    async def clear_all_memory(
        self,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Clear ALL memory for the current user after confirmation. THIS CANNOT BE UNDONE!
        
        This tool deletes all entities, relationships, and episodes from your memory space.
        Use with extreme caution as this operation is irreversible.
        
        IMPORTANT: Requires user confirmation dialog before execution.
        
        :return: Result message with deletion status
        
        Note: __user__, __event_emitter__, and __event_call__ are automatically injected by the system.
        """
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, None)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        try:
            group_id = self.helper.get_group_id(__user__)
            
            if not group_id:
                return "❌ Error: Group ID is required for memory clearing. Please check your group_id_format configuration."
            
            # Count existing items using get_by_group_ids methods
            try:
                nodes = await EntityNode.get_by_group_ids(self.helper.graphiti.driver, [group_id])
                node_count = len(nodes)
            except Exception as e:
                if self.valves.debug_print:
                    print(f"Error counting nodes: {e}")
                node_count = 0
            
            try:
                edges = await EntityEdge.get_by_group_ids(self.helper.graphiti.driver, [group_id])
                edge_count = len(edges)
            except Exception as e:
                if self.valves.debug_print:
                    print(f"Error counting edges: {e}")
                edge_count = 0
            
            try:
                episodes = await EpisodicNode.get_by_group_ids(self.helper.graphiti.driver, [group_id])
                episode_count = len(episodes)
            except Exception as e:
                if self.valves.debug_print:
                    print(f"Error counting episodes: {e}")
                episode_count = 0
            
            if node_count == 0 and edge_count == 0 and episode_count == 0:
                return "ℹ️ Memory is already empty"
            
            # Get user's language preference
            is_japanese = self.helper.is_japanese_preferred(__user__)
            
            # Show confirmation dialog
            if is_japanese:
                preview_items = [
                    f"エンティティ(Entity): {node_count}個",
                    f"関係性(Fact): {edge_count}個",
                    f"エピソード(Episode): {episode_count}個",
                ]
            else:
                preview_items = [
                    f"Entities: {node_count} items",
                    f"Facts: {edge_count} items",
                    f"Episodes: {episode_count} items",
                ]
            
            confirmed, error_msg = await self.helper.show_confirmation_dialog(
                title="⚠️ 全メモリ削除の最終確認" if is_japanese else "⚠️ Final Confirmation: Clear All Memory",
                items=preview_items,
                warning_message="🔥 この操作は完全に元に戻せません！全てのメモリデータが永久に失われます。" if is_japanese else "🔥 This operation is completely irreversible! All memory data will be permanently lost.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                return error_msg
            
            # Require text input confirmation
            if __event_call__:
                try:
                    if is_japanese:
                        input_task = __event_call__(
                            {
                                "type": "input",
                                "data": {
                                    "title": "最終確認",
                                    "message": f"本当に全メモリ({node_count + edge_count + episode_count}件)を削除しますか？\n確認のため 'CLEAR_ALL_MEMORY' と入力してください。\n⏰ {self.valves.confirmation_timeout}秒以内に入力しないと自動的にキャンセルされます。",
                                    "placeholder": "CLEAR_ALL_MEMORY"
                                }
                            }
                        )
                    else:
                        input_task = __event_call__(
                            {
                                "type": "input",
                                "data": {
                                    "title": "Final Confirmation",
                                    "message": f"Are you sure you want to delete all memory ({node_count + edge_count + episode_count} items)?\nPlease type 'CLEAR_ALL_MEMORY' to confirm.\n⏰ Auto-cancel in {self.valves.confirmation_timeout} seconds if no input is provided.",
                                    "placeholder": "CLEAR_ALL_MEMORY"
                                }
                            }
                        )
                    
                    input_result = await asyncio.wait_for(input_task, timeout=self.valves.confirmation_timeout)
                    
                    if input_result != "CLEAR_ALL_MEMORY":
                        return "🚫 Confirmation text does not match. Memory clearing cancelled."
                except asyncio.TimeoutError:
                    return "🚫 Input timeout. Memory clearing cancelled."
                except Exception as e:
                    if self.valves.debug_print:
                        print(f"Input confirmation error: {e}")
                    return "🚫 Input confirmation cancelled."
            
            # Use Node.delete_by_group_id() - the correct method for clearing all data
            try:
                if self.valves.debug_print:
                    print(f"Deleting all data for group_id: {group_id}")
                
                await EntityNode.delete_by_group_id(self.helper.graphiti.driver, group_id)
                
                result = f"🗑️ All memory cleared:\n"
                result += f"  - {node_count} entities deleted\n"
                result += f"  - {edge_count} facts deleted\n"
                result += f"  - {episode_count} episodes deleted"
                
                return result
            except Exception as e:
                error_msg = f"❌ Error deleting memory: {str(e)}"
                if self.valves.debug_print:
                    traceback.print_exc()
                return error_msg
            
        except Exception as e:
            error_msg = f"❌ Error clearing memory: {str(e)}"
            if self.valves.debug_print:
                traceback.print_exc()
            return error_msg
