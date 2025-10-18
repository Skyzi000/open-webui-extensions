"""
title: Graphiti Memory
author: Skyzi000
description: Automatically identify and store valuable information from chats as Memories.
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-extensions
version: 0.5
requirements: graphiti-core[falkordb]

Design:
- Main class: Filter
- Related components:
  - Graphiti: Knowledge graph memory system
  - FalkorDriver: FalkorDB backend driver for graph storage
  - OpenAIClient: OpenAI client with JSON structured output support
  - OpenAIGenericClient: Generic OpenAI-compatible client
  - OpenAIEmbedder: Embedding model for semantic search
  - OpenAIRerankerClient: Cross-encoder for result reranking

Architecture:
- Initialization: _initialize_graphiti() sets up the graph database connection
- LLM Client Selection: Configurable client type selection
  - OpenAI client: Better for some providers/models
  - Generic client: Better for others
  - Try both to see which works better for your setup
- Search Strategy: Three performance/quality tradeoffs
  - Fast: BM25 only (~100ms, no embedding calls)
  - Balanced: BM25 + Cosine Similarity (~500ms, 1 embedding call) - DEFAULT
  - Quality: + Cross-Encoder reranking (~1-5s, multiple LLM calls)
- Lazy initialization: _ensure_graphiti_initialized() provides automatic retry
- Memory search: inlet() retrieves relevant memories before chat processing
- Memory storage: outlet() stores new information after chat completion
"""

import asyncio
import contextvars
import hashlib
import os
import re
import time
import traceback
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any
from urllib.parse import quote

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType
from openai import AsyncOpenAI
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    NodeSearchConfig,
    EpisodeSearchConfig,
    EdgeSearchMethod,
    NodeSearchMethod,
    EpisodeSearchMethod,
    EdgeReranker,
    NodeReranker,
    EpisodeReranker,
)
from graphiti_core.driver.falkordb_driver import FalkorDriver

# Context variable to store user-specific headers for each async request
# This ensures complete isolation between concurrent requests without locks
user_headers_context = contextvars.ContextVar('user_headers', default={})


class MultiUserOpenAIClient(OpenAIClient):
    """
    Custom OpenAI LLM client that retrieves user-specific headers from context variables.
    This allows a single Graphiti instance to safely handle concurrent requests from multiple users.
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
    """
    
    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig()
        
        # Store base client for dynamic header injection
        self._base_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        
        # Initialize parent with our base client
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

class Filter:
    """
    Open WebUI Filter for Graphiti-based memory management.
    
    Design References:
    - See module docstring for overall architecture
    - Graphiti documentation: https://github.com/getzep/graphiti-core
    
    Related Classes:
    - Valves: Configuration settings for the filter
    - UserValves: Per-user configuration settings
    
    Key Methods:
    - _initialize_graphiti(): Initialize the graph database connection
    - _ensure_graphiti_initialized(): Lazy initialization with retry logic
    - inlet(): Pre-process messages, inject relevant memories
    - outlet(): Post-process messages, store new memories
    
    Flow:
    1. User sends message → inlet() is called
    2. Search for relevant memories in graph database
    3. Inject found memories into conversation context
    4. LLM processes message with memory context
    5. outlet() is called with LLM response
    6. Extract and store new memories in graph database
    """
    class Valves(BaseModel):
        llm_client_type: str = Field(
            default="openai",
            description="Type of LLM client to use: 'openai' for OpenAI client, 'generic' for OpenAI-compatible generic client. Try both to see which works better with your LLM provider.",
        )
        openai_api_url: str = Field(
            default="https://api.openai.com/v1",
            description="openai compatible endpoint",
        )
        model: str = Field(
            default="gpt-5-mini",
            description="Model to use for memory processing.",
        )
        small_model: str = Field(
            default="gpt-5-nano",
            description="Smaller model to use for memory processing in legacy mode.",
        )
        embedding_model: str = Field(
            default="text-embedding-3-small",
            description="Model to use for embedding memories.",
        )
        embedding_dim: int = Field(
            default=1536, description="Dimension of the embedding model."
        )
        api_key: str = Field(
            default="", description="API key for OpenAI compatible endpoint"
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

        save_user_message: bool = Field(
            default=True,
            description="Automatically save user messages as memories",
        )
        
        save_assistant_response: bool = Field(
            default=False,
            description="Automatically save assistant responses (latest) as memories",
        )
        
        save_previous_assistant_message: bool = Field(
            default=True,
            description="Save the assistant message that the user is responding to (the one before the latest user message). This provides context about what the user is replying to.",
        )
        
        inject_facts: bool = Field(
            default=True,
            description="Inject facts (EntityEdge/relationships) from memory search results into the conversation. Facts represent relationships and events with temporal validity.",
        )
        
        inject_entities: bool = Field(
            default=True,
            description="Inject entities (EntityNode summaries) from memory search results into the conversation. Entities represent people, places, concepts with summarized information.",
        )
        
        update_communities: bool = Field(
            default=False,
            description="Update community detection when adding episodes using label propagation. EXPERIMENTAL: May cause errors with some Graphiti versions. Set to True to enable community updates.",
        )
        
        add_episode_timeout: int = Field(
            default=240,
            description="Timeout in seconds for adding episodes to memory. Set to 0 to disable timeout.",
        )
        
        semaphore_limit: int = Field(
            default=10,
            description="Maximum number of concurrent LLM operations in Graphiti. Default is 10 to prevent 429 rate limit errors. Increase for faster processing if your LLM provider allows higher throughput. Decrease if you encounter rate limit errors.",
        )
        
        max_search_message_length: int = Field(
            default=5000,
            description="Maximum length of user message to send to Graphiti search. Messages longer than this will be truncated (keeping first and last parts, dropping middle). Set to 0 to disable truncation. Note: This should be set to a size that the embedding model can handle to avoid errors.",
        )
        
        sanitize_search_query: bool = Field(
            default=True,
            description="Sanitize search queries to avoid FalkorDB/RediSearch syntax errors by removing special characters like @, :, \", (, ). Disable if you want to use raw queries or if using a different backend.",
        )
        
        search_strategy: str = Field(
            default="balanced",
            description="Search strategy: 'fast' (BM25 only, ~0.1s), 'balanced' (BM25+Cosine, ~0.5s), 'quality' (Cross-Encoder, ~1-5s)",
        )

        group_id_format: str = Field(
            default="{user_id}",
            description="Format string for group_id. Available placeholders: {user_id}, {user_email}, {user_name}. Email addresses are automatically sanitized (@ becomes _at_, . becomes _). Examples: '{user_id}', '{user_id}_chat', 'user_{user_id}'. Set to 'none' to disable group filtering (all users share the same memory space). Recommended: Use {user_id} (default) as it's stable; email/name changes could cause memory access issues.",
        )
        
        memory_message_role: str = Field(
            default="system",
            description="Role to use when injecting memory search results into the conversation. Options: 'system' (system message, more authoritative), 'user' (user message, more conversational). Default is 'system'.",
        )
        
        forward_user_info_headers: str = Field(
            default="default",
            description="Forward user information headers (User-Name, User-Id, User-Email, User-Role, Chat-Id) to OpenAI API. Options: 'default' (follow environment variable ENABLE_FORWARD_USER_INFO_HEADERS, defaults to false if not set), 'true' (always forward), 'false' (never forward).",
        )
        
        use_user_name_in_episode: bool = Field(
            default=True,
            description="Use actual user name instead of 'User' label when saving conversations to memory. When enabled, episodes will be saved as '{user_name}: {message}' instead of 'User: {message}'.",
        )
        
        debug_print: bool = Field(
            default=False,
            description="Enable debug printing to console. When enabled, prints detailed information about search results, memory injection, and processing steps. Useful for troubleshooting.",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable or disable Graphiti Memory feature for this user. When disabled, no memory search or storage will be performed.",
        )
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        save_user_message: str = Field(
            default="default",
            description="Automatically save user messages as memories. Options: 'default' (use global setting), 'true' (always save), 'false' (never save).",
        )
        save_assistant_response: str = Field(
            default="default",
            description="Automatically save assistant responses (latest) as memories. Options: 'default' (use global setting), 'true' (always save), 'false' (never save).",
        )
        
        save_previous_assistant_message: str = Field(
            default="default",
            description="Save the assistant message that the user is responding to (the one before the latest user message). Options: 'default' (use global setting), 'true' (always save), 'false' (never save).",
        )
        
        inject_facts: str = Field(
            default="default",
            description="Inject facts (EntityEdge/relationships) from memory search results. Options: 'default' (use global setting), 'true' (always inject), 'false' (never inject).",
        )
        
        inject_entities: str = Field(
            default="default",
            description="Inject entities (EntityNode summaries) from memory search results. Options: 'default' (use global setting), 'true' (always inject), 'false' (never inject).",
        )


    def __init__(self):
        self.valves = self.Valves()
        self.graphiti = None
        self._indices_built = False  # Track if indices have been built
        self._last_config = None  # Track configuration for change detection
        # Try to initialize, but it's okay if it fails - will retry later
        try:
            self._initialize_graphiti()
        except Exception as e:
            if self.valves.debug_print:
                print(f"Initial Graphiti initialization skipped (will retry on first use): {e}")
    
    def _get_config_hash(self) -> str:
        """
        Generate a hash of current configuration to detect changes.
        
        Returns:
            str: Hash of relevant configuration values
        """
        # Get all valve values as dict, excluding non-config fields
        valve_dict = self.valves.model_dump(
            exclude={
                'debug_print',  # Debugging settings don't affect initialization
                'group_id_format',  # Group ID format doesn't affect Graphiti init
                'search_strategy',  # Search strategy doesn't affect Graphiti init
                'save_assistant_response',  # Message saving behavior doesn't affect Graphiti init
                'inject_facts',  # Memory injection settings don't affect Graphiti init
                'inject_entities',  # Memory injection settings don't affect Graphiti init
                'update_communities',  # Community update setting doesn't affect Graphiti init
                'add_episode_timeout',  # Timeout settings don't affect Graphiti init
                'max_search_message_length',  # Message truncation doesn't affect Graphiti init
                'sanitize_search_query',  # Query sanitization doesn't affect Graphiti init
                'memory_message_role',  # Message role doesn't affect Graphiti init
                'forward_user_info_headers',  # Header forwarding doesn't affect Graphiti init
                'use_user_name_in_episode',  # Episode formatting doesn't affect Graphiti init
            }
        )
        # Sort keys for consistent hashing
        config_str = '|'.join(f"{k}={v}" for k, v in sorted(valve_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _config_changed(self) -> bool:
        """
        Check if configuration has changed since last initialization.
        
        Returns:
            bool: True if configuration changed, False otherwise
        """
        current_hash = self._get_config_hash()
        if self._last_config != current_hash:
            if self._last_config is not None:
                if self.valves.debug_print:
                    print(f"Configuration change detected, will reinitialize Graphiti")
            return True
        return False
    
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
    
    def _initialize_graphiti(self) -> bool:
        """
        Initialize Graphiti instance with configured backend.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'true' if self.valves.graphiti_telemetry_enabled else 'false'
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

            falkor_driver = None
            if self.valves.graph_db_backend.lower() == "falkordb":
                falkor_driver = FalkorDriver(
                    host=self.valves.falkordb_host,
                    port=self.valves.falkordb_port,
                    username=self.valves.falkordb_username,
                    password=self.valves.falkordb_password,
                )

            # Initialize Graphiti
            if falkor_driver:
                self.graphiti = Graphiti(
                    graph_driver=falkor_driver,
                    llm_client=llm_client,
                    embedder=MultiUserOpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=self.valves.api_key,
                            embedding_model=self.valves.embedding_model,
                            embedding_dim=self.valves.embedding_dim,
                            base_url=self.valves.openai_api_url,
                        )
                    ),
                    # OpenAIRerankerClient requires AsyncOpenAI client
                    # Use base_client from our custom multi-user client
                    cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
                )
            elif self.valves.graph_db_backend.lower() == "neo4j":
                self.graphiti = Graphiti(
                    self.valves.neo4j_uri,
                    self.valves.neo4j_user,
                    self.valves.neo4j_password,
                    llm_client=llm_client,
                    embedder=MultiUserOpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=self.valves.api_key,
                            embedding_model=self.valves.embedding_model,
                            embedding_dim=self.valves.embedding_dim,
                            base_url=self.valves.openai_api_url,
                        )
                    ),
                    # OpenAIRerankerClient requires AsyncOpenAI client
                    # Use base_client from our custom multi-user client
                    cross_encoder=OpenAIRerankerClient(client=llm_client._base_client, config=llm_config),
                )
            else:
                print(f"Unsupported graph database backend: {self.valves.graph_db_backend}. Supported backends are 'neo4j' and 'falkordb'.")
                return False
            
            # Save current configuration hash after successful initialization
            self._last_config = self._get_config_hash()
            if self.valves.debug_print:
                print("Graphiti initialized successfully.")
            return True
            
        except Exception as e:
            print(f"Graphiti initialization failed (will retry later if needed): {e}")
            # Only print traceback in debug scenarios
            # import traceback
            # traceback.print_exc()
            self.graphiti = None
            return False
    
    async def _build_indices(self) -> bool:
        """
        Build database indices and constraints for Graphiti.
        This should be called once after initialization and before the first query.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.graphiti is None:
            return False
            
        if self._indices_built:
            return True
            
        try:
            if self.valves.debug_print:
                print("Building Graphiti database indices and constraints...")
            await self.graphiti.build_indices_and_constraints()
            self._indices_built = True
            if self.valves.debug_print:
                print("Graphiti indices and constraints built successfully.")
            return True
        except Exception as e:
            print(f"Failed to build Graphiti indices: {e}")
            return False
    
    async def _ensure_graphiti_initialized(self) -> bool:
        """
        Ensure Graphiti is initialized and indices are built, attempting re-initialization if necessary.
        Automatically reinitializes if configuration changes are detected.
        
        Returns:
            bool: True if Graphiti is ready to use, False otherwise
        """
        # Check if configuration changed - if so, force reinitialization
        if self._config_changed():
            if self.valves.debug_print:
                print("Configuration changed, reinitializing Graphiti...")
            self.graphiti = None
            self._indices_built = False
        
        if self.graphiti is None:
            if self.valves.debug_print:
                print("Graphiti not initialized. Attempting to initialize...")
            if not self._initialize_graphiti():
                return False
        
        # Build indices if not already built
        if not self._indices_built:
            if not await self._build_indices():
                return False
        
        return True
    
    def _get_group_id(self, user: dict) -> Optional[str]:
        """
        Generate group_id for the user based on format string configuration.
        
        Args:
            user: User dictionary containing 'id' and optionally 'email', 'name'
            
        Returns:
            Sanitized group_id safe for Graphiti (alphanumeric, dashes, underscores only),
            or None if group_id_format is 'none' (to disable group filtering)
        """
        # Return None if format is 'none' (disable group filtering for shared memory space)
        if self.valves.group_id_format.lower().strip() == "none":
            return None
        
        # Prepare replacement values
        user_id = user.get('id', 'unknown')
        user_email = user.get('email', user_id)
        user_name = user.get('name', user_id)
        
        # Sanitize email to meet Graphiti's group_id requirements
        sanitized_email = user_email.replace('@', '_at_').replace('.', '_')
        
        # Sanitize name (replace spaces and special characters)
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', user_name)
        
        # Format the group_id using the template
        group_id = self.valves.group_id_format.format(
            user_id=user_id,
            user_email=sanitized_email,
            user_name=sanitized_name,
        )
        
        # Final sanitization to ensure only alphanumeric, dashes, underscores
        group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', group_id)
        
        return group_id
    
    def _get_search_config(self):
        """
        Get search configuration based on the configured search strategy.
        
        Returns:
            SearchConfig: Configured search strategy
        """
        strategy = self.valves.search_strategy.lower()
        
        if strategy == "fast":
            # BM25 only - fastest, no embedding calls
            return SearchConfig(
                edge_config=EdgeSearchConfig(
                    search_methods=[EdgeSearchMethod.bm25],
                    reranker=EdgeReranker.rrf,
                ),
                node_config=NodeSearchConfig(
                    search_methods=[NodeSearchMethod.bm25],
                    reranker=NodeReranker.rrf,
                ),
                episode_config=EpisodeSearchConfig(
                    search_methods=[EpisodeSearchMethod.bm25],
                    reranker=EpisodeReranker.rrf,
                ),
            )
        elif strategy == "quality":
            # Cross-encoder - highest quality, slowest
            return COMBINED_HYBRID_SEARCH_CROSS_ENCODER
        else:
            # Default: balanced (BM25 + Cosine Similarity + RRF)
            # Best speed/quality tradeoff for most use cases
            return COMBINED_HYBRID_SEARCH_RRF
    
    def _get_content_from_message(self, message: dict) -> Optional[str]:
        """
        Extract text content from a message, handling both string and list formats.
        
        Open WebUI messages can have content as:
        - Simple string: "Hello"
        - List with text and images: [{"type": "text", "text": "Hello"}, {"type": "image_url", ...}]
        
        Args:
            message: Message dictionary with 'content' field
            
        Returns:
            Extracted text content, or None if no text content found
        """
        content = message.get("content")
        
        # Handle list format (with images)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
            return ""  # No text found in list
        
        # Handle string format
        return content if isinstance(content, str) else ""
    
    def _sanitize_search_query(self, query: str) -> str:
        """
        Sanitize search query to avoid FalkorDB/RediSearch syntax errors.
        
        Only removes the most problematic characters that cause RediSearch errors.
        Keeps most punctuation to preserve query meaning.
        
        Args:
            query: The original search query
            
        Returns:
            Sanitized query safe for FalkorDB search
        """
        # Only remove the most problematic RediSearch operators:
        # ( ) - parentheses cause syntax errors with AND operator
        # @ - field selector
        # : - field separator  
        # " - quote operator
        # Keep: !, ?, ., ,, and other common punctuation
        sanitized = re.sub(r'[@:"()]', ' ', query)
        
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        if self.valves.debug_print:
            print(f"inlet:{__name__}")
            print(f"inlet:user:{__user__}")
        
        # Check if user has disabled the feature
        if __user__:
            user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
            if not user_valves.enabled:
                if self.valves.debug_print:
                    print("Graphiti Memory feature is disabled for this user.")
                return body
        
        # Check if graphiti is initialized, retry if not
        if not await self._ensure_graphiti_initialized() or self.graphiti is None:
            if self.valves.debug_print:
                print("Graphiti initialization failed. Skipping memory search.")
            if __user__ and __user__.get("valves", self.UserValves()).show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Memory service unavailable", "done": True},
                    }
                )
            return body
        
        # Set user headers in context variable (before any API calls)
        chat_id = __metadata__.get('chat_id') if __metadata__ else None
        headers = self._get_user_info_headers(__user__, chat_id)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")
        
        if __user__ is None:
            if self.valves.debug_print:
                print("User information is not available. Skipping memory search.")
            return body
        
        # Check if this is a "Continue Response" action
        # When user clicks "Continue Response" button, the last message is an assistant message
        # In this case, we should skip memory search to avoid injecting memories again
        messages = body.get("messages", [])
        if messages and messages[-1].get("role") == "assistant":
            if self.valves.debug_print:
                print("Detected 'Continue Response' action (last message is assistant). Skipping memory search.")
            return body
        
        # Find the last user message (ignore assistant/tool messages)
        user_message = None
        original_length = 0
        for msg in reversed(messages):
            if msg.get("role") == "user":
                # Extract text content from message (handles both string and list formats)
                user_message = self._get_content_from_message(msg)
                original_length = len(user_message) if user_message else 0
                break
        
        if not user_message:
            if self.valves.debug_print:
                print("No user message found. Skipping memory search.")
            return body
        
        # Sanitize query for FalkorDB/RediSearch compatibility (before truncation)
        sanitized_query = user_message
        if self.valves.sanitize_search_query:
            sanitized_query = self._sanitize_search_query(user_message)
            if not sanitized_query:
                if self.valves.debug_print:
                    print("Search query is empty after sanitization. Skipping memory search.")
                return body
            
            if sanitized_query != user_message:
                if self.valves.debug_print:
                    print(f"Search query sanitized: removed problematic characters")
        
        # Truncate message if too long (keep first and last parts, drop middle)
        original_length = len(sanitized_query)
        max_length = self.valves.max_search_message_length
        if max_length > 0 and len(sanitized_query) > max_length:
            keep_length = max_length // 2 - 25  # Leave room for separator
            sanitized_query = (
                sanitized_query[:keep_length] 
                + "\n\n[...]\n\n" 
                + sanitized_query[-keep_length:]
            )
            if self.valves.debug_print:
                print(f"User message truncated from {original_length} to {len(sanitized_query)} characters")
            
        user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
        
        # Get search configuration based on strategy
        search_config = self._get_search_config()
        
        if self.valves.debug_print:
            print(f"Using search strategy: {self.valves.search_strategy}")
        
        if user_valves.show_status:
            preview = sanitized_query[:100] + "..." if len(sanitized_query) > 100 else sanitized_query
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"🔍 Searching Graphiti: {preview}", "done": False},
                }
            )
        
        # Generate group_id using configured method (email or user ID)
        group_id = self._get_group_id(__user__)
        
        # Perform search with error handling for FalkorDB/RediSearch syntax issues
        # Measure search time
        search_start_time = time.time()
        
        try:
            # Use search_() for advanced search that returns SearchResults with nodes, edges, episodes, communities
            # Search configuration is determined by search_strategy setting
            # Only pass group_ids if group_id is not None
            if group_id is not None:
                results = await self.graphiti.search_(
                    query=sanitized_query,
                    group_ids=[group_id],
                    config=search_config,
                )
            else:
                results = await self.graphiti.search_(
                    query=sanitized_query,
                    config=search_config,
                )
            
            # Calculate search duration
            search_duration = time.time() - search_start_time
            
            if self.valves.debug_print:
                print(f"Search completed in {search_duration:.2f}s")
        except Exception as e:
            search_duration = time.time() - search_start_time
            error_msg = str(e)
            if "Syntax error" in error_msg or "RediSearch" in error_msg:
                print(f"FalkorDB/RediSearch syntax error during search (after {search_duration:.2f}s): {error_msg}")
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Memory search unavailable (syntax error, {search_duration:.2f}s)", "done": True},
                        }
                    )
            else:
                print(f"Unexpected error during Graphiti search (after {search_duration:.2f}s): {e}")
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Memory search failed ({search_duration:.2f}s)", "done": True},
                        }
                    )
            return body
        
        # Check if any results were found
        if len(results.edges) == 0 and len(results.nodes) == 0:
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"No relevant memories found ({search_duration:.2f}s)", "done": True},
                    }
                )
            return body

        # Determine whether to inject facts and entities based on settings
        # Use UserValves setting if available, otherwise fall back to Valves setting
        user_inject_facts_setting = user_valves.inject_facts.lower()
        if user_inject_facts_setting == "default":
            should_inject_facts = self.valves.inject_facts
        elif user_inject_facts_setting == "true":
            should_inject_facts = True
        elif user_inject_facts_setting == "false":
            should_inject_facts = False
        else:
            # Invalid value, use global setting as fallback
            should_inject_facts = self.valves.inject_facts
        
        user_inject_entities_setting = user_valves.inject_entities.lower()
        if user_inject_entities_setting == "default":
            should_inject_entities = self.valves.inject_entities
        elif user_inject_entities_setting == "true":
            should_inject_entities = True
        elif user_inject_entities_setting == "false":
            should_inject_entities = False
        else:
            # Invalid value, use global setting as fallback
            should_inject_entities = self.valves.inject_entities

        # Print search results (if debug mode enabled)
        if self.valves.debug_print:
            print('\nSearch Results:')

        facts = []
        entities = {}  # Dictionary to store unique entities: {name: summary}
        
        # Process EntityEdge results (relations/facts) only if enabled
        if should_inject_facts:
            for idx, result in enumerate(results.edges, 1):
                if self.valves.debug_print:
                    print(f'Edge UUID: {result.uuid}')
                    print(f'Fact({result.name}): {result.fact}')
                    if hasattr(result, 'valid_at') and result.valid_at:
                        print(f'Valid from: {result.valid_at}')
                    if hasattr(result, 'invalid_at') and result.invalid_at:
                        print(f'Valid until: {result.invalid_at}')

                facts.append((result.fact, result.valid_at, result.invalid_at, result.name))
                
                # Emit status for each fact found
                if user_valves.show_status:
                    emoji = "🔚" if result.invalid_at else "🔛"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"{emoji} Fact {idx}/{len(results.edges)}: {result.fact}", "done": False},
                        }
                    )
                
                if self.valves.debug_print:
                    print('---')
        else:
            if self.valves.debug_print:
                print(f'Skipping {len(results.edges)} facts (inject_facts is disabled)')
        
        # Process EntityNode results (entities with summaries) only if enabled
        if should_inject_entities:
            for idx, result in enumerate(results.nodes, 1):
                if self.valves.debug_print:
                    print(f'Node UUID: {result.uuid}')
                    print(f'Entity({result.name}): {result.summary}')
                
                # Store entity information
                if result.name and result.summary:
                    entities[result.name] = result.summary
                    
                    # Emit status for each entity found
                    if user_valves.show_status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": f"👤 Entity {idx}/{len(results.nodes)}: {result.name} - {result.summary}", "done": False},
                            }
                        )
                
                if self.valves.debug_print:
                    print('---')
        else:
            if self.valves.debug_print:
                print(f'Skipping {len(results.nodes)} entities (inject_entities is disabled)')

            
        # Insert memory message if we have facts OR entities
        if len(facts) > 0 or len(entities) > 0:
            # Find the index of the last user message
            last_user_msg_index = None
            for i in range(len(body['messages']) - 1, -1, -1):
                if body['messages'][i].get("role") == "user":
                    last_user_msg_index = i
                    break
            
            # Determine the role to use for memory message (default to system if invalid value)
            memory_role = self.valves.memory_message_role.lower()
            if memory_role not in ["system", "user"]:
                if self.valves.debug_print:
                    print(f"Invalid memory_message_role '{memory_role}', using 'system'")
                memory_role = "system"
            
            # Format memory content with improved structure
            memory_content = "FACTS and ENTITIES represent relevant context to the current conversation.  \n"
            
            # Add facts section if any facts were found
            if len(facts) > 0:
                memory_content += "# These are the most relevant facts and their valid date ranges  \n"
                memory_content += "# format: FACT (Date range: from - to)  \n"
                memory_content += "<FACTS>  \n"
                
                for fact, valid_at, invalid_at, name in facts:
                    # Format date range
                    valid_str = str(valid_at) if valid_at else "unknown"
                    invalid_str = str(invalid_at) if invalid_at else "present"
                    
                    memory_content += f"  - {fact} ({valid_str} - {invalid_str})  \n"
                
                memory_content += "</FACTS>"
            
            # Add entities section if any entities were found
            if len(entities) > 0:
                if len(facts) > 0:
                    memory_content += "  \n\n"  # Add spacing between sections
                memory_content += "# These are the most relevant entities  \n"
                memory_content += "# ENTITY_NAME: entity summary  \n"
                memory_content += "<ENTITIES>  \n"
                
                for entity_name, entity_summary in entities.items():
                    memory_content += f"  - {entity_name}: {entity_summary}  \n"
                
                memory_content += "</ENTITIES>"
            
            # Insert memory before the last user message
            memory_message = {
                "role": memory_role,
                "content": memory_content
            }
            
            if last_user_msg_index is not None:
                body['messages'].insert(last_user_msg_index, memory_message)
            else:
                # Fallback: if no user message found, append to end
                body['messages'].append(memory_message)
            
            if user_valves.show_status:
                # Build status message showing what was found
                status_parts = []
                if len(facts) > 0:
                    status_parts.append(f"{len(facts)} fact{'s' if len(facts) != 1 else ''}")
                if len(entities) > 0:
                    status_parts.append(f"{len(entities)} entit{'ies' if len(entities) != 1 else 'y'}")
                
                status_msg = "🧠 " + " and ".join(status_parts) + f" found ({search_duration:.2f}s)"
                
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_msg, "done": True},
                    }
                )
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        # Check if user has disabled the feature
        if __user__:
            user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
            if not user_valves.enabled:
                if self.valves.debug_print:
                    print("Graphiti Memory feature is disabled for this user.")
                return body
        
        # Check if graphiti is initialized, retry if not
        if not await self._ensure_graphiti_initialized() or self.graphiti is None:
            if self.valves.debug_print:
                print("Graphiti initialization failed. Skipping memory addition.")
            return body
            
        if __user__ is None:
            if self.valves.debug_print:
                print("User information is not available. Skipping memory addition.")
            return body
        chat_id = __metadata__.get('chat_id', 'unknown') if __metadata__ else 'unknown'
        message_id = __metadata__.get('message_id', 'unknown') if __metadata__ else 'unknown'
        if self.valves.debug_print:
            print(f"outlet:{__name__}, chat_id:{chat_id}, message_id:{message_id}")
        
        # Set user headers in context variable (before any API calls)
        headers = self._get_user_info_headers(__user__, chat_id)
        if headers:
            user_headers_context.set(headers)
            if self.valves.debug_print:
                print(f"Set user headers in context: {list(headers.keys())}")


        user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
        messages = body.get("messages", [])
        if len(messages) == 0:
            return body
        
        # Determine which messages to save based on settings
        messages_to_save = []
        
        # Find the last user message, last assistant message, and previous assistant message
        last_user_message = None
        last_assistant_message = None
        previous_assistant_message = None
        
        for msg in reversed(messages):
            if msg.get("role") == "user" and last_user_message is None:
                last_user_message = msg
            elif msg.get("role") == "assistant":
                if last_user_message is None:
                    # This is after the last user message (the latest assistant response)
                    if last_assistant_message is None:
                        last_assistant_message = msg
                else:
                    # This is before the last user message (the assistant message user is responding to)
                    if previous_assistant_message is None:
                        previous_assistant_message = msg
                        break  # We found everything we need
        
        # Save previous assistant message based on setting (the one user is responding to)
        # Use UserValves setting if available, otherwise fall back to Valves setting
        user_save_previous_assistant_setting = user_valves.save_previous_assistant_message.lower()
        if user_save_previous_assistant_setting == "default":
            save_previous_assistant = self.valves.save_previous_assistant_message
        elif user_save_previous_assistant_setting == "true":
            save_previous_assistant = True
        elif user_save_previous_assistant_setting == "false":
            save_previous_assistant = False
        else:
            # Invalid value, use global setting as fallback
            save_previous_assistant = self.valves.save_previous_assistant_message
        
        if save_previous_assistant and previous_assistant_message:
            previous_assistant_content = self._get_content_from_message(previous_assistant_message)
            if previous_assistant_content:
                messages_to_save.append(("previous_assistant", previous_assistant_content))
        
        # Save user messages based on setting
        # Use UserValves setting if available, otherwise fall back to Valves setting
        user_save_user_setting = user_valves.save_user_message.lower()
        if user_save_user_setting == "default":
            save_user = self.valves.save_user_message
        elif user_save_user_setting == "true":
            save_user = True
        elif user_save_user_setting == "false":
            save_user = False
        else:
            # Invalid value, use global setting as fallback
            save_user = self.valves.save_user_message
        
        if save_user and last_user_message:
            user_content = self._get_content_from_message(last_user_message)
            if user_content:
                messages_to_save.append(("user", user_content))
        
        # Save assistant responses based on setting
        # Use UserValves setting if available, otherwise fall back to Valves setting
        user_save_assistant_setting = user_valves.save_assistant_response.lower()
        if user_save_assistant_setting == "default":
            save_assistant = self.valves.save_assistant_response
        elif user_save_assistant_setting == "true":
            save_assistant = True
        elif user_save_assistant_setting == "false":
            save_assistant = False
        else:
            # Invalid value, use global setting as fallback
            save_assistant = self.valves.save_assistant_response
        
        if save_assistant and last_assistant_message:
            assistant_content = self._get_content_from_message(last_assistant_message)
            if assistant_content:
                messages_to_save.append(("assistant", assistant_content))
        
        if len(messages_to_save) == 0:
            return body
        
        # Construct episode body in "Assistant: {message}\nUser: {message}\nAssistant: {message}" format for EpisodeType.message
        # Sort messages to maintain chronological order: previous_assistant -> user -> assistant
        role_order = {"previous_assistant": 0, "user": 1, "assistant": 2}
        messages_to_save.sort(key=lambda x: role_order.get(x[0], 99))
        
        episode_parts = []
        for role, content in messages_to_save:
            if role == "user":
                # Use actual user name if enabled, otherwise use "User"
                if self.valves.use_user_name_in_episode and __user__.get('name'):
                    role_label = __user__['name']
                else:
                    role_label = "User"
            elif role in ("assistant", "previous_assistant"):
                role_label = "Assistant"
            else:
                role_label = role.capitalize()
            episode_parts.append(f"{role_label}: {content}")
        
        episode_body = "\n".join(episode_parts)
        
        if user_valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"✍️ Adding conversation to Graphiti memory...", "done": False},
                }
            )
        
        # Generate group_id using configured method (email or user ID)
        group_id = self._get_group_id(__user__)
        saved_count = 0
        
        # Store add_episode results for detailed status display
        add_results = None
        
        try:
            # Apply timeout if configured
            if self.valves.add_episode_timeout > 0:
                if group_id is not None:
                    add_results = await asyncio.wait_for(
                        self.graphiti.add_episode(
                            name=f"Chat_Interaction_{chat_id}_{message_id}",
                            episode_body=episode_body,
                            source=EpisodeType.message,
                            source_description="Chat conversation",
                            reference_time=datetime.now(),
                            group_id=group_id,
                            update_communities=self.valves.update_communities,
                        ),
                        timeout=self.valves.add_episode_timeout
                    )
                else:
                    add_results = await asyncio.wait_for(
                        self.graphiti.add_episode(
                            name=f"Chat_Interaction_{chat_id}_{message_id}",
                            episode_body=episode_body,
                            source=EpisodeType.message,
                            source_description="Chat conversation",
                            reference_time=datetime.now(),
                            update_communities=self.valves.update_communities,
                        ),
                        timeout=self.valves.add_episode_timeout
                    )
            else:
                if group_id is not None:
                    add_results = await self.graphiti.add_episode(
                        name=f"Chat_Interaction_{chat_id}_{message_id}",
                        episode_body=episode_body,
                        source=EpisodeType.message,
                        source_description="Chat conversation",
                        reference_time=datetime.now(),
                        group_id=group_id,
                        update_communities=self.valves.update_communities,
                    )
                else:
                    add_results = await self.graphiti.add_episode(
                        name=f"Chat_Interaction_{chat_id}_{message_id}",
                        episode_body=episode_body,
                        source=EpisodeType.message,
                        source_description="Chat conversation",
                        reference_time=datetime.now(),
                        update_communities=self.valves.update_communities,
                    )
            if self.valves.debug_print:
                print(f"Added conversation to Graphiti memory: {episode_body[:100]}...")
                if add_results:
                    print(f"Extracted {len(add_results.nodes)} entities and {len(add_results.edges)} relationships")
            saved_count = 1

            # Display extracted entities and facts in status
            if user_valves.show_status and add_results:
                # Show all Facts
                if add_results.edges:
                    for idx, edge in enumerate(add_results.edges, 1):
                        emoji = "🔚" if edge.invalid_at else "🔛"
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": f"{emoji} Fact {idx}/{len(add_results.edges)}: {edge.fact}", "done": False},
                            }
                        )
                
                # Show all entities
                if add_results.nodes:
                    for idx, node in enumerate(add_results.nodes, 1):
                        # Display entity name and summary (if available)
                        entity_display = f"{node.name}"
                        if hasattr(node, 'summary') and node.summary:
                            entity_display += f" - {node.summary}"
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": f"👤 Entity {idx}/{len(add_results.nodes)}: {entity_display}", "done": False},
                            }
                        )
        except asyncio.TimeoutError:
            print(f"Timeout adding conversation to Graphiti memory after {self.valves.add_episode_timeout}s")
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Warning: Memory save timed out", "done": False},
                    }
                )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Provide more specific error messages for common issues
            if "ValidationError" in error_type:
                print(f"Graphiti LLM response validation error for conversation: {error_msg}")
                user_msg = "Graphiti: LLM response format error (will retry on next message)"
            elif "ConnectionError" in error_type or "timeout" in error_msg.lower():
                print(f"Graphiti connection error adding conversation: {error_msg}")
                user_msg = "Graphiti: Connection error (temporary)"
            else:
                print(f"Graphiti error adding conversation: {e}")
                user_msg = f"Graphiti: Memory save failed ({error_type})"
            
            # Only print full traceback for unexpected errors
            if "ValidationError" not in error_type:
                traceback.print_exc()
            
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Warning: {user_msg}", "done": False},
                    }
                )
        
        # Only increment count for successfully saved messages
        if saved_count > 0:
            pass  # Successfully saved messages

        if user_valves.show_status:
            if saved_count == 0:
                status_msg = "❌ Failed to save conversation to Graphiti memory"
            else:
                # Build detailed status message with entity and relationship counts
                status_parts = ["✅ Added conversation to Graphiti memory"]
                if add_results:
                    detail_parts = []
                    if add_results.nodes:
                        detail_parts.append(f"{len(add_results.nodes)} entit{'ies' if len(add_results.nodes) != 1 else 'y'}")
                    if add_results.edges:
                        detail_parts.append(f"{len(add_results.edges)} relation{'s' if len(add_results.edges) != 1 else ''}")
                    if detail_parts:
                        status_msg = " - ".join(status_parts + [" and ".join(detail_parts) + " extracted"])
                    else:
                        status_msg = status_parts[0]
                else:
                    status_msg = status_parts[0]
            
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": status_msg, "done": True},
                }
            )
        
        return body
