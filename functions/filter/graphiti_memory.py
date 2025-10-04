"""
title: Graphiti Memory
author: Skyzi000
description: Automatically identify and store valuable information from chats as Memories.
author_email: dev@skyzi.jp
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-extensions
version: 0.2
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
- Lazy initialization: _ensure_graphiti_initialized() provides automatic retry
- Memory search: inlet() retrieves relevant memories before chat processing
- Memory storage: outlet() stores new information after chat completion
"""

import ast
import json
import os
import time
import asyncio
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any
from urllib.parse import quote

import aiohttp
from aiohttp import ClientError
from fastapi.requests import Request
from pydantic import BaseModel, Field

from openai import AsyncOpenAI

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.driver.falkordb_driver import FalkorDriver

from open_webui.main import app as webui_app
from open_webui.models.users import Users, UserModel
from open_webui.routers.memories import (
    add_memory,
    AddMemoryForm,
    delete_memory_by_id,
    query_memory,
    QueryMemoryForm,
)


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
            default="falkordb",
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

        save_assistant_response: bool = Field(
            default=False,
            description="Automatically save assistant responses as memories",
        )
        
        update_communities: bool = Field(
            default=False,
            description="Update community detection when adding episodes using label propagation. EXPERIMENTAL: May cause errors with some Graphiti versions. Set to True to enable community updates.",
        )
        
        add_episode_timeout: int = Field(
            default=120,
            description="Timeout in seconds for adding episodes to memory. Set to 0 to disable timeout.",
        )
        
        semaphore_limit: int = Field(
            default=10,
            description="Maximum number of concurrent LLM operations in Graphiti. Default is 10 to prevent 429 rate limit errors. Increase for faster processing if your LLM provider allows higher throughput. Decrease if you encounter rate limit errors.",
        )
        
        max_search_message_length: int = Field(
            default=10000,
            description="Maximum length of user message to send to Graphiti search. Messages longer than this will be truncated (keeping first and last parts, dropping middle). Set to 0 to disable truncation.",
        )
        
        sanitize_search_query: bool = Field(
            default=True,
            description="Sanitize search queries to avoid FalkorDB/RediSearch syntax errors by removing special characters like @, :, \", (, ). Disable if you want to use raw queries or if using a different backend.",
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

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable or disable Graphiti Memory feature for this user. When disabled, no memory search or storage will be performed.",
        )
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        save_assistant_response: str = Field(
            default="default",
            description="Automatically save assistant responses as memories. Options: 'default' (use global setting), 'true' (always save), 'false' (never save).",
        )


    def __init__(self):
        self.valves = self.Valves()
        self.graphiti = None
        self._indices_built = False  # Track if indices have been built
        self._last_config = None  # Track configuration for change detection
        self._current_user_headers = {}  # Track current user info headers
        # Try to initialize, but it's okay if it fails - will retry later
        try:
            self._initialize_graphiti()
        except Exception as e:
            print(f"Initial Graphiti initialization skipped (will retry on first use): {e}")
    
    def _get_config_hash(self) -> str:
        """
        Generate a hash of current configuration to detect changes.
        
        Returns:
            str: Hash of relevant configuration values
        """
        import hashlib
        config_str = f"{self.valves.llm_client_type}|{self.valves.openai_api_url}|{self.valves.model}|{self.valves.small_model}|{self.valves.embedding_model}|{self.valves.embedding_dim}|{self.valves.api_key}|{self.valves.graph_db_backend}|{self.valves.falkordb_host}|{self.valves.falkordb_port}|{self.valves.neo4j_uri}|{self.valves.semaphore_limit}"
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
    
    def _update_llm_client_headers(self, headers: dict) -> None:
        """
        Update LLM client with extra headers using AsyncOpenAI's copy() method.
        
        Args:
            headers: Dictionary of headers to add to all requests
        """
        if not headers or self.graphiti is None or self.graphiti.llm_client is None:
            return
        
        # Only update if headers changed
        if headers == self._current_user_headers:
            return
        
        self._current_user_headers = headers
        
        try:
            llm_client = self.graphiti.llm_client
            
            # OpenAIClient and OpenAIGenericClient both have a 'client' attribute that is AsyncOpenAI
            if hasattr(llm_client, 'client'):
                old_client = llm_client.client  # type: ignore
                
                # Use AsyncOpenAI's copy() method to create a new client with additional headers
                # This preserves all existing settings while adding our headers
                new_client = old_client.copy(default_headers=headers)  # type: ignore
                
                # Replace the client
                llm_client.client = new_client  # type: ignore
                print(f"Updated OpenAI client with headers using copy(): {list(headers.keys())}")
            else:
                print("Warning: LLM client does not have 'client' attribute")
        
        except Exception as e:
            print(f"Failed to update OpenAI client headers: {e}")
            import traceback
            traceback.print_exc()


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

            # Select LLM client based on configuration
            if self.valves.llm_client_type.lower() == "openai":
                llm_client = OpenAIClient(config=llm_config)
                print("Using OpenAI client")
            elif self.valves.llm_client_type.lower() == "generic":
                llm_client = OpenAIGenericClient(config=llm_config)
                print("Using OpenAI-compatible generic client")
            else:
                # Default to OpenAI client for unknown values
                llm_client = OpenAIClient(config=llm_config)
                print(f"Unknown client type '{self.valves.llm_client_type}', defaulting to OpenAI client")

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
                    embedder=OpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=self.valves.api_key,
                            embedding_model=self.valves.embedding_model,
                            embedding_dim=self.valves.embedding_dim,
                            base_url=self.valves.openai_api_url,
                        )
                    ),
                    cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
                )
            elif self.valves.graph_db_backend.lower() == "neo4j":
                self.graphiti = Graphiti(
                    self.valves.neo4j_uri,
                    self.valves.neo4j_user,
                    self.valves.neo4j_password,
                    llm_client=llm_client,
                    embedder=OpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=self.valves.api_key,
                            embedding_model=self.valves.embedding_model,
                            embedding_dim=self.valves.embedding_dim,
                            base_url=self.valves.openai_api_url,
                        )
                    ),
                    cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
                )
            else:
                print(f"Unsupported graph database backend: {self.valves.graph_db_backend}. Supported backends are 'neo4j' and 'falkordb'.")
                return False
            
            # Save current configuration hash after successful initialization
            self._last_config = self._get_config_hash()
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
            print("Building Graphiti database indices and constraints...")
            await self.graphiti.build_indices_and_constraints()
            self._indices_built = True
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
            print("Configuration changed, reinitializing Graphiti...")
            self.graphiti = None
            self._indices_built = False
        
        if self.graphiti is None:
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
        import re
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
        import re
        
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
        print(f"inlet:{__name__}")
        print(f"inlet:user:{__user__}")
        
        # Check if user has disabled the feature
        if __user__:
            user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
            if not user_valves.enabled:
                print("Graphiti Memory feature is disabled for this user.")
                return body
        
        # Check if graphiti is initialized, retry if not
        if not await self._ensure_graphiti_initialized() or self.graphiti is None:
            print("Graphiti initialization failed. Skipping memory search.")
            if __user__ and __user__.get("valves", self.UserValves()).show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Memory service unavailable", "done": True},
                    }
                )
            return body
        
        # Update LLM client headers with user info (before any API calls)
        chat_id = __metadata__.get('chat_id') if __metadata__ else None
        headers = self._get_user_info_headers(__user__, chat_id)
        if headers:
            self._update_llm_client_headers(headers)
        
        if __user__ is None:
            print("User information is not available. Skipping memory search.")
            return body
        
        # Find the last user message (ignore assistant/tool messages)
        user_message = None
        original_length = 0
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                original_length = len(user_message)
                break
        
        if not user_message:
            print("No user message found. Skipping memory search.")
            return body
        
        # Sanitize query for FalkorDB/RediSearch compatibility (before truncation)
        sanitized_query = user_message
        if self.valves.sanitize_search_query:
            sanitized_query = self._sanitize_search_query(user_message)
            if not sanitized_query:
                print("Search query is empty after sanitization. Skipping memory search.")
                return body
            
            if sanitized_query != user_message:
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
            print(f"User message truncated from {original_length} to {len(sanitized_query)} characters")
            
        user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
        if user_valves.show_status:
            preview = sanitized_query[:100] + "..." if len(sanitized_query) > 100 else sanitized_query
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Searching Graphiti: {preview}", "done": False},
                }
            )
        
        # Generate group_id using configured method (email or user ID)
        group_id = self._get_group_id(__user__)
        
        # Perform search with error handling for FalkorDB/RediSearch syntax issues
        try:
            # Only pass group_ids if group_id is not None
            if group_id is not None:
                results = await self.graphiti.search(
                    query=sanitized_query,
                    group_ids=[group_id],
                )
            else:
                results = await self.graphiti.search(
                    query=sanitized_query,
                )
        except Exception as e:
            error_msg = str(e)
            if "Syntax error" in error_msg or "RediSearch" in error_msg:
                print(f"FalkorDB/RediSearch syntax error during search: {error_msg}")
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Memory search unavailable (syntax error)", "done": True},
                        }
                    )
            else:
                print(f"Unexpected error during Graphiti search: {e}")
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Memory search failed", "done": True},
                        }
                    )
            return body
        
        if len(results) == 0:
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "No relevant memories found", "done": True},
                    }
                )
            return body

        # Print search results

        print('\nSearch Results:')

        facts = []
        id = 0
        
        for result in results:
            
            print(f'UUID: {result.uuid}')

            print(f'Fact({result.name}): {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')

            facts.append((result.fact, result.valid_at, result.invalid_at, result.name))

            print('---')
            # # Emit citation for each memory
            # await __event_emitter__(
            #     {
            #         "type": "citation",
            #         "data": {
            #             "source": {
            #                 "name": "Graphiti Memory",
            #                 "id": str(result.uuid),
            #             },
            #             "document": [result.fact],
            #             "metadata": [
            #                 {
            #                     "source": "Graphiti Memory",
            #                     "parameters": {
            #                         "source": "Graphiti Memory",
            #                         "name": str(result.name),
            #                         "group_id": group_id,
            #                         "uuid": str(result.uuid),
            #                         "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
            #                         "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None,
            #                         "attributes": str(result.attributes) if hasattr(result, 'attributes') and result.attributes else None,
            #                         "source_node_uuid": str(result.source_node_uuid) if hasattr(result, 'source_node_uuid') and result.source_node_uuid else None,
            #                         "target_node_uuid": str(result.target_node_uuid) if hasattr(result, 'target_node_uuid') and result.target_node_uuid else None,
            #                     }
            #                 }
            #             ]
            #         }
            #     }
            # )
            
        if len(facts) > 0:
            # Find the index of the last user message
            last_user_msg_index = None
            for i in range(len(body['messages']) - 1, -1, -1):
                if body['messages'][i].get("role") == "user":
                    last_user_msg_index = i
                    break
            
            # Determine the role to use for memory message (default to system if invalid value)
            memory_role = self.valves.memory_message_role.lower()
            if memory_role not in ["system", "user"]:
                print(f"Invalid memory_message_role '{memory_role}', using 'system'")
                memory_role = "system"
            
            # Insert memory before the last user message
            memory_message = {
                "role": memory_role,
                "content": f"Graphiti memories were found:\n" + "\n".join([f"- {name}: {fact} (valid_at: {valid_at}, invalid_at: {invalid_at})" for fact, valid_at, invalid_at, name in facts])
            }
            
            if last_user_msg_index is not None:
                body['messages'].insert(last_user_msg_index, memory_message)
            else:
                # Fallback: if no user message found, append to end
                body['messages'].append(memory_message)
            
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"{len(facts)} memories found: {', '.join([fact for fact, _, _, _ in facts])}", "done": True},
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
                print("Graphiti Memory feature is disabled for this user.")
                return body
        
        # Check if graphiti is initialized, retry if not
        if not await self._ensure_graphiti_initialized() or self.graphiti is None:
            print("Graphiti initialization failed. Skipping memory addition.")
            return body
            
        if __user__ is None:
            print("User information is not available. Skipping memory addition.")
            return body
        chat_id = __metadata__.get('chat_id', 'unknown') if __metadata__ else 'unknown'
        message_id = __metadata__.get('message_id', 'unknown') if __metadata__ else 'unknown'
        print(f"outlet:{__name__}, chat_id:{chat_id}, message_id:{message_id}")
        
        # Update LLM client headers with user info (before any API calls)
        headers = self._get_user_info_headers(__user__, chat_id)
        if headers:
            self._update_llm_client_headers(headers)


        user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
        messages = body.get("messages", [])
        if len(messages) == 0:
            return body
        
        # Determine which messages to save based on save_assistant_response setting
        messages_to_save = []
        
        # Find the last user message
        last_user_message = None
        last_assistant_message = None
        
        for msg in reversed(messages):
            if msg.get("role") == "user" and last_user_message is None:
                last_user_message = msg
            elif msg.get("role") == "assistant" and last_assistant_message is None:
                last_assistant_message = msg
            
            if last_user_message and last_assistant_message:
                break
        
        # Always save user messages
        if last_user_message:
            messages_to_save.append(("user", last_user_message["content"]))
        
        # Optionally save assistant responses
        # Use UserValves setting if available, otherwise fall back to Valves setting
        user_save_setting = user_valves.save_assistant_response.lower()
        if user_save_setting == "default":
            save_assistant = self.valves.save_assistant_response
        elif user_save_setting == "true":
            save_assistant = True
        elif user_save_setting == "false":
            save_assistant = False
        else:
            # Invalid value, use global setting as fallback
            save_assistant = self.valves.save_assistant_response
        
        if save_assistant and last_assistant_message:
            messages_to_save.append(("assistant", last_assistant_message["content"]))
        
        if len(messages_to_save) == 0:
            return body
        
        # Construct episode body in "User: {message}\nAssistant: {message}" format for EpisodeType.message
        episode_parts = []
        for role, content in messages_to_save:
            role_label = "User" if role == "user" else "Assistant"
            episode_parts.append(f"{role_label}: {content}")
        
        episode_body = "\n".join(episode_parts)
        
        if user_valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Adding conversation to Graphiti memory...", "done": False},
                }
            )
        
        # Generate group_id using configured method (email or user ID)
        group_id = self._get_group_id(__user__)
        saved_count = 0
        
        try:
            # Apply timeout if configured
            if self.valves.add_episode_timeout > 0:
                if group_id is not None:
                    await asyncio.wait_for(
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
                    await asyncio.wait_for(
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
                    await self.graphiti.add_episode(
                        name=f"Chat_Interaction_{chat_id}_{message_id}",
                        episode_body=episode_body,
                        source=EpisodeType.message,
                        source_description="Chat conversation",
                        reference_time=datetime.now(),
                        group_id=group_id,
                        update_communities=self.valves.update_communities,
                    )
                else:
                    await self.graphiti.add_episode(
                        name=f"Chat_Interaction_{chat_id}_{message_id}",
                        episode_body=episode_body,
                        source=EpisodeType.message,
                        source_description="Chat conversation",
                        reference_time=datetime.now(),
                        update_communities=self.valves.update_communities,
                    )
            print(f"Added conversation to Graphiti memory: {episode_body[:100]}...")
            saved_count = 1
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
                import traceback
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
                status_msg = "Failed to save conversation to Graphiti memory"
            else:
                status_msg = f"Added conversation to Graphiti memory"
            
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": status_msg, "done": True},
                }
            )
        
        return body
