"""
title: Graphiti Memory Manage Tool
author: Skyzi000
description: Manage specific entities, relationships, or episodes in Graphiti knowledge graph memory.
author_email: dev@skyzi.jp
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-extensions
version: 0.1
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
import asyncio
from datetime import datetime
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
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge


class GraphitiHelper:
    def __init__(self, tools_instance):
        self.tools = tools_instance
        self.graphiti = None
        self._last_config = None
    
    @property
    def valves(self):
        """Always get fresh valves from Tools instance."""
        return self.tools.valves
    
    def get_config_hash(self) -> str:
        """Generate configuration hash for change detection."""
        import hashlib
        config_str = f"{self.valves.llm_client_type}|{self.valves.openai_api_url}|{self.valves.model}|{self.valves.embedding_model}|{self.valves.api_key}|{self.valves.graph_db_backend}|{self.valves.falkordb_host}|{self.valves.falkordb_port}"
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

        # Select LLM client based on configuration
        if self.valves.llm_client_type.lower() == "openai":
            llm_client = OpenAIClient(config=llm_config)
            if self.valves.debug_print:
                print("Using OpenAI client")
        elif self.valves.llm_client_type.lower() == "generic":
            llm_client = OpenAIGenericClient(config=llm_config)
            if self.valves.debug_print:
                print("Using OpenAI-compatible generic client")
        else:
            # Default to OpenAI client for unknown values
            llm_client = OpenAIClient(config=llm_config)
            if self.valves.debug_print:
                print(f"Unknown client type '{self.valves.llm_client_type}', defaulting to OpenAI client")

        # Initialize embedder
        embedder = OpenAIEmbedder(
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
                # OpenAIRerankerClient requires AsyncOpenAI client, not LLMClient wrapper
                # Both OpenAIClient and OpenAIGenericClient have .client attribute (AsyncOpenAI instance)
                cross_encoder=OpenAIRerankerClient(client=llm_client.client, config=llm_config),
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
                # OpenAIRerankerClient requires AsyncOpenAI client, not LLMClient wrapper
                # Both OpenAIClient and OpenAIGenericClient have .client attribute (AsyncOpenAI instance)
                cross_encoder=OpenAIRerankerClient(client=llm_client.client, config=llm_config),
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
                    import traceback
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
                import traceback
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
                import traceback
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
                    import traceback
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
    ) -> bool:
        """
        Show confirmation dialog and wait for user response.
        Helper method for Tools class - not exposed to AI.
        
        :param title: Dialog title
        :param items: List of items to display for confirmation
        :param warning_message: Warning message to show
        :param timeout: Timeout in seconds
        :param __user__: User information dictionary
        :param __event_call__: Event caller for confirmation dialog
        :return: True if confirmed, False if cancelled or timeout
        """
        import asyncio
        
        if not __event_call__:
            return True  # If no event_call, proceed without confirmation
        
        preview_text = "  \n".join(items)
        
        # Get user's language preference from UserValves
        user_valves = __user__.get("valves")
        if user_valves and hasattr(user_valves, 'message_language'):
            is_japanese = user_valves.message_language.lower() == 'ja'
        else:
            # Default to English if UserValves not set
            is_japanese = False
        
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
                return bool(result)
            except asyncio.TimeoutError:
                return False
        except Exception:
            return False


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
        
        semaphore_limit: int = Field(
            default=10,
            description="Maximum number of concurrent LLM operations",
        )
        
        group_id_format: str = Field(
            default="{user_id}",
            description="Format string for group_id. Available placeholders: {user_id}, {user_email}, {user_name}. Set to 'none' to disable group filtering.",
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
    
    async def search_entities(
        self,
        query: str,
        limit: int = 10,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for entities by name or description without deleting them.
        
        This tool allows you to preview entities before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_entities.
        
        :param query: Search query to find entities (e.g., "John Smith", "Python programming")
        :param limit: Maximum number of entities to return (default: 10, max: 100)
        :return: List of found entities with their details
        
        Note: __user__ and __event_emitter__ are automatically injected by the system.
        """
        import copy
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
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
                result += f"   UUID: `{uuid}`\n\n"
            
            result += f"💡 To delete these entities, use `search_and_delete_entities` with the same query and limit."
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching entities: {str(e)}"
            if self.valves.debug_print:
                import traceback
                traceback.print_exc()
            return error_msg
    
    async def search_facts(
        self,
        query: str,
        limit: int = 10,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for facts (relationships) without deleting them.
        
        This tool allows you to preview relationships before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_facts.
        
        :param query: Search query to find relationships (e.g., "works at", "friends with")
        :param limit: Maximum number of facts to return (default: 10, max: 100)
        :return: List of found facts with their details
        
        Note: __user__ and __event_emitter__ are automatically injected by the system.
        """
        import copy
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
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
                result += f"   UUID: `{uuid}`\n\n"
            
            result += f"💡 To delete these facts, use `search_and_delete_facts` with the same query and limit."
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching facts: {str(e)}"
            if self.valves.debug_print:
                import traceback
                traceback.print_exc()
            return error_msg
    
    async def search_episodes(
        self,
        query: str,
        limit: int = 10,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Search for episodes (conversation history) without deleting them.
        
        This tool allows you to preview episodes before deciding to delete them.
        Use this to verify what will be deleted before calling search_and_delete_episodes.
        
        :param query: Search query to find episodes (e.g., "conversation about Python")
        :param limit: Maximum number of episodes to return (default: 10, max: 100)
        :return: List of found episodes with their details
        
        Note: __user__ and __event_emitter__ are automatically injected by the system.
        """
        import copy
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
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
                result += f"   UUID: `{uuid}`\n\n"
            
            result += f"💡 To delete these episodes, use `search_and_delete_episodes` with the same query and limit."
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error searching episodes: {str(e)}"
            if self.valves.debug_print:
                import traceback
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
        import asyncio
        import copy
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
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
            user_valves = __user__.get("valves")
            is_japanese = user_valves and hasattr(user_valves, 'message_language') and user_valves.message_language.lower() == 'ja'
            
            # Show confirmation dialog
            confirmed = await self.helper.show_confirmation_dialog(
                title=f"エンティティの削除確認 ({total_count}件)" if is_japanese else f"Confirm Entity Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。関連する関係性も削除されます。" if is_japanese else "⚠️ This operation cannot be undone. Related relationships will also be deleted.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                if __event_call__:
                    return "🚫 User cancelled entity deletion"
            
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
                import traceback
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
        import asyncio
        import copy
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
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
            
            fact_list = []
            preview_items = []
            for i, edge in enumerate(edges, 1):
                fact_text = getattr(edge, 'fact', 'Unknown relationship')
                if len(fact_text) > 80:
                    fact_text = fact_text[:80] + "..."
                valid_at = getattr(edge, 'valid_at', 'unknown')
                invalid_at = getattr(edge, 'invalid_at', 'present')
                fact_list.append(f"{i}. {fact_text} ({valid_at} - {invalid_at})")
                preview_items.append(f"[{i}] {fact_text}  \n期間: {valid_at} - {invalid_at}")
            
            # Get user's language preference
            user_valves = __user__.get("valves")
            is_japanese = user_valves and hasattr(user_valves, 'message_language') and user_valves.message_language.lower() == 'ja'
            
            # Show confirmation dialog
            confirmed = await self.helper.show_confirmation_dialog(
                title=f"関係性の削除確認 ({total_count}件)" if is_japanese else f"Confirm Fact Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。" if is_japanese else "⚠️ This operation cannot be undone.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                if __event_call__:
                    return "🚫 User cancelled fact deletion"
            
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
                import traceback
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
        import asyncio
        import copy
        
        if not await self.helper.ensure_graphiti_initialized() or self.helper.graphiti is None:
            return "❌ Error: Memory service is not available"
        
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
            
            episode_list = []
            preview_items = []
            for i, episode in enumerate(episodes, 1):
                name = getattr(episode, 'name', 'Unknown episode')
                content = getattr(episode, 'content', '')
                if len(content) > 80:
                    content_preview = content[:80] + "..."
                else:
                    content_preview = content
                created_at = getattr(episode, 'created_at', 'unknown')
                episode_list.append(f"{i}. {name}: {content_preview} (created: {created_at})")
                preview_items.append(f"[{i}] {name}  \n{content_preview}  \n作成日時: {created_at}")
            
            # Get user's language preference
            user_valves = __user__.get("valves")
            is_japanese = user_valves and hasattr(user_valves, 'message_language') and user_valves.message_language.lower() == 'ja'
            
            # Show confirmation dialog
            confirmed = await self.helper.show_confirmation_dialog(
                title=f"エピソードの削除確認 ({total_count}件)" if is_japanese else f"Confirm Episode Deletion ({total_count} items)",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。関連するエンティティと関係性も削除されます。" if is_japanese else "⚠️ This operation cannot be undone. Related entities and relationships will also be deleted.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                if __event_call__:
                    return "🚫 User cancelled episode deletion"
            
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
                import traceback
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
        
        try:
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
                            preview_items.append(f"[Node {i}] {name}  \nUUID: {uuid}  \n概要: {summary}")
                        else:
                            preview_items.append(f"[Node {i}] ⚠️ Not found  \nUUID: {uuid}")
                    except Exception as e:
                        preview_items.append(f"[Node {i}] ⚠️ Error fetching details  \nUUID: {uuid}")
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
                            preview_items.append(f"[Edge {i}] {fact}  \nUUID: {uuid}  \n期間: {valid_at} → {invalid_at}")
                        else:
                            preview_items.append(f"[Edge {i}] ⚠️ Not found  \nUUID: {uuid}")
                    except Exception as e:
                        preview_items.append(f"[Edge {i}] ⚠️ Error fetching details  \nUUID: {uuid}")
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
                            preview_items.append(f"[Episode {i}] {name}  \nUUID: {uuid}  \n内容: {content}  \n作成: {created_at}")
                        else:
                            preview_items.append(f"[Episode {i}] ⚠️ Not found  \nUUID: {uuid}")
                    except Exception as e:
                        preview_items.append(f"[Episode {i}] ⚠️ Error fetching details  \nUUID: {uuid}")
                        if self.valves.debug_print:
                            print(f"Error fetching episode {uuid}: {e}")
            
            if not preview_items:
                return "ℹ️ No UUIDs provided for deletion"
            
            # Get user's language preference
            user_valves = __user__.get("valves")
            is_japanese = user_valves and hasattr(user_valves, 'message_language') and user_valves.message_language.lower() == 'ja'
            
            # Show confirmation dialog
            confirmed = await self.helper.show_confirmation_dialog(
                title="UUID指定削除の確認" if is_japanese else "Confirm UUID-based Deletion",
                items=preview_items,
                warning_message="⚠️ この操作は取り消すことができません。UUIDを直接指定しての削除は慎重に行ってください。" if is_japanese else "⚠️ This operation cannot be undone. Please be careful when deleting by UUID.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                if __event_call__:
                    return "🚫 User cancelled UUID-based deletion"
            
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
                import traceback
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
                return "ℹ️ メモリは既に空です"
            
            # Get user's language preference
            user_valves = __user__.get("valves")
            is_japanese = user_valves and hasattr(user_valves, 'message_language') and user_valves.message_language.lower() == 'ja'
            
            # Show confirmation dialog
            preview_items = [
                f"エンティティ(Entity): {node_count}個",
                f"関係性(Fact): {edge_count}個",
                f"エピソード(Episode): {episode_count}個",
            ]
            
            confirmed = await self.helper.show_confirmation_dialog(
                title="⚠️ 全メモリ削除の最終確認" if is_japanese else "⚠️ Final Confirmation: Clear All Memory",
                items=preview_items,
                warning_message="🔥 この操作は完全に元に戻せません！全てのメモリデータが永久に失われます。" if is_japanese else "🔥 This operation is completely irreversible! All memory data will be permanently lost.",
                timeout=self.valves.confirmation_timeout,
                __user__=__user__,
                __event_call__=__event_call__,
            )
            
            if not confirmed:
                if __event_call__:
                    return "🚫 User cancelled memory clearing"
            
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
                        return "🚫 確認文字列が一致しません。メモリ削除をキャンセルしました。" if is_japanese else "🚫 Confirmation text does not match. Memory clearing cancelled."
                except asyncio.TimeoutError:
                    return "🚫 入力タイムアウト。メモリ削除をキャンセルしました。" if is_japanese else "🚫 Input timeout. Memory clearing cancelled."
                except Exception as e:
                    if self.valves.debug_print:
                        print(f"Input confirmation error: {e}")
                    return "🚫 確認入力がキャンセルされました。" if is_japanese else "🚫 Input confirmation cancelled."
            
            # Use Node.delete_by_group_id() - the correct method for clearing all data
            try:
                if self.valves.debug_print:
                    print(f"Deleting all data for group_id: {group_id}")
                
                await EntityNode.delete_by_group_id(self.helper.graphiti.driver, group_id)
                
                result = f"🗑️ 全メモリを削除しました:\n"
                result += f"  - {node_count} エンティティ削除\n"
                result += f"  - {edge_count} 関係性削除\n"
                result += f"  - {episode_count} エピソード削除"
                
                return result
            except Exception as e:
                error_msg = f"❌ Error deleting memory: {str(e)}"
                if self.valves.debug_print:
                    import traceback
                    traceback.print_exc()
                return error_msg
            
        except Exception as e:
            error_msg = f"❌ Error clearing memory: {str(e)}"
            if self.valves.debug_print:
                import traceback
                traceback.print_exc()
            return error_msg
