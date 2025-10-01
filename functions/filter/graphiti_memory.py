"""
title: Graphiti Memory
author: Skyzi000
description: Automatically identify and store valuable information from chats as Memories.
author_email: dev@skyzi.jp
author_url: https://github.com/Skyzi000
repository_url: https://github.com/Skyzi000/open-webui-extensions
version: 0.1
requirements: graphiti-core[falkordb]

Design:
- Main class: Filter
- Related components:
  - Graphiti: Knowledge graph memory system
  - FalkorDriver: FalkorDB backend driver for graph storage
  - OpenAIGenericClient: LLM client for memory processing
  - OpenAIEmbedder: Embedding model for semantic search
  - OpenAIRerankerClient: Cross-encoder for result reranking

Architecture:
- Initialization: _initialize_graphiti() sets up the graph database connection
- Lazy initialization: _ensure_graphiti_initialized() provides automatic retry
- Memory search: inlet() retrieves relevant memories before chat processing
- Memory storage: outlet() stores new information after chat completion
"""

import ast
import json
import os
import time
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any

import aiohttp
from aiohttp import ClientError
from fastapi.requests import Request
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
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
        openai_api_url: str = Field(
            default="https://api.openai.com",
            description="openai compatible endpoint",
        )
        model: str = Field(
            default="gpt-4.1",
            description="Model to use for memory processing.",
        )
        small_model: str = Field(
            default="gpt-4.1-mini",
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

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        # openai_api_url: Optional[str] = Field(
        #     default=None,
        #     description="User-specific openai compatible endpoint (overrides global)",
        # )
        # model: Optional[str] = Field(
        #     default=None,
        #     description="User-specific model to use (overrides global). An intelligent model is highly recommended, as it will be able to better understand the context of the conversation.",
        # )
        # api_key: Optional[str] = Field(
        #     default=None, description="User-specific API key (overrides global)"
        # )
        # use_legacy_mode: bool = Field(
        #     default=False,
        #     description="Use legacy mode for memory processing. This means using legacy prompts, and only analyzing the last User message.",
        # )
        # messages_to_consider: int = Field(
        #     default=4,
        #     description="Number of messages to consider for memory processing, starting from the last message. Includes assistant responses.",
        # )

    def __init__(self):
        self.valves = self.Valves()
        self.graphiti = None
        self._initialize_graphiti()

    def _initialize_graphiti(self) -> bool:
        """
        Initialize Graphiti instance with configured backend.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'true' if self.valves.graphiti_telemetry_enabled else 'false'
            
            # Configure LLM client
            llm_config = LLMConfig(
                api_key=self.valves.api_key,
                model=self.valves.model,
                small_model=self.valves.small_model,
                base_url=self.valves.openai_api_url,
            )

            llm_client = OpenAIGenericClient(config=llm_config)

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
            
            print("Graphiti initialized successfully.")
            return True
            
        except Exception as e:
            print(f"Error initializing Graphiti: {e}")
            import traceback
            traceback.print_exc()
            self.graphiti = None
            return False
    
    def _ensure_graphiti_initialized(self) -> bool:
        """
        Ensure Graphiti is initialized, attempting re-initialization if necessary.
        
        Returns:
            bool: True if Graphiti is ready to use, False otherwise
        """
        if self.graphiti is not None:
            return True
        
        print("Graphiti not initialized. Attempting to initialize...")
        return self._initialize_graphiti()

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        print(f"inlet:{__name__}")
        print(f"inlet:user:{__user__}")
        
        # Check if graphiti is initialized, retry if not
        if not self._ensure_graphiti_initialized():
            print("Graphiti initialization failed. Skipping memory search.")
            if __user__ and __user__.get("valves", self.UserValves()).show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Memory service unavailable", "done": True},
                    }
                )
            return body
        
        if __user__ is None:
            print("User information is not available. Skipping memory search.")
            return body
            
        user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
        if user_valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Searching Graphiti: {body['messages'][-1]['content']}", "done": False},
                }
            )
        
        results = await self.graphiti.search(
            query=body["messages"][-1]["content"],
            group_ids=[f"{__user__['id']}_chat"],
        )
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
        for result in results:

            print(f'UUID: {result.uuid}')

            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')

            facts.append((result.fact, result.valid_at, result.invalid_at))

            print('---')
            
        if len(facts) > 0:
            body['messages'].append({
                "role": "system",
                "content": f"Relevant memories were found:\n" + "\n".join([f"- {fact} (Valid from: {valid_at}, Valid until: {invalid_at})" for fact, valid_at, invalid_at in facts])
            })
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Added {len(facts)} relevant memories to the conversation", "done": True},
                    }
                )
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        # Check if graphiti is initialized, retry if not
        if not self._ensure_graphiti_initialized():
            print("Graphiti initialization failed. Skipping memory addition.")
            return body
            
        if __user__ is None:
            print("User information is not available. Skipping memory addition.")
            return body
            
        user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())
        if len(body.get("messages", [])) == 0:
            return body
        if user_valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Adding Graphiti memory...", "done": False},
                }
            )
        await self.graphiti.add_episode(

            name=f"Chat_Message_{datetime.now().isoformat()}",

            episode_body=(
                body["messages"][-1]["content"]
            ),

            source=EpisodeType.message,

            source_description="Chat Message",

            reference_time=datetime.now(),
            group_id=f"{__user__['id']}_chat",
        )

        if user_valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Added Graphiti memory", "done": True},
                }
            )

        return body
