# pyright: reportMissingImports=false
"""Test Open WebUI internal API compatibility.

These tests detect breaking changes in Open WebUI's internal API that would affect extensions.
When tests fail, it indicates Open WebUI has changed in a way that may break extensions.

Tested structures:
- __user__: UserModel.model_dump() structure (id, name, role, valves)
- body: Request body structure (messages, model, metadata)
- __metadata__: Metadata dict (chat_id, message_id, files)
- messages[]: ChatMessage structure (role, content)
- Internal ORM: Memories, Chats, Groups, Models

Usage by extensions:
- tools/memory.py: Memories, __user__["id"]
- functions/filter/user_info_injector.py: Groups, __user__["valves"]
- graphiti/: Chats, Models, __metadata__["chat_id"], message["role"]
"""

import ast
import inspect
from typing import get_origin, get_args

import pytest


def import_error_message(module_path: str) -> str:
    """Generate helpful error message for import failures."""
    return (
        f"Failed to import {module_path}. "
        "Open WebUI's internal API may have changed. "
        "Check if the module path has been renamed or moved in the latest Open WebUI version."
    )


def method_missing_message(cls_name: str, method: str) -> str:
    """Generate helpful error message for missing methods."""
    return (
        f"{cls_name}.{method} is missing. "
        "This method is required by extensions. "
        "Open WebUI may have renamed or removed this method."
    )


def signature_changed_message(cls_name: str, method: str, expected: str, actual: str) -> str:
    """Generate helpful error message for signature changes."""
    return (
        f"{cls_name}.{method} signature has changed. "
        f"Expected parameters: {expected}, Got: {actual}. "
        "Extensions may need to be updated."
    )


def field_type_message(cls_name: str, field: str, expected: str, actual: str) -> str:
    """Generate helpful error message for field type changes."""
    return (
        f"{cls_name}.{field} type has changed. "
        f"Expected: {expected}, Got: {actual}. "
        "Extensions may break if they assume the original type."
    )


# =============================================================================
# AST Analysis Helpers
# =============================================================================


class ValvesAssignmentVisitor(ast.NodeVisitor):
    """AST visitor to find __user__["valves"] = ... assignments."""

    def __init__(self):
        self.found_valves_injection = False

    def visit_Assign(self, node):
        """Check for assignment to __user__["valves"] or params["__user__"]["valves"]."""
        for target in node.targets:
            if self._is_valves_subscript(target):
                self.found_valves_injection = True
                return
        self.generic_visit(node)

    def _is_valves_subscript(self, node) -> bool:
        """Check if node is a subscript ending in ["valves"] within __user__ context."""
        if not isinstance(node, ast.Subscript):
            return False

        # Check for ["valves"] key
        if isinstance(node.slice, ast.Constant) and node.slice.value == "valves":
            # Check if parent contains __user__
            return self._contains_user_key(node.value)
        return False

    def _contains_user_key(self, node) -> bool:
        """Recursively check if node contains "__user__" key."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and node.slice.value == "__user__":
                return True
            return self._contains_user_key(node.value)
        if isinstance(node, ast.Name) and node.id == "__user__":
            return True
        return False


class MetadataDictVisitor(ast.NodeVisitor):
    """AST visitor to find metadata-specific dict construction with specific keys.

    Only detects keys in:
    1. Assignment to 'metadata' variable: metadata = {..., "key": value, ...}
    2. metadata.get("key") or metadata["key"] access patterns

    Does NOT match arbitrary dicts that happen to contain the same keys.
    """

    def __init__(self, required_keys: list[str]):
        self.required_keys = set(required_keys)
        self.found_keys = set()

    def visit_Assign(self, node):
        """Check for metadata = {...} assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "metadata":
                # Found: metadata = ...
                if isinstance(node.value, ast.Dict):
                    self._extract_keys_from_dict(node.value)
        self.generic_visit(node)

    def visit_Call(self, node):
        """Check for metadata.get("key") patterns."""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if self._is_metadata_name(node.func.value):
                # Found: metadata.get(...)
                if node.args and isinstance(node.args[0], ast.Constant):
                    key = node.args[0].value
                    if key in self.required_keys:
                        self.found_keys.add(key)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        """Check for metadata["key"] patterns."""
        if isinstance(node.slice, ast.Constant) and node.slice.value in self.required_keys:
            if self._is_metadata_name(node.value):
                self.found_keys.add(node.slice.value)
        self.generic_visit(node)

    def _extract_keys_from_dict(self, dict_node):
        """Extract required keys from a dict literal."""
        for key in dict_node.keys:
            if isinstance(key, ast.Constant) and key.value in self.required_keys:
                self.found_keys.add(key.value)

    def _is_metadata_name(self, node) -> bool:
        """Check if node is 'metadata' variable access."""
        if isinstance(node, ast.Name) and node.id == "metadata":
            return True
        return False


class ExtraParamsVisitor(ast.NodeVisitor):
    """AST visitor to find extra_params dict construction with __metadata__ key.

    Detects: extra_params = {..., "__metadata__": ..., ...}
    """

    def __init__(self):
        self.found_metadata_key = False

    def visit_Assign(self, node):
        """Check for extra_params = {...} assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "extra_params":
                if isinstance(node.value, ast.Dict):
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and key.value == "__metadata__":
                            self.found_metadata_key = True
                            return
        self.generic_visit(node)


def find_chat_message_class():
    """Find ChatMessage class from multiple possible locations.

    Returns the class if found, None otherwise.
    Open WebUI may move schema definitions during refactoring.
    """
    # List of possible import paths (add more as Open WebUI evolves)
    possible_paths = [
        ("open_webui.routers.ollama", "ChatMessage"),
        ("open_webui.routers.openai", "ChatMessage"),
        ("open_webui.schemas.chat", "ChatMessage"),
        ("open_webui.models.chat", "ChatMessage"),
    ]

    for module_path, class_name in possible_paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except ImportError:
            continue
    return None


def find_chat_completion_form():
    """Find GenerateChatCompletionForm from multiple possible locations."""
    possible_paths = [
        ("open_webui.routers.ollama", "GenerateChatCompletionForm"),
        ("open_webui.routers.openai", "GenerateChatCompletionForm"),
        ("open_webui.schemas.chat", "GenerateChatCompletionForm"),
        ("open_webui.schemas.request", "GenerateChatCompletionForm"),
    ]

    for module_path, class_name in possible_paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except ImportError:
            continue
    return None


def find_openai_chat_message_class():
    """Find OpenAIChatMessage class for multimodal content support.

    OpenAIChatMessage.content can be Union[Optional[str], list[...]]
    which is the multimodal format.
    """
    possible_paths = [
        ("open_webui.routers.ollama", "OpenAIChatMessage"),
        ("open_webui.routers.openai", "OpenAIChatMessage"),
        ("open_webui.schemas.chat", "OpenAIChatMessage"),
    ]

    for module_path, class_name in possible_paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except ImportError:
            continue
    return None


def find_openai_chat_completion_form():
    """Find OpenAIChatCompletionForm from multiple possible locations.

    This is the form used for OpenAI-compatible API requests.
    """
    possible_paths = [
        ("open_webui.routers.ollama", "OpenAIChatCompletionForm"),
        ("open_webui.routers.openai", "OpenAIChatCompletionForm"),
        ("open_webui.schemas.chat", "OpenAIChatCompletionForm"),
    ]

    for module_path, class_name in possible_paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except ImportError:
            continue
    return None


# =============================================================================
# UserModel Tests (__user__ structure)
# =============================================================================


class TestUserModel:
    """Test UserModel structure for __user__ parameter compatibility.

    __user__ is passed to tools/filters via UserModel.model_dump().
    Extensions rely on specific fields being present in __user__.

    Critical fields used by extensions:
    - __user__["id"]: str - Used for DB queries (Memories, Chats)
    - __user__["name"]: str - Used for display and episode labels
    - __user__["role"]: str - Used for authorization checks
    - __user__["valves"]: dict - Used with UserValves.model_validate()
    """

    @pytest.fixture
    def user_model_class(self):
        """Import UserModel class with helpful error on failure."""
        try:
            from open_webui.models.users import UserModel
            return UserModel
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.models.users.UserModel") + f"\n{e}")

    def test_user_model_dump_output(self, user_model_class):
        """Verify UserModel.model_dump() produces expected output structure.

        This is the critical integration test - even if type annotations are correct,
        we verify that actual serialization produces a dict with the expected fields.
        Extensions receive __user__ as UserModel.model_dump() output.
        """
        # Get required fields from model
        required_fields = list(user_model_class.model_fields.keys())

        # Build minimal test data with all required fields
        test_data = {}
        for field_name, field_info in user_model_class.model_fields.items():
            annotation = field_info.annotation
            origin = get_origin(annotation)
            # Handle Optional types
            if origin is not None:
                args = get_args(annotation)
                if type(None) in args:
                    # Optional field - use None or a test value
                    non_none_types = [a for a in args if a is not type(None)]
                    if non_none_types:
                        base_type = non_none_types[0]
                        if base_type is str:
                            test_data[field_name] = f"test_{field_name}"
                        elif base_type is int:
                            test_data[field_name] = 123
                        elif base_type is bool:
                            test_data[field_name] = True
                        else:
                            test_data[field_name] = None
                    else:
                        test_data[field_name] = None
            elif annotation is str:
                test_data[field_name] = f"test_{field_name}"
            elif annotation is int:
                test_data[field_name] = 123
            elif annotation is bool:
                test_data[field_name] = True
            else:
                # For complex types, try None if allowed or skip
                if field_info.default is not None:
                    continue  # Use default
                test_data[field_name] = None

        # Create instance
        try:
            user = user_model_class(**test_data)
        except Exception as e:
            pytest.fail(f"Cannot instantiate UserModel with test data: {e}")

        # Get model_dump output
        dumped = user.model_dump()

        # Verify critical fields are present in dump output
        critical_fields = ["id", "name", "role"]
        for field in critical_fields:
            assert field in dumped, \
                f"UserModel.model_dump() missing '{field}' field. " \
                f"Extensions expect __user__['{field}']."

        # Verify types in dump output
        assert isinstance(dumped.get("id"), str), \
            f"UserModel.model_dump()['id'] should be str. Got: {type(dumped.get('id'))}"
        assert isinstance(dumped.get("name"), str), \
            f"UserModel.model_dump()['name'] should be str. Got: {type(dumped.get('name'))}"
        assert isinstance(dumped.get("role"), str), \
            f"UserModel.model_dump()['role'] should be str. Got: {type(dumped.get('role'))}"

    def test_user_model_has_id_field(self, user_model_class):
        """Verify UserModel has 'id' field."""
        assert hasattr(user_model_class, "model_fields"), \
            "UserModel is not a Pydantic model (no model_fields)"
        assert "id" in user_model_class.model_fields, \
            "UserModel missing 'id' field. Extensions use __user__['id'] extensively."

    def test_user_model_id_is_str_type(self, user_model_class):
        """Verify UserModel.id is str type - critical for DB queries.

        Extensions pass __user__["id"] directly to Memories.get_memories_by_user_id().
        If id becomes int or None, all memory operations will fail.
        """
        field_info = user_model_class.model_fields.get("id")
        assert field_info is not None, "UserModel missing 'id' field"

        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            base_types = [arg for arg in args if arg is not type(None)]
            assert str in base_types or len(base_types) == 0, \
                field_type_message("UserModel", "id", "str", str(annotation))
        else:
            assert annotation is str, \
                field_type_message("UserModel", "id", "str", str(annotation))

    def test_user_model_has_name_field(self, user_model_class):
        """Verify UserModel has 'name' field."""
        assert "name" in user_model_class.model_fields, \
            "UserModel missing 'name' field. Extensions use __user__['name'] for user display."

    def test_user_model_name_is_str_type(self, user_model_class):
        """Verify UserModel.name is str type."""
        field_info = user_model_class.model_fields.get("name")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            base_types = [arg for arg in args if arg is not type(None)]
            assert str in base_types or len(base_types) == 0, \
                field_type_message("UserModel", "name", "str", str(annotation))
        else:
            assert annotation is str, \
                field_type_message("UserModel", "name", "str", str(annotation))

    def test_user_model_has_role_field(self, user_model_class):
        """Verify UserModel has 'role' field."""
        assert "role" in user_model_class.model_fields, \
            "UserModel missing 'role' field. Extensions use __user__['role'] for authorization."

    def test_user_model_role_is_str_type(self, user_model_class):
        """Verify UserModel.role is str type."""
        field_info = user_model_class.model_fields.get("role")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            base_types = [arg for arg in args if arg is not type(None)]
            assert str in base_types or len(base_types) == 0, \
                field_type_message("UserModel", "role", "str", str(annotation))
        else:
            assert annotation is str, \
                field_type_message("UserModel", "role", "str", str(annotation))

    def test_user_model_has_email_field(self, user_model_class):
        """Verify UserModel has 'email' field."""
        assert "email" in user_model_class.model_fields, \
            "UserModel missing 'email' field."


class TestUserValvesInjection:
    """Test that __user__["valves"] injection exists in Open WebUI.

    Extensions use: self.UserValves.model_validate((__user__ or {}).get("valves", {}))
    This requires Open WebUI to inject valves into __user__ dict.

    Uses AST analysis to detect the injection pattern, avoiding false positives
    from minor code refactoring like quote style changes.
    """

    def test_user_valves_injection_in_tools(self):
        """Verify Open WebUI's tool module injects valves into __user__.

        Uses AST to find assignment pattern: __user__["valves"] = ...
        """
        try:
            from open_webui.utils import tools
            source = inspect.getsource(tools)
            tree = ast.parse(source)

            visitor = ValvesAssignmentVisitor()
            visitor.visit(tree)

            assert visitor.found_valves_injection, \
                "Open WebUI tools.py no longer injects __user__['valves']. " \
                "Extensions using UserValves will break. " \
                "Expected AST pattern: __user__[\"valves\"] = ..."
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.utils.tools") + f"\n{e}")

    def test_user_valves_injection_in_filters(self):
        """Verify Open WebUI's filter module injects valves into __user__.

        Uses AST to find assignment pattern: params["__user__"]["valves"] = ...
        """
        try:
            from open_webui.utils import filter as filter_utils
            source = inspect.getsource(filter_utils)
            tree = ast.parse(source)

            visitor = ValvesAssignmentVisitor()
            visitor.visit(tree)

            assert visitor.found_valves_injection, \
                "Open WebUI filter.py no longer injects __user__['valves']. " \
                "Extensions using UserValves will break. " \
                "Expected AST pattern: ...['__user__']['valves'] = ..."
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.utils.filter") + f"\n{e}")


# =============================================================================
# Message Structure Tests
# =============================================================================


class TestChatMessageStructure:
    """Test ChatMessage structure for message compatibility.

    Extensions access messages like:
    - message["role"] (str) - always present
    - message["content"] (str or None) - for text content

    This test searches multiple possible locations for the ChatMessage schema
    to avoid false positives when Open WebUI refactors schema locations.
    """

    @pytest.fixture
    def chat_message_class(self):
        """Find ChatMessage class from available locations."""
        cls = find_chat_message_class()
        if cls is None:
            pytest.fail(
                "Cannot find ChatMessage class in any known location. "
                "Open WebUI may have significantly restructured. "
                "Checked: open_webui.routers.ollama, open_webui.routers.openai, "
                "open_webui.schemas.chat, open_webui.models.chat"
            )
        return cls

    def test_chat_message_has_role_field(self, chat_message_class):
        """Verify ChatMessage has 'role' field - always required."""
        assert "role" in chat_message_class.model_fields, \
            "ChatMessage missing 'role' field. All extensions expect message['role']."

    def test_chat_message_role_is_str(self, chat_message_class):
        """Verify ChatMessage.role is str type."""
        field_info = chat_message_class.model_fields.get("role")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            base_types = [arg for arg in args if arg is not type(None)]
            assert str in base_types, \
                field_type_message("ChatMessage", "role", "str", str(annotation))
        else:
            assert annotation is str, \
                field_type_message("ChatMessage", "role", "str", str(annotation))

    def test_chat_message_has_content_field(self, chat_message_class):
        """Verify ChatMessage has 'content' field."""
        assert "content" in chat_message_class.model_fields, \
            "ChatMessage missing 'content' field. Extensions expect message['content']."

    def test_chat_message_content_allows_str(self, chat_message_class):
        """Verify ChatMessage.content accepts str type."""
        field_info = chat_message_class.model_fields.get("content")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            base_types = [arg for arg in args if arg is not type(None)]
            assert str in base_types or len(base_types) == 0, \
                f"ChatMessage.content should accept str. Got: {annotation}"
        else:
            assert annotation is str, \
                f"ChatMessage.content should be str. Got: {annotation}"

    def test_chat_message_can_be_instantiated_with_str_content(self, chat_message_class):
        """Verify ChatMessage can be created with string content."""
        try:
            msg = chat_message_class(role="user", content="test message")
            assert msg.role == "user"
            assert msg.content == "test message"
        except Exception as e:
            pytest.fail(f"Cannot create ChatMessage with string content: {e}")

    def test_chat_message_content_is_optional(self, chat_message_class):
        """Verify ChatMessage.content field is Optional (can be None)."""
        field_info = chat_message_class.model_fields.get("content")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            assert type(None) in args, \
                "ChatMessage.content should be Optional (allow None for tool_calls messages)"


# =============================================================================
# Body Structure Tests
# =============================================================================


class TestBodyStructure:
    """Test expected body structure for filter inlet/outlet.

    Filters receive 'body' parameter which should contain:
    - messages: list - required
    - model: str - required

    This test searches multiple possible locations for the schema to avoid
    false positives when Open WebUI refactors schema locations.
    """

    @pytest.fixture
    def chat_completion_form(self):
        """Find GenerateChatCompletionForm from available locations."""
        cls = find_chat_completion_form()
        if cls is None:
            pytest.fail(
                "Cannot find GenerateChatCompletionForm class in any known location. "
                "Open WebUI may have significantly restructured. "
                "Checked: open_webui.routers.ollama, open_webui.routers.openai, "
                "open_webui.schemas.chat, open_webui.schemas.request"
            )
        return cls

    def test_body_has_messages_field(self, chat_completion_form):
        """Verify body has 'messages' field."""
        assert "messages" in chat_completion_form.model_fields, \
            "Body missing 'messages' field. Filters expect body['messages']."

    def test_body_messages_is_list(self, chat_completion_form):
        """Verify body.messages is a list type."""
        field_info = chat_completion_form.model_fields.get("messages")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        assert origin is list, \
            f"body['messages'] should be list type. Got: {annotation}"

    def test_body_has_model_field(self, chat_completion_form):
        """Verify body has 'model' field."""
        assert "model" in chat_completion_form.model_fields, \
            "Body missing 'model' field. Filters may access body['model']."

    def test_body_model_is_str(self, chat_completion_form):
        """Verify body.model is str type."""
        field_info = chat_completion_form.model_fields.get("model")
        assert field_info is not None
        annotation = field_info.annotation
        assert annotation is str, \
            field_type_message("GenerateChatCompletionForm", "model", "str", str(annotation))


class TestMetadataStructure:
    """Test __metadata__ structure passed to filters/tools.

    __metadata__ contains runtime context:
    - chat_id: str - Used by Graphiti for conversation tracking
    - message_id: str - Used by Graphiti for message tracking
    - files: list - Used by full_context_mode_toggle

    Uses AST analysis to verify these keys are actually used in metadata dict
    construction, avoiding false negatives from simple string matching.
    """

    def test_metadata_has_chat_id_key(self):
        """Verify middleware uses chat_id in metadata construction.

        Uses AST to find 'chat_id' in dict literals or metadata subscript access.
        """
        try:
            from open_webui.utils import middleware
            source = inspect.getsource(middleware)
            tree = ast.parse(source)

            visitor = MetadataDictVisitor(["chat_id"])
            visitor.visit(tree)

            assert "chat_id" in visitor.found_keys, \
                "middleware.py doesn't use 'chat_id' in metadata dict. " \
                "Extensions using __metadata__['chat_id'] will break."
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.utils.middleware") + f"\n{e}")

    def test_metadata_has_message_id_key(self):
        """Verify middleware uses message_id in metadata construction."""
        try:
            from open_webui.utils import middleware
            source = inspect.getsource(middleware)
            tree = ast.parse(source)

            visitor = MetadataDictVisitor(["message_id"])
            visitor.visit(tree)

            assert "message_id" in visitor.found_keys, \
                "middleware.py doesn't use 'message_id' in metadata dict. " \
                "Extensions using __metadata__['message_id'] will break."
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.utils.middleware") + f"\n{e}")

    def test_metadata_has_files_key(self):
        """Verify middleware uses files key in metadata construction.

        full_context_mode_toggle uses body.get("metadata", {}).get("files", [])
        """
        try:
            from open_webui.utils import middleware
            source = inspect.getsource(middleware)
            tree = ast.parse(source)

            visitor = MetadataDictVisitor(["files"])
            visitor.visit(tree)

            assert "files" in visitor.found_keys, \
                "middleware.py doesn't use 'files' in metadata dict. " \
                "Extensions using body['metadata']['files'] will break."
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.utils.middleware") + f"\n{e}")

    def test_extra_params_has_metadata_key(self):
        """Verify extra_params dict includes __metadata__ key.

        Filters receive __metadata__ via extra_params["__metadata__"].
        """
        try:
            from open_webui.utils import middleware
            source = inspect.getsource(middleware)
            tree = ast.parse(source)

            visitor = ExtraParamsVisitor()
            visitor.visit(tree)

            assert visitor.found_metadata_key, \
                "middleware.py doesn't pass __metadata__ in extra_params. " \
                "Filters won't receive metadata."
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.utils.middleware") + f"\n{e}")


# =============================================================================
# Internal ORM Tests
# =============================================================================


class TestMemoriesModule:
    """Test open_webui.models.memories.Memories compatibility.

    Used by: tools/memory.py, graphiti/tools/graphiti_memory_manage.py
    """

    @pytest.fixture
    def memories_class(self):
        """Import Memories class."""
        try:
            from open_webui.models.memories import Memories
            return Memories
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.models.memories.Memories") + f"\n{e}")

    def test_get_memories_by_user_id_exists(self, memories_class):
        """Verify get_memories_by_user_id method exists."""
        assert hasattr(memories_class, "get_memories_by_user_id"), \
            method_missing_message("Memories", "get_memories_by_user_id")

    def test_get_memories_by_user_id_signature(self, memories_class):
        """Verify get_memories_by_user_id has user_id parameter."""
        method = getattr(memories_class, "get_memories_by_user_id")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "user_id" in params, \
            signature_changed_message("Memories", "get_memories_by_user_id", "user_id", str(params))

    def test_get_memories_by_user_id_is_sync(self, memories_class):
        """Verify get_memories_by_user_id is synchronous."""
        method = getattr(memories_class, "get_memories_by_user_id")
        assert not inspect.iscoroutinefunction(method), \
            "Memories.get_memories_by_user_id should be synchronous, but it's async"

    def test_insert_new_memory_exists(self, memories_class):
        """Verify insert_new_memory method exists."""
        assert hasattr(memories_class, "insert_new_memory"), \
            method_missing_message("Memories", "insert_new_memory")

    def test_insert_new_memory_signature(self, memories_class):
        """Verify insert_new_memory has user_id and content parameters."""
        method = getattr(memories_class, "insert_new_memory")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "user_id" in params, \
            signature_changed_message("Memories", "insert_new_memory", "user_id, content", str(params))
        assert "content" in params, \
            signature_changed_message("Memories", "insert_new_memory", "user_id, content", str(params))

    def test_delete_memory_by_id_and_user_id_exists(self, memories_class):
        """Verify delete_memory_by_id_and_user_id method exists."""
        assert hasattr(memories_class, "delete_memory_by_id_and_user_id"), \
            method_missing_message("Memories", "delete_memory_by_id_and_user_id")

    def test_delete_memory_by_id_and_user_id_signature(self, memories_class):
        """Verify delete_memory_by_id_and_user_id has required parameters."""
        method = getattr(memories_class, "delete_memory_by_id_and_user_id")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "id" in params, \
            signature_changed_message("Memories", "delete_memory_by_id_and_user_id", "id, user_id", str(params))
        assert "user_id" in params, \
            signature_changed_message("Memories", "delete_memory_by_id_and_user_id", "id, user_id", str(params))

    def test_update_memory_by_id_and_user_id_exists(self, memories_class):
        """Verify update_memory_by_id_and_user_id method exists."""
        assert hasattr(memories_class, "update_memory_by_id_and_user_id"), \
            method_missing_message("Memories", "update_memory_by_id_and_user_id")

    def test_update_memory_by_id_and_user_id_signature(self, memories_class):
        """Verify update_memory_by_id_and_user_id has required parameters."""
        method = getattr(memories_class, "update_memory_by_id_and_user_id")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "id" in params, \
            signature_changed_message("Memories", "update_memory_by_id_and_user_id", "id, user_id, content", str(params))
        assert "user_id" in params, \
            signature_changed_message("Memories", "update_memory_by_id_and_user_id", "id, user_id, content", str(params))
        assert "content" in params, \
            signature_changed_message("Memories", "update_memory_by_id_and_user_id", "id, user_id, content", str(params))


class TestChatsModule:
    """Test open_webui.models.chats.Chats compatibility.

    Used by: graphiti/functions/filter/graphiti_memory.py,
             graphiti/functions/action/add_graphiti_memory_action.py
    """

    @pytest.fixture
    def chats_class(self):
        """Import Chats class."""
        try:
            from open_webui.models.chats import Chats
            return Chats
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.models.chats.Chats") + f"\n{e}")

    def test_get_chat_by_id_exists(self, chats_class):
        """Verify get_chat_by_id method exists."""
        assert hasattr(chats_class, "get_chat_by_id"), \
            method_missing_message("Chats", "get_chat_by_id")

    def test_get_chat_by_id_signature(self, chats_class):
        """Verify get_chat_by_id has id parameter."""
        method = getattr(chats_class, "get_chat_by_id")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "id" in params, \
            signature_changed_message("Chats", "get_chat_by_id", "id", str(params))

    def test_get_chat_title_by_id_exists(self, chats_class):
        """Verify get_chat_title_by_id method exists.

        Used by: graphiti/functions/filter/graphiti_memory.py,
                 graphiti/functions/action/add_graphiti_memory_action.py
        """
        assert hasattr(chats_class, "get_chat_title_by_id"), \
            method_missing_message("Chats", "get_chat_title_by_id")

    def test_chat_model_has_id_field(self, chats_class):
        """Verify ChatModel has 'id' field."""
        try:
            from open_webui.models.chats import ChatModel
            assert hasattr(ChatModel, "model_fields"), "ChatModel is not a Pydantic model"
            assert "id" in ChatModel.model_fields, "ChatModel missing 'id' field"
        except ImportError:
            pytest.skip("ChatModel not available")

    def test_chat_model_has_chat_field(self, chats_class):
        """Verify ChatModel has 'chat' or 'messages' field for conversation data."""
        try:
            from open_webui.models.chats import ChatModel
            fields = ChatModel.model_fields
            assert "chat" in fields or "messages" in fields, \
                "ChatModel missing 'chat' or 'messages' field"
        except ImportError:
            pytest.skip("ChatModel not available")


class TestGroupsModule:
    """Test open_webui.models.groups.Groups compatibility.

    Used by: functions/filter/user_info_injector.py
    """

    @pytest.fixture
    def groups_class(self):
        """Import Groups class."""
        try:
            from open_webui.models.groups import Groups
            return Groups
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.models.groups.Groups") + f"\n{e}")

    def test_get_groups_by_member_id_exists(self, groups_class):
        """Verify get_groups_by_member_id method exists."""
        assert hasattr(groups_class, "get_groups_by_member_id"), \
            method_missing_message("Groups", "get_groups_by_member_id")

    def test_get_groups_by_member_id_signature(self, groups_class):
        """Verify get_groups_by_member_id has id parameter."""
        method = getattr(groups_class, "get_groups_by_member_id")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        has_id_param = "member_id" in params or "user_id" in params or "id" in params
        assert has_id_param, \
            signature_changed_message("Groups", "get_groups_by_member_id", "member_id/user_id", str(params))

    def test_group_model_has_name_field(self, groups_class):
        """Verify GroupModel has 'name' field."""
        try:
            from open_webui.models.groups import GroupModel
            if hasattr(GroupModel, "model_fields"):
                assert "name" in GroupModel.model_fields, "GroupModel missing 'name' field"
        except ImportError:
            pytest.skip("GroupModel not available")


class TestModelsModule:
    """Test open_webui.models.models.Models compatibility.

    Used by: graphiti/functions/filter/graphiti_memory.py
    """

    @pytest.fixture
    def models_class(self):
        """Import Models class."""
        try:
            from open_webui.models.models import Models
            return Models
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.models.models.Models") + f"\n{e}")

    def test_get_model_by_id_exists(self, models_class):
        """Verify get_model_by_id method exists.

        Used by: graphiti/functions/filter/graphiti_memory.py
        """
        assert hasattr(models_class, "get_model_by_id"), \
            method_missing_message("Models", "get_model_by_id")
        assert callable(getattr(models_class, "get_model_by_id")), \
            "Models.get_model_by_id should be callable"

    def test_get_model_by_id_signature(self, models_class):
        """Verify get_model_by_id has id parameter."""
        method = getattr(models_class, "get_model_by_id")
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "id" in params, \
            signature_changed_message("Models", "get_model_by_id", "id", str(params))


# =============================================================================
# OpenAI Router Form Tests
# =============================================================================


class TestOpenAIChatMessageStructure:
    """Test OpenAIChatMessage structure for multimodal content support.

    OpenAIChatMessage.content can be:
    - str - simple text message
    - list[OpenAIChatMessageContent] - multimodal format with type/text/image_url

    Extensions handling multimodal input must check for both formats.
    Graphiti also uses message["id"] for episode tracking.
    """

    @pytest.fixture
    def openai_chat_message_class(self):
        """Find OpenAIChatMessage class from available locations."""
        cls = find_openai_chat_message_class()
        if cls is None:
            pytest.fail(
                "Cannot find OpenAIChatMessage class in any known location. "
                "Open WebUI may have significantly restructured. "
                "Checked: open_webui.routers.ollama, open_webui.routers.openai, "
                "open_webui.schemas.chat"
            )
        return cls

    def test_openai_chat_message_has_role_field(self, openai_chat_message_class):
        """Verify OpenAIChatMessage has 'role' field."""
        assert "role" in openai_chat_message_class.model_fields, \
            "OpenAIChatMessage missing 'role' field."

    def test_openai_chat_message_has_content_field(self, openai_chat_message_class):
        """Verify OpenAIChatMessage has 'content' field."""
        assert "content" in openai_chat_message_class.model_fields, \
            "OpenAIChatMessage missing 'content' field."

    def test_openai_chat_message_content_accepts_str(self, openai_chat_message_class):
        """Verify OpenAIChatMessage.content accepts str type."""
        field_info = openai_chat_message_class.model_fields.get("content")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            # Flatten Union types
            base_types = []
            for arg in args:
                if arg is type(None):
                    continue
                inner_origin = get_origin(arg)
                if inner_origin is not None:
                    base_types.append(inner_origin)
                else:
                    base_types.append(arg)
            assert str in base_types or any(t is str for t in base_types), \
                f"OpenAIChatMessage.content should accept str. Got: {annotation}"
        else:
            assert annotation is str, \
                f"OpenAIChatMessage.content should accept str. Got: {annotation}"

    def test_openai_chat_message_content_accepts_list(self, openai_chat_message_class):
        """Verify OpenAIChatMessage.content accepts list type (multimodal format).

        Multimodal content format: list[{type: "text"|"image_url", text?: str, image_url?: {...}}]
        If this fails, multimodal (image/file) support has been removed.
        """
        field_info = openai_chat_message_class.model_fields.get("content")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)

        assert origin is not None, \
            f"OpenAIChatMessage.content should be Union type for multimodal support. Got: {annotation}"

        args = get_args(annotation)
        has_list = False
        for arg in args:
            if arg is type(None):
                continue
            inner_origin = get_origin(arg)
            if inner_origin is list or arg is list:
                has_list = True
                break

        assert has_list, \
            f"OpenAIChatMessage.content should accept list for multimodal. Got: {annotation}"

    def test_openai_chat_message_can_instantiate_with_str_content(self, openai_chat_message_class):
        """Verify OpenAIChatMessage can be created with string content."""
        try:
            msg = openai_chat_message_class(role="user", content="test message")
            assert msg.role == "user"
            assert msg.content == "test message"
        except Exception as e:
            pytest.fail(f"Cannot create OpenAIChatMessage with string content: {e}")

    def test_openai_chat_message_can_instantiate_with_list_content(self, openai_chat_message_class):
        """Verify OpenAIChatMessage can be created with list content (multimodal).

        Multimodal support is required for image/file handling.
        If this fails, extensions handling multimodal content will break.
        """
        try:
            multimodal_content = [{"type": "text", "text": "Hello"}]
            msg = openai_chat_message_class(role="user", content=multimodal_content)
            assert msg.role == "user"
            assert isinstance(msg.content, list)
        except Exception as e:
            pytest.fail(
                f"OpenAIChatMessage cannot accept list content (multimodal). "
                f"Extensions handling images/files will break. Error: {e}"
            )


class TestOpenAIChatCompletionForm:
    """Test OpenAIChatCompletionForm structure for OpenAI-compatible API.

    This form is used when routing requests to OpenAI-compatible backends.
    Extensions may receive body in this format depending on the routing.

    Required fields:
    - messages: list - required
    - model: str - required
    """

    @pytest.fixture
    def openai_chat_completion_form(self):
        """Find OpenAIChatCompletionForm from available locations."""
        cls = find_openai_chat_completion_form()
        if cls is None:
            pytest.fail(
                "Cannot find OpenAIChatCompletionForm class in any known location. "
                "Open WebUI may have significantly restructured. "
                "Checked: open_webui.routers.ollama, open_webui.routers.openai, "
                "open_webui.schemas.chat"
            )
        return cls

    def test_form_has_messages_field(self, openai_chat_completion_form):
        """Verify OpenAIChatCompletionForm has 'messages' field."""
        assert "messages" in openai_chat_completion_form.model_fields, \
            "OpenAIChatCompletionForm missing 'messages' field."

    def test_form_messages_is_list(self, openai_chat_completion_form):
        """Verify OpenAIChatCompletionForm.messages is a list type."""
        field_info = openai_chat_completion_form.model_fields.get("messages")
        assert field_info is not None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        assert origin is list, \
            f"OpenAIChatCompletionForm['messages'] should be list. Got: {annotation}"

    def test_form_has_model_field(self, openai_chat_completion_form):
        """Verify OpenAIChatCompletionForm has 'model' field."""
        assert "model" in openai_chat_completion_form.model_fields, \
            "OpenAIChatCompletionForm missing 'model' field."

    def test_form_model_is_str(self, openai_chat_completion_form):
        """Verify OpenAIChatCompletionForm.model is str type."""
        field_info = openai_chat_completion_form.model_fields.get("model")
        assert field_info is not None
        annotation = field_info.annotation
        assert annotation is str, \
            field_type_message("OpenAIChatCompletionForm", "model", "str", str(annotation))


# =============================================================================
# Message ID Tests (Graphiti)
# =============================================================================


class TestMessageIdSupport:
    """Test that Open WebUI supports message ID tracking.

    Graphiti uses message["id"] for tracking individual messages in episodes.
    Message IDs come from body["messages"] dict (not Pydantic schema) and
    are stored/retrieved via the Chats module.

    Note: OpenAIChatMessage/ChatMessage Pydantic schemas don't have an 'id' field.
    The 'id' is passed through the raw body dict and stored in database.
    """

    def test_chats_has_get_message_by_id_and_message_id(self):
        """Verify Chats has get_message_by_id_and_message_id method.

        Graphiti calls Chats.get_message_by_id_and_message_id(chat_id, msg_id)
        to get message data from database. This exact method is required.
        """
        try:
            from open_webui.models.chats import Chats
            assert hasattr(Chats, "get_message_by_id_and_message_id"), \
                method_missing_message("Chats", "get_message_by_id_and_message_id")
            assert callable(getattr(Chats, "get_message_by_id_and_message_id")), \
                "Chats.get_message_by_id_and_message_id should be callable"
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.models.chats.Chats") + f"\n{e}")

    def test_chats_get_message_by_id_and_message_id_signature(self):
        """Verify get_message_by_id_and_message_id has required parameters."""
        try:
            from open_webui.models.chats import Chats
            method = getattr(Chats, "get_message_by_id_and_message_id")
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            assert "id" in params, \
                signature_changed_message("Chats", "get_message_by_id_and_message_id", "id, message_id", str(params))
            assert "message_id" in params, \
                signature_changed_message("Chats", "get_message_by_id_and_message_id", "id, message_id", str(params))
        except ImportError as e:
            pytest.fail(import_error_message("open_webui.models.chats.Chats") + f"\n{e}")

    def test_chat_model_stores_messages(self):
        """Verify ChatModel can store message data (including ids)."""
        try:
            from open_webui.models.chats import ChatModel
            fields = ChatModel.model_fields
            assert "chat" in fields or "messages" in fields, \
                "ChatModel missing field for storing messages with IDs"
        except ImportError:
            pytest.skip("ChatModel not available")

    def test_openai_chat_message_allows_extra_fields(self):
        """Verify OpenAIChatMessage preserves extra fields like 'id'.

        Graphiti relies on message["id"] being preserved through the system.
        OpenAIChatMessage must have extra="allow" in model_config to preserve
        fields not explicitly defined in the schema.

        If this fails, message IDs will be silently dropped during Pydantic
        validation, breaking Graphiti's episode tracking.
        """
        # Find OpenAIChatMessage class
        cls = None
        for module_path in [
            "open_webui.routers.ollama",
            "open_webui.routers.openai",
            "open_webui.schemas.chat",
        ]:
            try:
                module = __import__(module_path, fromlist=["OpenAIChatMessage"])
                if hasattr(module, "OpenAIChatMessage"):
                    cls = module.OpenAIChatMessage
                    break
            except ImportError:
                continue

        assert cls is not None, \
            "Cannot find OpenAIChatMessage class to verify extra field handling"

        # Check model_config for extra="allow"
        config = getattr(cls, "model_config", {})
        extra_setting = config.get("extra", None)

        assert extra_setting == "allow", \
            f"OpenAIChatMessage.model_config['extra'] must be 'allow' to preserve message IDs. " \
            f"Got: {extra_setting!r}. " \
            "This is a BREAKING CHANGE - message IDs will be dropped."

    def test_openai_chat_message_preserves_id_field(self):
        """Verify OpenAIChatMessage actually preserves 'id' in model_dump().

        This is the critical integration test - even if extra="allow" is set,
        we verify that passing an 'id' field results in it being preserved
        when serializing the model.
        """
        # Find OpenAIChatMessage class
        cls = None
        for module_path in [
            "open_webui.routers.ollama",
            "open_webui.routers.openai",
            "open_webui.schemas.chat",
        ]:
            try:
                module = __import__(module_path, fromlist=["OpenAIChatMessage"])
                if hasattr(module, "OpenAIChatMessage"):
                    cls = module.OpenAIChatMessage
                    break
            except ImportError:
                continue

        assert cls is not None, \
            "Cannot find OpenAIChatMessage class to verify id preservation"

        # Create message with id
        test_id = "test-message-id-12345"
        msg = cls(role="user", content="test content", id=test_id)

        # Verify id is accessible as attribute
        assert hasattr(msg, "id"), \
            "OpenAIChatMessage does not preserve 'id' as attribute. " \
            "Message IDs will be lost - BREAKING CHANGE for Graphiti."

        assert getattr(msg, "id") == test_id, \
            f"OpenAIChatMessage.id mismatch. Expected: {test_id}, Got: {getattr(msg, 'id')}"

        # Verify id is included in serialization
        dumped = msg.model_dump()
        assert "id" in dumped, \
            "OpenAIChatMessage.model_dump() does not include 'id'. " \
            "Message IDs will be lost during serialization - BREAKING CHANGE for Graphiti."

        assert dumped["id"] == test_id, \
            f"Serialized 'id' mismatch. Expected: {test_id}, Got: {dumped['id']}"
