"""A model backend wrapping the Ollama Python SDK."""

import asyncio
import datetime
import functools
import json as _json
from collections.abc import AsyncIterator, Coroutine, Sequence
from typing import Any

import httpx
import ollama
from tqdm import tqdm

from ..backends import ModelIdentifier, model_ids
from ..core import (
    BaseModelSubclass,
    C,
    CBlock,
    Component,
    Context,
    GenerateLog,
    GenerateType,
    ImageUrlBlock,
    MelleaLogger,
    ModelOutputThunk,
    ModelToolCall,
    RawProviderResponse,
)
from ..core.base import AbstractMelleaTool
from ..formatters import ChatFormatter, TemplateFormatter, granite as granite_formatters
from ..formatters.granite.base.types import (
    AssistantMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from ..helpers import (
    DEFAULT_CHUNK_TIMEOUT,
    ClientCache,
    get_current_event_loop,
    message_to_openai_message,
    messages_to_docs,
    send_to_queue,
)
from ..stdlib.components import Intrinsic, Message
from ..stdlib.requirements import ALoraRequirement
from ..telemetry.context import generate_request_id, with_context
from .adapters import AdapterMixin, AdapterType, IntrinsicAdapter
from .backend import FormatterBackend
from .model_options import ModelOption
from .tools import (
    add_tools_from_context_actions,
    add_tools_from_model_options,
    convert_tools_to_json,
)

format: None = None  # typing this variable in order to shadow the global format function and ensure mypy checks for errors

# Maps the mellea intrinsic catalog name to the conventional Ollama model
# name prefix used for the merged aLoRA/LoRA models distributed by IBM.
# Tag suffix (e.g. ":8b") is derived automatically from the base model at runtime.
# Users can override the full tag via add_adapter(..., ollama_model="...") when
# their local Ollama naming differs from this convention.
# TODO: add remaining intrinsics once their Ollama model names are confirmed
_INTRINSIC_OLLAMA_PREFIX: dict[str, str] = {
    "uncertainty": "granite-uncertainty",
    "requirement-check": "granite-requirement-check",
    "context-attribution": "granite-context-attribution",
}


def _apply_transforms_no_logprobs(raw_json: str, transforms: list[dict]) -> str:
    """Apply io.yaml output transformations to a raw JSON string without logprobs.

    The ``likelihood`` transformation normally requires token logprobs to compute
    a weighted-average float score.  Ollama does not expose logprobs for its
    merged intrinsic models, so this function substitutes a direct lookup from
    the rule's ``categories_to_values`` mapping, which is sufficient for all
    current Granite intrinsics (categorical ``"yes"``/``"no"`` or ``"0"``-``"9"``
    outputs that map to fixed floats).

    Structural transforms (``nest``, ``project``) are applied by manipulating the
    parsed dict directly.  Unknown transform types are skipped with a warning.

    Args:
        raw_json: Raw JSON string from the Ollama model (e.g. ``'{"score":"yes"}'``).
        transforms: List of transformation dicts from the io.yaml ``transformations``
            block.

    Returns:
        Transformed JSON string ready to be consumed by ``output_to_bool``.
    """
    if not raw_json.strip():
        return raw_json

    try:
        data: Any = _json.loads(raw_json)
    except _json.JSONDecodeError:
        return raw_json  # pass through; upstream will raise a cleaner error

    for rule in transforms:
        rule_type = rule.get("type")
        input_path: list = rule.get("input_path", [])

        if rule_type == "likelihood":
            # Direct lookup: replace the categorical token string with its float value.
            ctv: dict | None = rule.get("categories_to_values")
            if ctv is None:
                continue
            # Navigate to the parent and replace the leaf value.
            obj = data
            for key in input_path[:-1]:
                obj = obj[key]
            leaf_key = input_path[-1] if input_path else None
            if leaf_key is not None:
                raw_val = str(obj[leaf_key])
                obj[leaf_key] = ctv.get(raw_val, 0.0)
            else:
                # input_path == [] means the whole object is the value — unusual.
                raw_val = str(data)
                data = ctv.get(raw_val, 0.0)

        elif rule_type == "nest":
            field_name: str = rule["field_name"]
            obj = data
            for key in input_path[:-1]:
                obj = obj[key]
            if input_path:
                leaf_key = input_path[-1]
                obj[leaf_key] = {field_name: obj[leaf_key]}
            else:
                data = {field_name: data}

        elif rule_type == "project":
            retained: dict | list = rule["retained_fields"]
            if isinstance(retained, list):
                retained = {f: f for f in retained}
            obj = data
            for key in input_path[:-1]:
                obj = obj[key]
            if input_path:
                leaf_key = input_path[-1]
                src = obj[leaf_key]
                if isinstance(src, dict):
                    obj[leaf_key] = {retained[k]: src.get(k) for k in retained}
                elif isinstance(src, list):
                    obj[leaf_key] = [
                        {retained[k]: item.get(k) for k in retained} for item in src
                    ]
            else:
                if isinstance(data, dict):
                    data = {retained[k]: data.get(k) for k in retained}

        else:
            MelleaLogger.get_logger().debug(
                "_apply_transforms_no_logprobs: skipping unsupported transform type %r",
                rule_type,
            )

    return _json.dumps(data)


def _strip_data_uri_prefix(images: list[str]) -> list[str]:
    """Strip data URI prefix from base64 image strings for Ollama.

    Ollama expects raw base64 strings without the 'data:image/...;base64,' prefix.
    This function removes the prefix if present, leaving just the base64 data.

    Args:
        images: List of base64 image strings, potentially with data URI prefixes.

    Returns:
        List of base64 strings with data URI prefixes removed.
    """
    stripped = []
    for img in images:
        if "data:" in img and "base64," in img:
            img = img.split("base64,")[1]
        stripped.append(img)
    return stripped


class OllamaModelBackend(FormatterBackend, AdapterMixin):
    """A model that uses the Ollama Python SDK for local inference.

    Args:
        model_id (str | ModelIdentifier): Ollama model ID. If a
            ``ModelIdentifier`` is passed, its ``ollama_name`` attribute must
            be set.
        formatter (ChatFormatter | None): Formatter for rendering components.
            Defaults to ``TemplateFormatter``.
        base_url (str | None): Ollama server endpoint; defaults to
            ``env(OLLAMA_HOST)`` or ``http://localhost:11434``.
        model_options (dict | None): Default model options for generation requests.
        timeout (float | None): Per-operation HTTP timeout in seconds (connect,
            read, write, pool). Defaults to 300 s. For streaming requests this
            bounds the wait between consecutive chunks; for non-streaming requests
            it bounds total time-to-response. Pass ``None`` to use the upstream
            ``ollama`` SDK default (no timeout).

    Attributes:
        to_mellea_model_opts_map (dict): Mapping from Ollama-specific option names
            to Mellea ``ModelOption`` sentinel keys.
        from_mellea_model_opts_map (dict): Mapping from Mellea ``ModelOption``
            sentinel keys to Ollama-specific option names.

    Raises:
        ValueError: If ``model_id`` is a ``ModelIdentifier`` with no ``ollama_name`` set.
        ConnectionError: If the Ollama server is not running at ``base_url``.
        OSError: If the model cannot be pulled from the Ollama library.
    """

    def __init__(
        self,
        model_id: str | ModelIdentifier = model_ids.IBM_GRANITE_4_1_3B,
        formatter: ChatFormatter | None = None,
        base_url: str | None = None,
        model_options: dict | None = None,
        timeout: float | None = 300.0,
    ):
        """Initialize an Ollama backend, connecting to the server and pulling the model if needed."""
        super().__init__(
            model_id=model_id,
            formatter=(
                formatter
                if formatter is not None
                else TemplateFormatter(model_id=model_id)
            ),
            model_options=model_options,
        )
        # Resolve to a concrete ollama model name; raises ValueError if no ollama_name is set.
        ollama_model_id = (
            model_id.ollama_name if isinstance(model_id, ModelIdentifier) else model_id
        )
        if ollama_model_id is None or ollama_model_id == "":
            raise ValueError(
                "Cannot create OllamaModelBackend: the ModelIdentifier has no ollama_name set. "
                "Check mellea/backends/model_ids.py and ensure the constant you are using "
                "has an ollama_name value, or pass the Ollama model tag as a plain string."
            )
        self._model_id: str = ollama_model_id
        self._provider: str = "ollama"

        # Registry for adapters registered via add_adapter; required by AdapterMixin._find_adapter.
        self._added_adapters: dict[str, IntrinsicAdapter] = {}

        # Maps intrinsic_name → resolved Ollama model tag for that intrinsic.
        # Populated by add_adapter(); consulted by _generate_from_intrinsic().
        # Derived automatically from _INTRINSIC_OLLAMA_PREFIX + base model size tag,
        # or set explicitly via add_adapter(..., ollama_model="...").
        self._intrinsic_model_map: dict[str, str] = {}

        # Setup the client and ensure that we have the model available.
        self._base_url = base_url
        self._timeout = timeout
        client_kwargs: dict[str, Any] = {}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client_kwargs = client_kwargs
        self._client = ollama.Client(base_url, **client_kwargs)

        self._client_cache = ClientCache(2)

        # Call once to set up an async client and prepopulate the cache.
        _ = self._async_client

        if not self._check_ollama_server():
            err = f"could not create OllamaModelBackend: ollama server not running at {base_url}"
            MelleaLogger.get_logger().error(err)
            raise ConnectionError(err)
        if not self._pull_ollama_model():
            err = (
                f"Model '{self._model_id}' could not be pulled from the Ollama library. "
                f"Check that the model name is correct (run 'ollama list' to see locally "
                f"available models, or 'ollama pull {self._model_id}' to fetch it manually)."
            )
            MelleaLogger.get_logger().error(err)
            raise OSError(err)

        # A mapping of common options for this backend mapped to their Mellea ModelOptions equivalent.
        self.to_mellea_model_opts_map = {
            "system": ModelOption.SYSTEM_PROMPT,
            "think": ModelOption.THINKING,
            "num_ctx": ModelOption.CONTEXT_WINDOW,
            "num_predict": ModelOption.MAX_NEW_TOKENS,
            "seed": ModelOption.SEED,
            "tools": ModelOption.TOOLS,
            "stream": ModelOption.STREAM,
            "stop": ModelOption.STOP_SEQUENCES,
        }

        # A mapping of Mellea specific ModelOptions to the specific names for this backend.
        self.from_mellea_model_opts_map = {
            ModelOption.CONTEXT_WINDOW: "num_ctx",
            ModelOption.MAX_NEW_TOKENS: "num_predict",
            ModelOption.SEED: "seed",
            ModelOption.STOP_SEQUENCES: "stop",
        }

    # ------------------------------------------------------------------
    # AdapterMixin implementation
    # ------------------------------------------------------------------

    @property
    def base_model_name(self) -> str:
        """Return the short model name used for adapter catalog lookup.

        Strips an ``org/`` prefix if present, mirroring the convention used by
        ``OpenAIBackend``. For example, ``"ibm-granite/granite-3.3-8b-instruct"``
        returns ``"granite-3.3-8b-instruct"``, while a plain tag such as
        ``"granite3.3:8b"`` is returned unchanged.

        Returns:
            str: The short model name.
        """
        if "/" in self._model_id:
            return self._model_id.split("/")[1]
        return self._model_id

    def add_adapter(
        self, adapter: IntrinsicAdapter, *, ollama_model: str | None = None
    ) -> None:
        """Register an adapter with this backend.

        Ollama does not support hot-loading PEFT adapter weights at runtime.
        This method records the adapter's I/O config (needed for
        ``call_intrinsic`` prompt formatting) and resolves which Ollama model
        tag should be used to serve that intrinsic.

        The target Ollama model is resolved in this order:

        1. ``ollama_model`` — use exactly as given when provided.
        2. Auto-derive from ``_INTRINSIC_OLLAMA_PREFIX`` + size tag extracted
           from ``self._model_id``.  For example, a backend running
           ``"granite4.1:8b"`` will map ``"uncertainty"`` →
           ``"granite-uncertainty:8b"`` automatically.
        3. Fall back to ``self._model_id`` (the base model) at call time if
           neither of the above yields a result.  A warning is logged so the
           mismatch is visible.

        Args:
            adapter (IntrinsicAdapter): The adapter to register.
            ollama_model (str | None): Explicit Ollama model tag for this
                intrinsic (e.g. ``"granite-uncertainty:8b"``).  Pass this when
                your local Ollama naming differs from the IBM convention.
                Defaults to ``None`` (auto-derive).

        Raises:
            TypeError: If ``adapter`` is not an ``IntrinsicAdapter``.
            Exception: If ``adapter`` is already registered with a different backend.
        """
        if not isinstance(adapter, IntrinsicAdapter):
            raise TypeError(
                f"OllamaModelBackend only accepts IntrinsicAdapter. Got: {type(adapter).__name__}. "
                "Use LocalHFBackend for full PEFT adapter weight loading."
            )
        if adapter.backend is not None and adapter.backend is not self:
            raise Exception(
                f"adapter {adapter.name!r} has already been added to a different backend: {adapter.backend}"
            )
        if adapter.qualified_name in self._added_adapters:
            MelleaLogger.get_logger().warning(
                "Adapter %r is already registered with this backend; skipping duplicate add_adapter call.",
                adapter.qualified_name,
            )
            return
        adapter.backend = self
        self._added_adapters[adapter.qualified_name] = adapter

        # Resolve and store the Ollama model tag for this intrinsic.
        if ollama_model is not None:
            resolved = ollama_model
        else:
            # Auto-derive: prefix from the catalog map + size tag from self._model_id.
            prefix = _INTRINSIC_OLLAMA_PREFIX.get(adapter.intrinsic_name)
            size = self._model_id.split(":")[-1] if ":" in self._model_id else None
            if prefix and size:
                resolved = f"{prefix}:{size}"
            else:
                resolved = None  # will fall back to self._model_id at call time

        if resolved is not None:
            self._intrinsic_model_map[adapter.intrinsic_name] = resolved
            MelleaLogger.get_logger().debug(
                "Adapter %r will route intrinsic calls to Ollama model %r.",
                adapter.intrinsic_name,
                resolved,
            )
        else:
            MelleaLogger.get_logger().warning(
                "Could not resolve an Ollama model tag for intrinsic %r "
                "(base model %r has no ':tag' suffix and no ollama_model= was given). "
                "Intrinsic calls will fall back to the base model, which will likely "
                "produce incorrect output. Pass ollama_model= explicitly to fix this.",
                adapter.intrinsic_name,
                self._model_id,
            )

    def load_adapter(self, adapter_qualified_name: str) -> None:
        """No-op for Ollama — adapter weights are pre-merged into the model.

        Args:
            adapter_qualified_name (str): Qualified name of the adapter to load.
        """
        MelleaLogger.get_logger().debug(
            "load_adapter(%r) is a no-op for OllamaModelBackend.",
            adapter_qualified_name,
        )

    def unload_adapter(self, adapter_qualified_name: str) -> None:
        """No-op for Ollama — there are no adapter weights to unload.

        Args:
            adapter_qualified_name (str): Qualified name of the adapter to unload.
        """
        MelleaLogger.get_logger().debug(
            "unload_adapter(%r) is a no-op for OllamaModelBackend.",
            adapter_qualified_name,
        )

    def list_adapters(self) -> list[str]:
        """Return the qualified names of all registered adapters.

        Returns:
            list[str]: Qualified adapter names registered via ``add_adapter``.
        """
        return list(self._added_adapters.keys())

    def _check_ollama_server(self) -> bool:
        """Requests generic info about the Ollama server to ensure it's running."""
        try:
            self._client.ps()
        except (ConnectionError, httpx.TimeoutException, httpx.ConnectError):
            return False
        return True

    def is_model_available(self, model_name):
        """Checks if a specific Ollama model is available locally.

        Args:
          model_name: The name of the model to check for (e.g., "llama2").

        Returns:
          True if the model is available, False otherwise.
        """
        try:
            models = self._client.list()
            for model in models["models"]:
                if model.model.startswith(model_name):
                    return True
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def _pull_ollama_model(self) -> bool:
        """Either gets the cached ollama model or else attempts to pull the provided model from Ollama. Raises an exception of the model cannot be pulled.

        This code was generated by ChatGPT.
        """
        if self.is_model_available(self._model_id):
            return True

        try:
            MelleaLogger.get_logger().debug(
                f"Loading/Pulling model from Ollama: {self._model_id}"
            )
            stream = self._client.pull(self._model_id, stream=True)
            progress_bars = {}
            for update in stream:
                status = update.status
                digest = update.digest
                completed = update.completed or 0
                total = update.total or 0
                if digest and total > 0:
                    if digest not in progress_bars:
                        progress_bars[digest] = tqdm(
                            total=total,
                            desc=f"{status} {digest[:12]}",
                            unit="B",
                            unit_scale=True,
                            leave=False,
                        )
                    pbar = progress_bars[digest]
                    delta = completed - pbar.n
                    if delta > 0:
                        pbar.update(delta)
            for pbar in progress_bars.values():
                pbar.close()
            return True
        except (ollama.ResponseError, httpx.TimeoutException, httpx.ConnectError):
            return False

    @property
    def _async_client(self) -> ollama.AsyncClient:
        """Ollama's client gets tied to a specific event loop. Reset it if needed here."""
        key = id(get_current_event_loop())

        _async_client = self._client_cache.get(key)
        if _async_client is None:
            _async_client = ollama.AsyncClient(self._base_url, **self._client_kwargs)
            self._client_cache.put(key, _async_client)
        return _async_client

    def _simplify_and_merge(
        self, model_options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Simplifies model_options to use the Mellea specific ModelOption.Option and merges the backend's model_options with those passed into this call.

        Rules:
        - Within a model_options dict, existing keys take precedence. This means remapping to mellea specific keys will maintain the value of the mellea specific key if one already exists.
        - When merging, the keys/values from the dictionary passed into this function take precedence.

        Because this function simplifies and then merges, non-Mellea keys from the passed in model_options will replace
        Mellea specific keys from the backend's model_options.

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        backend_model_opts = ModelOption.replace_keys(
            self.model_options, self.to_mellea_model_opts_map
        )

        if model_options is None:
            return backend_model_opts

        generate_call_model_opts = ModelOption.replace_keys(
            model_options, self.to_mellea_model_opts_map
        )
        merged = ModelOption.merge_model_options(
            backend_model_opts, generate_call_model_opts
        )
        return merged

    def _make_backend_specific_and_remove(
        self, model_options: dict[str, Any]
    ) -> dict[str, Any]:
        """Maps specified Mellea specific keys to their backend specific version and removes any remaining Mellea keys.

        Args:
            model_options: the model_options for this call

        Returns:
            a new dict
        """
        for opt, field in (
            (ModelOption.LOGITS, "generation.logits"),
            (ModelOption.RAW_LOGITS, "generation.raw_logits"),
        ):
            if model_options.get(opt) and opt not in self._warned_about:
                self._warned_about.add(opt)
                MelleaLogger.get_logger().warning(
                    f"{opt!r} is not supported by the Ollama backend; {field} will be None."
                )

        backend_specific = ModelOption.replace_keys(
            model_options, self.from_mellea_model_opts_map
        )
        return ModelOption.remove_special_keys(backend_specific)

    async def _generate_from_intrinsic(
        self,
        action: Intrinsic,
        ctx: Context,
        *,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk:
        """Generate a completion for an ``Intrinsic`` action via the Ollama chat API.

        Uses ``IntrinsicsRewriter`` (from the adapter's ``io.yaml``) to inject the
        control token / instruction message that activates structured output in the
        pre-merged Ollama intrinsic model.  The raw JSON the model emits is returned
        directly — e.g. ``{"score": "9"}`` for uncertainty or
        ``{"score": "yes"/"no"}`` for requirement-check.

        Args:
            action (Intrinsic): The intrinsic component to execute.
            ctx (Context): The current generation context (must be a chat context).
            model_options (dict | None): Per-call model options that override defaults.
            tool_calls (bool): If ``True``, expose available tools to the model.

        Returns:
            ModelOutputThunk: Thunk containing the raw JSON string from the model.

        Raises:
            ValueError: If no adapter is registered for the requested intrinsic.
            TypeError: If the registered adapter is not an ``IntrinsicAdapter``.
            NotImplementedError: If streaming is requested.
        """
        assert ctx.is_chat_context, "OllamaModelBackend only supports chat contexts."

        model_opts = self._simplify_and_merge(model_options)

        if model_opts.get(ModelOption.STREAM, False):
            raise NotImplementedError("Intrinsics do not support streaming.")

        # ── Collect tools ────────────────────────────────────────────────────
        tools: dict[str, AbstractMelleaTool] = {}
        if tool_calls:
            add_tools_from_model_options(tools, model_opts)
            add_tools_from_context_actions(tools, ctx.actions_for_available_tools())

        # ── Build the base conversation dict ─────────────────────────────────
        linearized_ctx = ctx.view_for_generation()
        assert linearized_ctx is not None
        ctx_as_messages: list[Message] = self.formatter.to_chat_messages(linearized_ctx)

        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        conversation: list[dict] = []
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})
        conversation.extend([message_to_openai_message(m) for m in ctx_as_messages])

        # Extract documents attached to context messages (needed for context-attribution).
        docs = messages_to_docs(ctx_as_messages)

        # ── Find the registered adapter ──────────────────────────────────────
        allowed_types = tuple(at.value for at in action.adapter_types)
        adapter = self._find_adapter(action.intrinsic_name, allowed_types)
        if adapter is None:
            raise ValueError(
                f"OllamaModelBackend has no adapter registered for intrinsic "
                f"{action.intrinsic_name!r}. Pre-register it via add_adapter() "
                f"before calling call_intrinsic()."
            )
        if not isinstance(adapter, IntrinsicAdapter):
            raise TypeError(
                f"OllamaModelBackend only supports IntrinsicAdapter; "
                f"got {type(adapter).__name__!r}."
            )

        # ── Apply io.yaml prompt rewriting ───────────────────────────────────
        rewriter = granite_formatters.IntrinsicsRewriter(
            config_dict=adapter.config, model_name=adapter.name
        )

        formatted_tools = [t["function"] for t in convert_tools_to_json(tools)]
        request_json: dict = {
            "messages": conversation,
            "extra_body": {"documents": docs},
            "tools": formatted_tools if formatted_tools else None,
        }
        rewritten = rewriter.transform(request_json, **action.intrinsic_kwargs)

        # Apply any user-supplied temperature override.
        temperature = model_opts.get(ModelOption.TEMPERATURE, None)
        if temperature is not None:
            rewritten = rewritten.model_copy(update={"temperature": temperature})

        # Extract max_tokens from rewritten request (from io.yaml parameters).
        rewritten_params = rewritten.model_dump(exclude_none=True)
        num_predict = rewritten_params.get(
            "max_completion_tokens", model_opts.get(ModelOption.MAX_NEW_TOKENS, 15)
        )

        # ── Resolve the Ollama model to call ─────────────────────────────────
        ollama_model_tag = self._intrinsic_model_map.get(
            action.intrinsic_name, self._model_id
        )

        # ── Send to Ollama ────────────────────────────────────────────────────
        ollama_messages = [m.model_dump(exclude_none=True) for m in rewritten.messages]

        # Pass the io.yaml response_format schema as Ollama's `format` parameter
        # to engage constrained decoding and guarantee valid JSON output.
        _raw_fmt = adapter.config.get("response_format")
        if isinstance(_raw_fmt, str):
            response_format_schema: dict | None = _json.loads(_raw_fmt)
        else:
            response_format_schema = _raw_fmt  # already a dict or None

        raw_response: ollama.ChatResponse = await self._async_client.chat(
            model=ollama_model_tag,
            messages=ollama_messages,
            options={"temperature": temperature or 0.0, "num_predict": num_predict},
            format=response_format_schema,
            stream=False,
        )

        raw_content: str = raw_response.message.content or ""
        if not raw_content.strip():
            MelleaLogger.get_logger().warning(
                "Intrinsic '%s' via Ollama model '%s' returned an empty response. "
                "Verify that the model is available locally ('ollama list') and is "
                "the correct intrinsic model for this capability.",
                action.intrinsic_name,
                ollama_model_tag,
            )

        # Apply the io.yaml output transformations.
        #
        # Two-phase strategy:
        # 1. `_apply_transforms_no_logprobs` handles `likelihood` rules by
        #    direct categories_to_values lookup (Ollama provides no logprobs).
        #    It also handles simple structural rules (nest, project) and skips
        #    complex ones (decode_sentences, explode, drop_duplicates) that
        #    need the ChatCompletion request object.
        # 2. `IntrinsicsResultProcessor` then runs the full transform chain
        #    (including decode_sentences) using the rewritten ChatCompletion
        #    as context.  It re-reads its own config, so any rule already
        #    applied in phase 1 will be a no-op on the already-transformed
        #    JSON (likelihood is now a float, not a string token; nest/project
        #    have already restructured the dict).
        #
        # For simple intrinsics (uncertainty, requirement-check) that have no
        # decode_sentences rules, phase 2 is a near-no-op: likelihood sees a
        # float value that doesn't match any category string and returns 0.0.
        # To avoid double-applying likelihood, we only run phase 2 when the
        # config has transforms that _apply_transforms_no_logprobs skips.
        transforms = adapter.config.get("transformations") or []
        skipped_types = {"explode", "drop_duplicates", "decode_sentences"}
        needs_processor = any(r.get("type") in skipped_types for r in transforms)

        pre_content = _apply_transforms_no_logprobs(raw_content, transforms)

        if needs_processor:
            processor = granite_formatters.IntrinsicsResultProcessor(
                config_dict=adapter.config
            )
            synthetic_response = ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=AssistantMessage(content=pre_content),
                        finish_reason="stop",
                    )
                ]
            )
            transformed_response = processor.transform(synthetic_response, rewritten)
            processed_content = transformed_response.choices[0].message.content or ""
        else:
            processed_content = pre_content

        mot = ModelOutputThunk(value=processed_content)
        mot.generation.model = ollama_model_tag
        mot.generation.provider = self._provider

        generate_log = GenerateLog()
        generate_log.backend = f"ollama::{ollama_model_tag}"
        generate_log.date = datetime.datetime.now()
        generate_log.model_options = model_opts
        generate_log.model_output = processed_content
        generate_log.action = action
        mot._generate_log = generate_log

        return mot

    async def _generate_from_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[ModelOutputThunk[C], Context]:
        """Generate a completion for ``action`` given ``ctx`` via the Ollama chat API.

        Routes ``Intrinsic`` and ``ALoraRequirement`` actions to
        ``_generate_from_intrinsic`` so that io.yaml prompt rewriting and output
        post-processing are applied. All other actions are forwarded to
        ``generate_from_chat_context``.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context (must be a chat context).
            format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured/constrained output decoding.
            model_options (dict | None): Per-call model options that override the
                backend's defaults.
            tool_calls (bool): If ``True``, expose available tools to the model and
                parse tool-call responses.

        Returns:
            tuple[ModelOutputThunk[C], Context]: A thunk holding the (lazy) model output
                and an updated context that includes ``action`` and the new output.
        """
        assert ctx.is_chat_context, (
            "The ollama backend only supports chat-like contexts."
        )

        _model_id_str = str(getattr(self, "model_id", "unknown"))
        with with_context(request_id=generate_request_id(), model_id=_model_id_str):
            model_opts = self._simplify_and_merge(model_options)

            # ALoraRequirement is a subclass of both Requirement and Intrinsic,
            # so this branch must come first.
            if isinstance(action, ALoraRequirement):
                mot = await self._generate_from_intrinsic(
                    action, ctx, model_options=model_opts, tool_calls=tool_calls
                )
                return mot, ctx.add(action).add(mot)

            if isinstance(action, Intrinsic):
                mot = await self._generate_from_intrinsic(
                    action, ctx, model_options=model_opts, tool_calls=tool_calls
                )
                return mot, ctx.add(action).add(mot)

            mot = await self.generate_from_chat_context(
                action,
                ctx,
                _format=format,
                model_options=model_options,
                tool_calls=tool_calls,
            )

        return mot, ctx.add(action).add(mot)

    async def generate_from_chat_context(
        self,
        action: Component[C] | CBlock | ModelOutputThunk,
        ctx: Context,
        *,
        _format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> ModelOutputThunk[C]:
        """Generate a new completion from the provided context using this backend's formatter.

        Treats the ``Context`` as a chat history and uses the ``ollama.Client.chat()``
        interface to generate a completion. Returns a thunk that lazily resolves
        the model output.

        Args:
            action (Component[C] | CBlock): The component or content block to generate
                a completion for.
            ctx (Context): The current generation context (must be a chat context).
            _format (type[BaseModelSubclass] | None): Optional Pydantic model class for
                structured output decoding.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): If ``True``, expose available tools and parse responses.

        Returns:
            ModelOutputThunk[C]: A thunk holding the (lazy) model output.

        Raises:
            RuntimeError: If not called from a thread with a running event loop.
            ValueError: If a message contains an ``ImageUrlBlock``; Ollama requires
                base64-encoded images — convert to an ``ImageBlock`` first.
            ValueError: If a message contains an ``AudioBlock`` or ``AudioUrlBlock``;
                Ollama does not support audio input.
        """
        await self.do_generate_walk(action)

        model_opts = self._simplify_and_merge(model_options)

        linearized_context = ctx.view_for_generation()
        assert linearized_context is not None, (
            "Cannot generate from a non-linear context in a FormatterBackend."
        )
        messages: list[Message] = self.formatter.to_chat_messages(linearized_context)
        messages.extend(self.formatter.to_chat_messages([action]))

        conversation: list[dict] = []
        system_prompt = model_opts.get(ModelOption.SYSTEM_PROMPT, "")
        if system_prompt != "":
            conversation.append({"role": "system", "content": system_prompt})

        for m in messages:
            if m.images is not None:
                for img in m.images:
                    if isinstance(img, ImageUrlBlock):
                        raise ValueError(
                            "OllamaModelBackend does not support URL images (ImageUrlBlock). "
                            "Convert the image to a base64-encoded ImageBlock before passing it to Ollama."
                        )
            if m.audio:
                raise ValueError(
                    "OllamaModelBackend does not support audio (AudioBlock/AudioUrlBlock). "
                    "Remove audio blocks before passing messages to Ollama."
                )
            conversation.append(
                {
                    "role": m.role,
                    "content": self.formatter.print(m),
                    "images": (
                        _strip_data_uri_prefix([str(img.value) for img in m.images])
                        if m.images
                        else None
                    ),
                }
            )

        tools: dict[str, AbstractMelleaTool] = dict()
        if tool_calls:
            if _format:
                MelleaLogger.get_logger().warning(
                    f"Tool calling typically uses constrained generation, but you have specified a `format` in your generate call. NB: tool calling is superseded by format; we will NOT call tools for your request: {action}"
                )
            else:
                add_tools_from_model_options(tools, model_opts)
                add_tools_from_context_actions(tools, ctx.actions_for_available_tools())
                add_tools_from_context_actions(tools, [action])
            MelleaLogger.get_logger().info(f"Tools for call: {tools.keys()}")

        logprobs = model_opts.pop("logprobs", None)
        top_logprobs = model_opts.pop("top_logprobs", None)

        chat_response: Coroutine[
            Any, Any, AsyncIterator[ollama.ChatResponse] | ollama.ChatResponse
        ] = self._async_client.chat(
            model=self._model_id,
            messages=conversation,
            tools=[t.as_json_tool for t in tools.values()],
            think=model_opts.get(ModelOption.THINKING, None),
            stream=model_opts.get(ModelOption.STREAM, False),
            options=self._make_backend_specific_and_remove(model_opts),
            format=_format.model_json_schema() if _format is not None else None,  # type: ignore
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )  # type: ignore

        output = ModelOutputThunk(None)
        output._gen.start = datetime.datetime.now()
        output._call.context = linearized_context
        output._call.action = action
        output._call.model_options = model_opts

        output._gen.process = functools.partial(self.processing, tools=tools)
        output._gen.post_process = functools.partial(
            self.post_processing,
            conversation=conversation,
            tools=tools,
            _format=_format,
        )

        output.generation.model = self._model_id
        output.generation.provider = self._provider

        try:
            output._gen.generate = asyncio.create_task(
                send_to_queue(
                    chat_response,
                    output._gen.queue,
                    chunk_timeout=model_opts.get(
                        ModelOption.STREAM_TIMEOUT, DEFAULT_CHUNK_TIMEOUT
                    ),
                )
            )
            output._gen.generate_type = GenerateType.ASYNC
        except RuntimeError as e:
            raise e

        return output

    async def _generate_from_raw(
        self,
        actions: Sequence[Component[C] | CBlock],
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict | None = None,
        tool_calls: bool = False,
    ) -> tuple[list[ModelOutputThunk], dict[str, Any] | None]:
        """Generate completions for multiple actions without chat templating via Ollama.

        Passes formatted prompt strings directly to Ollama's generate endpoint.
        Requests are submitted concurrently to make use of Ollama's concurrency support.

        Args:
            actions (Sequence[Component[C] | CBlock]): Actions to generate completions for.
            ctx (Context): The current generation context.
            format (type[BaseModelSubclass] | None): Optional Pydantic model for
                structured output decoding.
            model_options (dict | None): Per-call model options.
            tool_calls (bool): Ignored; tool calling is not supported on this endpoint.

        Returns:
            tuple[list[ModelOutputThunk], dict | None]: `(results, usage)` where
                `results` is a list of model output thunks, one per action, and
                `usage` is the aggregate token-usage dict for the batch (or `None`
                when no request reported usage).

                If Ollama returns an empty done response (``response=""``,
                ``done=True``, no thinking content) for an action, that thunk
                soft-fails: it has ``value=""`` and ``thunk.error`` carries the
                ``RuntimeError`` describing the cause. Other actions in the
                batch are unaffected.

        Note:
            Requests are awaited with ``asyncio.gather`` (all-or-nothing): if any
            request raises (e.g. ``ollama.ResponseError`` or a connection error),
            that exception propagates to the caller and no list is returned, even
            for requests that completed successfully.
        """
        if len(actions) > 1:
            MelleaLogger.get_logger().info(
                "Ollama doesn't support batching; will attempt to process concurrently."
            )
        if tool_calls:
            MelleaLogger.get_logger().warning(
                "The completion endpoint does not support tool calling at the moment."
            )

        model_opts = self._simplify_and_merge(model_options)

        await self.do_generate_walks(list(actions))
        prompts = [self.formatter.print(action) for action in actions]

        coroutines: list[Coroutine[Any, Any, ollama.GenerateResponse]] = []
        for prompt in prompts:
            co = self._async_client.generate(
                model=self._model_id,
                prompt=prompt,
                raw=True,
                think=model_opts.get(ModelOption.THINKING, None),
                format=format.model_json_schema() if format is not None else None,  # type: ignore
                options=self._make_backend_specific_and_remove(model_opts),
            )
            coroutines.append(co)

        responses = await asyncio.gather(*coroutines)

        results = []
        date = datetime.datetime.now()
        agg_prompt = 0
        agg_completion = 0
        for i, response in enumerate(responses):
            result = None
            per_mot_usage: dict[str, Any] | None = None
            if response.done and not response.response and not response.thinking:
                empty_err = RuntimeError(
                    f"generate_from_raw: request {i} returned an empty response from Ollama "
                    "(response='', done=True). This commonly occurs when the model is still "
                    "loading, but can also indicate an early stop or stop-sequence hit. "
                    "See https://github.com/generative-computing/mellea/issues/599 "
                    "and https://github.com/ollama/ollama/issues/16326"
                )
                MelleaLogger.get_logger().warning(str(empty_err))
                result = ModelOutputThunk(value="")
                result._error = empty_err
            else:
                n_in = response.prompt_eval_count
                n_out = response.eval_count
                if n_in is not None and n_out is not None:
                    agg_prompt += n_in
                    agg_completion += n_out
                    per_mot_usage = {
                        "prompt_tokens": n_in,
                        "completion_tokens": n_out,
                        "total_tokens": n_in + n_out,
                    }
                result = ModelOutputThunk(value=response.response)
                result.raw = RawProviderResponse(
                    provider=self._provider, response=response.model_dump()
                )
            result.generation.usage = per_mot_usage
            result.generation.model = self._model_id
            result.generation.provider = self._provider

            action = actions[i]
            result.parsed_repr = (
                action.parse(result) if isinstance(action, Component) else result.value
            )

            generate_log = GenerateLog()
            generate_log.prompt = prompts[i]
            generate_log.backend = f"ollama::{self.model_id!s}"
            generate_log.date = date
            generate_log.model_options = model_opts
            generate_log.model_output = result.value
            generate_log.extra = {
                "format": format,
                "thinking": model_opts.get(ModelOption.THINKING, None),
                "seed": model_opts.get(ModelOption.SEED, None),
            }
            generate_log.action = action
            result._generate_log = generate_log

            results.append(result)

        usage: dict[str, Any] | None = (
            {
                "prompt_tokens": agg_prompt,
                "completion_tokens": agg_completion,
                "total_tokens": agg_prompt + agg_completion,
            }
            if (agg_prompt or agg_completion)
            else None
        )
        return results, usage

    def _extract_model_tool_requests(
        self, tools: dict[str, AbstractMelleaTool], chat_response: ollama.ChatResponse
    ) -> dict[str, ModelToolCall] | None:
        from .tools import validate_tool_arguments

        model_tool_calls: dict[str, ModelToolCall] = {}

        if chat_response.message.tool_calls:
            for tool in chat_response.message.tool_calls:
                func = tools.get(tool.function.name)
                if func is None:
                    MelleaLogger.get_logger().warning(
                        f"model attempted to call a non-existing function: {tool.function.name}"
                    )
                    continue

                args = tool.function.arguments
                validated_args = validate_tool_arguments(func, args, strict=False)
                model_tool_calls[tool.function.name] = ModelToolCall(
                    tool.function.name, func, validated_args
                )

        if len(model_tool_calls) > 0:
            return model_tool_calls
        return None

    async def processing(
        self,
        mot: ModelOutputThunk,
        chunk: ollama.ChatResponse,
        tools: dict[str, AbstractMelleaTool],
    ):
        """Accumulate text and tool calls from a single Ollama ChatResponse chunk.

        Called for each streaming or non-streaming `ollama.ChatResponse`. Also
        extracts tool call requests inline and merges the chunk into the running
        aggregated response stored in `mot.raw.response`.

        Args:
            mot (ModelOutputThunk): The output thunk being populated.
            chunk (ollama.ChatResponse): A single chat response object from Ollama.
            tools (dict[str, AbstractMelleaTool]): Available tools, keyed by name,
                used for extracting tool call requests from the response.
        """
        if mot.thinking is None:
            mot.thinking = ""
        thinking_chunk = chunk.message.thinking
        if thinking_chunk is not None:
            mot.thinking += thinking_chunk

        if mot._underlying_value is None:
            mot._underlying_value = ""
        content_chunk = chunk.message.content
        if content_chunk is not None:
            mot._underlying_value += content_chunk

        tool_chunk = self._extract_model_tool_requests(tools, chunk)
        if tool_chunk is not None:
            if mot.tool_calls is None:
                mot.tool_calls = {}
            for key, val in tool_chunk.items():
                mot.tool_calls[key] = val

        chat_response_delta_merge(mot, chunk)

    async def post_processing(
        self,
        mot: ModelOutputThunk,
        conversation: list[dict],
        tools: dict[str, AbstractMelleaTool],
        _format,
    ):
        """Finalize the output thunk after Ollama generation completes.

        Attaches the generate log, records token usage metrics, emits telemetry,
        and cleans up the span reference.

        Args:
            mot (ModelOutputThunk): The output thunk to finalize.
            conversation (list[dict]): The chat conversation sent to the model,
                used for logging.
            tools (dict[str, AbstractMelleaTool]): Available tools, keyed by name.
            _format: The structured output format class used during generation, if any.
        """
        assert mot._call.action is not None, (
            "ModelOutputThunks should have their action assigned during generation"
        )
        assert mot._call.model_options is not None, (
            "ModelOutputThunks should have their model_opts assigned during generation"
        )

        generate_log = GenerateLog()
        generate_log.prompt = conversation
        generate_log.backend = f"ollama::{self._model_id}"
        generate_log.model_options = mot._call.model_options
        generate_log.date = datetime.datetime.now()
        generate_log.model_output = mot.raw.response
        generate_log.extra = {
            "format": _format,
            "thinking": mot._call.model_options.get(ModelOption.THINKING, None),
            "tools_available": tools,
            "tools_called": mot.tool_calls,
            "seed": mot._call.model_options.get(ModelOption.SEED, None),
        }
        generate_log.action = mot._call.action
        generate_log.result = mot

        mot._generate_log = generate_log
        mot._gen.generate = None

        response = mot.raw.response
        prompt_tokens = (
            getattr(response, "prompt_eval_count", None) if response else None
        )
        completion_tokens = getattr(response, "eval_count", None) if response else None

        if prompt_tokens is not None and completion_tokens is not None:
            mot.generation.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        mot.generation.model = self._model_id
        mot.generation.provider = self._provider
        mot.raw.provider = self._provider

        if response is not None:
            mot.generation.response_model = getattr(response, "model", None)
            if done_reason := getattr(response, "done_reason", None):
                mot.generation.finish_reasons = [done_reason]


def chat_response_delta_merge(mot: ModelOutputThunk, delta: ollama.ChatResponse):
    """Merges the individual ChatResponse chunks from a streaming response into a single ChatResponse.

    Args:
        mot: the ModelOutputThunk that the deltas are being used to populated.
        delta: the most recent ollama ChatResponse.
    """
    if mot.raw.response is None:
        mot.raw.response = delta
        return

    merged: ollama.ChatResponse = mot.raw.response
    if not merged.done:
        merged.done = delta.done
    if merged.done_reason is None:
        merged.done_reason = delta.done_reason
    if merged.total_duration is None:
        merged.total_duration = delta.total_duration
    if merged.load_duration is None:
        merged.load_duration = delta.load_duration
    if merged.prompt_eval_count is None:
        merged.prompt_eval_count = delta.prompt_eval_count
    if merged.prompt_eval_duration is None:
        merged.prompt_eval_duration = delta.prompt_eval_duration
    if merged.eval_count is None:
        merged.eval_count = delta.eval_count

    if merged.message.role == "":
        merged.message.role = delta.message.role

    if merged.message.content is None:
        merged.message.content = delta.message.content
    elif delta.message.content is not None:
        merged.message.content += delta.message.content

    if merged.message.thinking is None:
        merged.message.thinking = delta.message.thinking
    elif delta.message.thinking is not None:
        merged.message.thinking += delta.message.thinking

    if merged.message.tool_calls is None:
        merged.message.tool_calls = delta.message.tool_calls
    elif delta.message.tool_calls is not None:
        merged.message.tool_calls = [
            *merged.message.tool_calls,
            *delta.message.tool_calls,
        ]
