"""
Pluggable LLM backend so the agent isn't locked to a single paid provider.

Two backends are implemented:

  - AnthropicBackend: native Anthropic SDK (Claude). Requires a paid or
    trial-credit API key.
  - OpenAICompatBackend: works with ANY provider that speaks the OpenAI
    chat-completions wire format with tool/function calling. This covers
    three genuinely free options with no code changes, just different
    .env values:
      * Groq         (https://console.groq.com)            -- free tier
      * Google Gemini (https://aistudio.google.com/apikey)  -- free tier
      * Ollama        (https://ollama.com, runs locally)    -- 100% free

Both backends normalize their responses into the same tiny vocabulary
(TextBlock / ToolUseBlock) so src/agent.py never has to know which
provider is underneath. The one place wire formats genuinely differ is how
a tool-call turn gets appended back into the running message list, which
is why `append_tool_turn` is a method on the backend rather than something
agent.py builds by hand.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


Block = TextBlock | ToolUseBlock


def anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert our Anthropic-shaped tool schemas (name/description/input_schema)
    into OpenAI's {"type": "function", "function": {...}} shape."""
    out = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return out


class LLMBackend(ABC):
    @abstractmethod
    def create(self, system: str, messages: list, tools: list[dict]) -> tuple[list[Block], Any]:
        """Call the model. Returns (normalized_blocks, raw_provider_response_piece).
        `raw_provider_response_piece` is opaque to the caller and only ever
        round-tripped back into `append_tool_turn`."""

    @abstractmethod
    def append_tool_turn(self, messages: list, raw_assistant_piece: Any, tool_outputs: list[tuple[str, str, str]]) -> None:
        """Mutates `messages` in place: appends the assistant's tool-call turn,
        then the tool result(s). tool_outputs is a list of
        (tool_use_id, tool_name, result_json_string)."""


class AnthropicBackend(LLMBackend):
    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def create(self, system, messages, tools):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )
        blocks: list[Block] = []
        for b in response.content:
            if b.type == "text":
                blocks.append(TextBlock(text=b.text))
            elif b.type == "tool_use":
                blocks.append(ToolUseBlock(id=b.id, name=b.name, input=b.input))
        return blocks, response.content

    def append_tool_turn(self, messages, raw_assistant_piece, tool_outputs):
        messages.append({"role": "assistant", "content": raw_assistant_piece})
        tool_results = [
            {"type": "tool_result", "tool_use_id": tid, "content": content}
            for tid, _name, content in tool_outputs
        ]
        messages.append({"role": "user", "content": tool_results})


class OpenAICompatBackend(LLMBackend):
    """Works with Groq, Google Gemini, Ollama, OpenRouter, or any other
    OpenAI-compatible chat-completions endpoint -- just point base_url and
    api_key at the provider you want."""

    def __init__(self, api_key: str, model: str, base_url: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self.model = model

    def create(self, system, messages, tools):
        full_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=anthropic_tools_to_openai(tools),
            max_tokens=1024,
        )
        msg = response.choices[0].message
        blocks: list[Block] = []
        if msg.content:
            blocks.append(TextBlock(text=msg.content))
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            blocks.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=args))
        return blocks, msg

    def append_tool_turn(self, messages, raw_assistant_piece, tool_outputs):
        assistant_msg = {"role": "assistant", "content": raw_assistant_piece.content}
        if raw_assistant_piece.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in raw_assistant_piece.tool_calls
            ]
        messages.append(assistant_msg)
        for tid, _name, content in tool_outputs:
            messages.append({"role": "tool", "tool_call_id": tid, "content": content})


# Presets for the free providers so .env only needs LLM_PROVIDER + one key.
# Presets for free/openly-available providers so .env only needs
# LLM_PROVIDER + one key. All of these speak the OpenAI-compatible
# chat-completions wire format, so they all route through OpenAICompatBackend.
PROVIDER_PRESETS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "signup": "https://console.groq.com -- free tier, no card required",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "signup": "https://aistudio.google.com/apikey -- free tier, no card required",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
        "signup": "https://ollama.com -- fully local, no key or signup at all",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        # Free-tier models on OpenRouter carry a ":free" suffix. Others also
        # work here (deepseek/deepseek-chat, qwen/qwen-2.5-72b-instruct, ...)
        # if you'd rather use their paid-but-cheap versions.
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "signup": "https://openrouter.ai/keys -- one key covers many free models, "
                  "including DeepSeek and Qwen (look for the ':free' suffix)",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "signup": "https://platform.deepseek.com -- new accounts get free trial credit",
    },
    "qwen": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "signup": "https://bailian.console.alibabacloud.com -- Alibaba Model Studio, "
                  "free tier quota for new accounts",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        # NVIDIA NIM hosts many open models (Qwen, DeepSeek, Llama, Mistral...)
        # behind one endpoint -- change LLM_MODEL to switch between them.
        "default_model": "qwen/qwen2.5-72b-instruct",
        "signup": "https://build.nvidia.com -- free API key with a generous free-credit allotment",
    },
}


def build_backend() -> Optional[LLMBackend]:
    """Reads LLM_PROVIDER (+ related vars) from the environment and returns a
    ready-to-use backend, or None if nothing is configured."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        try:
            return AnthropicBackend(api_key=api_key, model=model)
        except ImportError:
            return None

    if provider in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[provider]
        api_key = os.environ.get("LLM_API_KEY", "")
        if provider != "ollama" and not api_key:
            return None  # ollama needs no real key; the others do
        model = os.environ.get("LLM_MODEL", preset["default_model"])
        base_url = os.environ.get("LLM_BASE_URL", preset["base_url"])
        try:
            return OpenAICompatBackend(api_key=api_key, model=model, base_url=base_url)
        except ImportError:
            return None

    if provider == "openai_compatible":
        # Fully custom endpoint, e.g. a self-hosted server not covered above.
        base_url = os.environ.get("LLM_BASE_URL")
        model = os.environ.get("LLM_MODEL")
        api_key = os.environ.get("LLM_API_KEY", "")
        if not base_url or not model:
            return None
        try:
            return OpenAICompatBackend(api_key=api_key, model=model, base_url=base_url)
        except ImportError:
            return None

    return None
