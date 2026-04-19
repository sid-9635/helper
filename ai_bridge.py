import aiohttp
import asyncio
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

from config import OPENAI_API_KEY as CONFIG_OPENAI_API_KEY

WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
GPT_URL = "https://api.openai.com/v1/chat/completions"
PROMPTS_DIR = Path(__file__).with_name("prompts")
COMMON_PROMPT_PATH = PROMPTS_DIR / "common_prompt.txt"
GPT4O_PROMPT_PATH = PROMPTS_DIR / "gpt_4o.txt"
GPT4O_MINI_PROMPT_PATH = PROMPTS_DIR / "gpt_4o_mini.txt"
GPT5_PROMPT_PATH = PROMPTS_DIR / "gpt_5.txt"
GENERIC_ANSWERS_PATH = PROMPTS_DIR / "generic_answers.jsonl"
GPT4O_MODEL = "gpt-4.1"
GPT5_MODEL = "gpt-5"
GPT_HINT_MODEL = "gpt-4o-mini"
GPT4O_MAX_TOKENS = 5324
GPT5_MAX_TOKENS = 5200
GPT5_FAST_MAX_TOKENS = 3600
HINT_MAX_TOKENS = 256
MAX_USER_INPUT_CHARS = 12000
CHAT_STREAMING_ENABLED = True


DEFAULT_COMMON_PROMPT = "You are an interview coach. Answer clearly, directly, and naturally."
DEFAULT_GPT4O_PROMPT = "Keep the response concise, clean, and practical."
DEFAULT_GPT4O_MINI_PROMPT = "You are a concise interview coach. Return only short, bulleted hints."
DEFAULT_GPT5_PROMPT = "Give a more complete and precise answer while staying structured and readable."
BASE_SYSTEM_MESSAGE = "You are a helpful interview coach. Follow the provided instructions exactly."

DEFAULT_GENERIC_ANSWERS = {
    "introduce_yourself": (
        "Sure. I’m a software engineer who likes working on practical backend and product problems, especially where I need to move from an idea to a working solution quickly. In my recent work, I’ve spent a lot of time on Python applications, API integrations, debugging, automation, and making systems more reliable under real usage. I’m usually at my best when the problem is a mix of engineering depth and execution, because I enjoy breaking things down, fixing root causes, and keeping the solution clean and maintainable."
    ),
    "roles_and_responsibilities": (
        "In my recent role, I was responsible for taking features from requirement to delivery. That included understanding the use case, designing the implementation, writing and testing the code, debugging issues, and making sure the final behavior was reliable in real usage. I also spent time improving existing flows where latency, stability, or usability were not good enough. So my work was not just writing code, it was owning the outcome end to end and making practical engineering decisions along the way."
    ),
    "strengths": (
        "One of my biggest strengths is that I’m very practical when solving engineering problems. I can usually break a vague issue into smaller parts quickly, identify what actually matters, and move toward a solution without adding unnecessary complexity. Another strength is debugging. I’m comfortable tracing behavior across components, finding the real cause, and fixing it in a way that holds up instead of just patching the symptom. I’d also say I have a strong sense of ownership, so I focus on whether the final result actually works for the user, not just whether the code compiles."
    ),
    "weaknesses": (
        "One weakness I’ve worked on is that I can spend too much time refining an implementation when I already have something that is good enough for the current need. Earlier, I used to optimize details a bit too early because I wanted the solution to feel complete from the start. Over time, I’ve improved that by being more deliberate about scope, shipping the version that solves the actual problem first, and then iterating only where the impact is real. That has helped me balance quality with speed much better."
    ),
}

_GENERIC_ANSWERS_CACHE_MTIME = None
_GENERIC_ANSWERS_CACHE = None

GENERIC_QUESTION_PATTERNS = {
    "introduce_yourself": {
        "strong_phrases": (
            "tell me about yourself",
            "introduce yourself",
            "can you introduce yourself",
            "walk me through your background",
            "give me a quick background",
            "tell me about your background",
            "quick introduction",
            "brief introduction",
        ),
        "token_groups": (("introduce", "introduction", "intro", "background"), ("yourself", "your", "you")),
    },
    "roles_and_responsibilities": {
        "strong_phrases": (
            "roles and responsibilities",
            "role and responsibilities",
            "what were your responsibilities",
            "what was your role",
            "day to day responsibilities",
            "what did you do in your last role",
            "walk me through your role",
            "your current role",
        ),
        "token_groups": (("role", "roles", "responsibilities", "responsibility"), ("current", "last", "recent", "day", "work")),
    },
    "strengths": {
        "strong_phrases": (
            "what are your strengths",
            "what is your greatest strength",
            "what is your biggest strength",
            "tell me your strengths",
            "strong points",
            "your strengths",
        ),
        "token_groups": (("strength", "strengths", "strong", "strongest"), ("your", "you", "biggest", "greatest")),
    },
    "weaknesses": {
        "strong_phrases": (
            "what are your weaknesses",
            "what is your biggest weakness",
            "what is your greatest weakness",
            "tell me your weaknesses",
            "area of improvement",
            "areas of improvement",
            "one weakness",
        ),
        "token_groups": (("weakness", "weaknesses", "weak", "improvement"), ("your", "you", "biggest", "greatest", "area", "areas")),
    },
}


def _is_gpt5_family(model: str) -> bool:
    return model.startswith("gpt-5")


def _token_limit_field(model: str, token_count: int) -> dict:
    if _is_gpt5_family(model):
        return {"max_completion_tokens": token_count}
    return {"max_tokens": token_count}


def _sampling_field(model: str) -> dict:
    if _is_gpt5_family(model):
        return {}
    return {"temperature": 0.3}


def _get_api_key():
    api_key = os.getenv("OPENAI_API_KEY") or CONFIG_OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in environment or config.py.")
    return api_key


def _hint_prompt(transcript: str) -> str:
    return (
        "Convert this interview transcript into very short, bulleted hints:\n"
        f"{transcript}\n"
        "Use only concise bullets, no paragraphs, and keep output token-efficient."
    )


def _read_prompt_file(path: Path, fallback: str) -> str:
    try:
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    except OSError:
        pass
    return fallback


def _common_prompt() -> str:
    return _read_prompt_file(COMMON_PROMPT_PATH, DEFAULT_COMMON_PROMPT)


def _model_prompt(selected_model: str) -> str:
    if selected_model == GPT5_MODEL:
        return _read_prompt_file(GPT5_PROMPT_PATH, DEFAULT_GPT5_PROMPT)
    if selected_model == GPT_HINT_MODEL:
        return _read_prompt_file(GPT4O_MINI_PROMPT_PATH, DEFAULT_GPT4O_MINI_PROMPT)
    return _read_prompt_file(GPT4O_PROMPT_PATH, DEFAULT_GPT4O_PROMPT)


def _instruction_messages(selected_model: str) -> list[dict[str, str]]:
    messages = []
    common_prompt = _common_prompt()
    model_prompt = _model_prompt(selected_model)

    if common_prompt:
        messages.append({"role": "system", "content": common_prompt})
    if model_prompt and model_prompt != common_prompt:
        messages.append({"role": "system", "content": model_prompt})

    return messages


def _resolve_model(selected_model: str) -> str:
    if selected_model == GPT5_MODEL:
        return GPT5_MODEL
    return GPT4O_MODEL


def _resolve_token_limit(model: str) -> int:
    if model == GPT5_MODEL:
        return GPT5_MAX_TOKENS
    return GPT4O_MAX_TOKENS


def _normalize_user_input(user_input: str) -> str:
    cleaned_input = (user_input or "").strip()
    if len(cleaned_input) <= MAX_USER_INPUT_CHARS:
        return cleaned_input
    return cleaned_input[:MAX_USER_INPUT_CHARS]


def _normalize_intent_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_answer_key(key: str) -> str:
    return re.sub(r"\s+", "_", _normalize_intent_text(key))


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokenize_for_match(text: str) -> set[str]:
    normalized = _normalize_intent_text(text)
    return {_stem_token(token) for token in normalized.split() if token}


def _load_relaxed_json_object(raw_text: str):
    cleaned = re.sub(r",(\s*[}\]])", r"\1", raw_text)
    return json.loads(cleaned, strict=False)


def _extract_answer_pairs_from_text(raw_text: str) -> dict[str, str]:
    pairs = {}
    pattern = re.compile(r'"(?P<key>[^"\\]+)"\s*:\s*"(?P<value>(?:[^"\\]|\\.|\r|\n)*)"', re.DOTALL)
    for match in pattern.finditer(raw_text):
        key = match.group("key")
        value = bytes(match.group("value"), "utf-8").decode("unicode_escape").strip()
        if key and value:
            pairs[key] = value
    return pairs


def _generic_answers() -> dict[str, str]:
    global _GENERIC_ANSWERS_CACHE_MTIME, _GENERIC_ANSWERS_CACHE

    try:
        stat = GENERIC_ANSWERS_PATH.stat()
        mtime_ns = stat.st_mtime_ns
        if _GENERIC_ANSWERS_CACHE is not None and _GENERIC_ANSWERS_CACHE_MTIME == mtime_ns:
            return dict(_GENERIC_ANSWERS_CACHE)

        raw_text = GENERIC_ANSWERS_PATH.read_text(encoding="utf-8-sig").strip()
        if not raw_text:
            return dict(DEFAULT_GENERIC_ANSWERS)

        merged = dict(DEFAULT_GENERIC_ANSWERS)

        # Backward-compatible path: accept a single JSON object even if the file
        # has a .jsonl extension.  Skip this path for multi-line JSONL files
        # (where every line is its own JSON object).
        _other_lines = [l.strip() for l in raw_text.splitlines()[1:] if l.strip()]
        _is_jsonl = bool(_other_lines) and _other_lines[0].startswith("{")
        if raw_text.startswith("{") and not _is_jsonl:
            try:
                payload = _load_relaxed_json_object(raw_text)
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if isinstance(key, str) and isinstance(value, str) and value.strip():
                            merged[key] = value.strip()
                    _GENERIC_ANSWERS_CACHE_MTIME = mtime_ns
                    _GENERIC_ANSWERS_CACHE = dict(merged)
                    return merged
            except json.JSONDecodeError:
                extracted_pairs = _extract_answer_pairs_from_text(raw_text)
                if extracted_pairs:
                    merged.update(extracted_pairs)
                    _GENERIC_ANSWERS_CACHE_MTIME = mtime_ns
                    _GENERIC_ANSWERS_CACHE = dict(merged)
                    return merged

        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line, strict=False)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            answer_id = payload.get("id") or payload.get("key") or payload.get("question_type")
            answer_text = payload.get("answer") or payload.get("text") or payload.get("response")

            if isinstance(answer_id, str) and isinstance(answer_text, str) and answer_text.strip():
                merged[answer_id] = answer_text.strip()
                continue

            if len(payload) == 1:
                only_key, only_value = next(iter(payload.items()))
                if isinstance(only_key, str) and isinstance(only_value, str) and only_value.strip():
                    merged[only_key] = only_value.strip()

        _GENERIC_ANSWERS_CACHE_MTIME = mtime_ns
        _GENERIC_ANSWERS_CACHE = dict(merged)
        return merged
    except OSError:
        pass
    except json.JSONDecodeError:
        pass
    return dict(DEFAULT_GENERIC_ANSWERS)


def _match_company_role_question(normalized: str, tokens: set[str], answers: dict[str, str]) -> str | None:
    role_cues = {
        "role", "roles", "work", "worked", "responsibility", "responsibilities",
        "did", "do", "explain", "tell", "describe", "about",
    }
    prepositions = {"at", "in", "with", "for"}

    role_cue_tokens = {_stem_token(cue) for cue in role_cues}
    if not tokens.intersection(role_cue_tokens):
        return None

    for answer_id, answer_text in answers.items():
        normalized_id = _normalize_answer_key(answer_id)
        match = re.match(r"^role_work_(?:at|in)_(.+)$", normalized_id)
        if not match or not answer_text.strip():
            continue

        company_slug = match.group(1).strip("_")
        if not company_slug:
            continue

        company_phrase = company_slug.replace("_", " ")
        company_tokens = {part for part in company_slug.split("_") if part and part not in prepositions}
        has_company_match = company_phrase in normalized or (company_tokens and company_tokens.issubset(tokens))
        if has_company_match:
            return answer_text

    return None


def _match_named_generic_answer(query_tokens: set[str], answers: dict[str, str]) -> str | None:
    stop_tokens = {"the", "a", "an", "what", "tell", "me", "about", "your", "did", "do", "in", "at", "for"}
    filtered_query_tokens = {token for token in query_tokens if token not in stop_tokens}
    if not filtered_query_tokens:
        return None

    best_answer = None
    best_score = 0.0

    for answer_id, answer_text in answers.items():
        if not answer_text or not answer_text.strip():
            continue

        answer_tokens = {
            token for token in _tokenize_for_match(answer_id.replace("_", " "))
            if token not in {"role", "work", "main"}
        }
        if not answer_tokens:
            continue

        overlap = filtered_query_tokens & answer_tokens
        if not overlap:
            continue

        # Allow single-token overlap only when:
        # - the query itself is a single token, OR
        # - the overlapping token is a specific/technical term (len > 5), meaning it's
        #   unlikely to be a coincidental match (e.g. "kubernetes", "grafana", "docker")
        if len(overlap) < 2:
            has_specific_term = any(len(t) > 5 for t in overlap)
            if len(filtered_query_tokens) != 1 and not has_specific_term:
                continue

        # Use the maximum of answer-side and query-side coverage so that long
        # compound IDs (e.g. microservices_aws_kubernetes_docker_port_forwarding)
        # are not unfairly penalised when the query's key term is clearly present.
        score = max(
            len(overlap) / len(answer_tokens),
            len(overlap) / len(filtered_query_tokens),
        )
        if overlap == filtered_query_tokens:
            score += 0.25
        score += 0.05 * len(overlap)
        if score > best_score:
            best_score = score
            best_answer = answer_text

    if best_score < 0.35:
        return None
    return best_answer


def _match_generic_question(text: str) -> str | None:
    # Screen captures and coding prompts produce long blobs — never match them
    # against generic Q&A. Real interview questions are always short (< 300 chars).
    if len(text) > 300:
        return None
    normalized = _normalize_intent_text(text)
    if not normalized:
        return None

    tokens = _tokenize_for_match(normalized)
    answers = _generic_answers()

    company_role_answer = _match_company_role_question(normalized, tokens, answers)
    if company_role_answer:
        return company_role_answer

    named_generic_answer = _match_named_generic_answer(tokens, answers)
    if named_generic_answer:
        return named_generic_answer

    best_match = None
    best_score = 0.0

    for answer_id, rule in GENERIC_QUESTION_PATTERNS.items():
        phrases = rule.get("strong_phrases", ())
        if any(phrase in normalized for phrase in phrases):
            score = 1.0
        else:
            groups = rule.get("token_groups", ())
            matched_groups = sum(1 for group in groups if any(_stem_token(token) in tokens for token in group))
            score = matched_groups / max(1, len(groups))
            if matched_groups and any(token in normalized for token in ("tell", "walk", "introduce", "describe", "what", "give")):
                score += 0.1

        if score > best_score:
            best_score = score
            best_match = answer_id

    if best_score < 0.95:
        return None
    return answers.get(best_match)


class AIBridge:
    _CACHE_MAX = 20  # maximum number of cached responses
    _MAX_CONTINUATIONS = 2

    def __init__(self, db):
        self.db = db
        self.session = None
        self._cached_api_key = None
        self._response_cache: OrderedDict[str, str] = OrderedDict()

    def match_generic_answer(self, text: str) -> str | None:
        """Return a stored generic answer if *text* matches one, else None."""
        return _match_generic_question(text)

    def _cache_get(self, key: str) -> str | None:
        """Return cached response for *key*, promoting it to most-recent."""
        normalized = key.strip().lower()
        if normalized in self._response_cache:
            self._response_cache.move_to_end(normalized)
            return self._response_cache[normalized]
        return None

    def _cache_set(self, key: str, value: str) -> None:
        """Store *value* under *key*, evicting the oldest entry if over capacity."""
        normalized = key.strip().lower()
        if normalized in self._response_cache:
            self._response_cache.move_to_end(normalized)
        self._response_cache[normalized] = value
        if len(self._response_cache) > self._CACHE_MAX:
            self._response_cache.popitem(last=False)  # evict oldest

    def _looks_incomplete_response(self, text: str) -> bool:
        stripped = (text or "").rstrip()
        if not stripped:
            return False
        if stripped.count("```") % 2 != 0:
            return True
        if stripped.endswith((":", ",", "(", "[", "{", "=", "->")):
            return True
        last_char = stripped[-1]
        if last_char.isalnum() or last_char in {'_', '"', "'"}:
            return True
        return False

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            # cache API key to avoid repeated getenv calls
            if not self._cached_api_key:
                self._cached_api_key = _get_api_key()

            # tuned for low-latency: DNS cached for 120s, 4s connect timeout, keepalive
            timeout = aiohttp.ClientTimeout(total=300, connect=4)
            connector = aiohttp.TCPConnector(
                limit=20,
                force_close=False,
                ssl=None,
                ttl_dns_cache=120,
                enable_cleanup_closed=True,
            )
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._cached_api_key}"},
                timeout=timeout,
                connector=connector,
                trust_env=True,
            )
        return self.session

    async def warmup(self):
        """Pre-open the HTTP session and resolve DNS so the first real request has no cold-start delay."""
        try:
            await self._ensure_session()
        except Exception:
            pass

    async def close(self):
        if self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None

    def _get_context(self, limit: int = 30):
        history = self.db.get_recent_messages(limit)
        return [
            {"role": role, "content": message}
            for role, message in history
        ]

    async def transcribe(self, wav_path: str) -> str | None:
        session = await self._ensure_session()
        data = aiohttp.FormData()
        file_name = Path(wav_path).name

        with open(wav_path, "rb") as audio_file:
            data.add_field("file", audio_file, filename=file_name, content_type="audio/wav")
            data.add_field("model", "whisper-1")

            async with session.post(WHISPER_URL, data=data) as resp:
                payload = await resp.json()
                return payload.get("text")

    async def _execute_chat_request(self, messages, api_model: str, *, stream: bool, on_delta, token_limit: int):
        stream = CHAT_STREAMING_ENABLED
        payload = {
            "model": api_model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(_token_limit_field(api_model, token_limit))
        payload.update(_sampling_field(api_model))

        session = await self._ensure_session()
        async with session.post(GPT_URL, json=payload) as resp:
            if resp.status != 200:
                try:
                    error_payload = await resp.json()
                except Exception:
                    error_payload = await resp.text()
                return None, None, f"API error {resp.status}: {error_payload}"

            if stream:
                content, finish_reason = await self._read_stream_response(resp, on_delta)
                return content, finish_reason, None

            result = await resp.json()
            if "error" in result:
                message = result["error"].get("message")
                return None, None, f"API error: {message or result['error']}"

            choice = result.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content") or ""
            finish_reason = choice.get("finish_reason")
            return content, finish_reason, None

    async def _continue_response(self, messages, api_model: str, partial_content: str, *, stream: bool, on_delta, token_limit: int) -> str:
        accumulated = partial_content
        for _ in range(self._MAX_CONTINUATIONS):
            continuation_messages = messages + [
                {"role": "assistant", "content": accumulated},
                {
                    "role": "user",
                    "content": "Continue exactly from where you stopped. Do not repeat earlier text. Finish any open code blocks, strings, lists, or sentences. Return only the remaining text.",
                },
            ]
            extra_content, finish_reason, error = await self._execute_chat_request(
                continuation_messages,
                api_model,
                stream=stream,
                on_delta=on_delta,
                token_limit=token_limit,
            )
            if error or not extra_content:
                break
            accumulated += extra_content
            if finish_reason != "length":
                break
        return accumulated

    async def generateResponse(self, userInput: str, selectedModel: str, *, stream: bool = True, on_delta=None, include_context: bool = True, token_limit: int | None = None) -> str | None:
        stream = CHAT_STREAMING_ENABLED
        generic_answer = _match_generic_question(userInput)
        if generic_answer:
            if on_delta:
                on_delta(generic_answer)
            self.db.save_message("user", userInput)
            self.db.save_message("assistant", generic_answer)
            return generic_answer

        cache_key = f"{selectedModel}::{userInput}"
        use_cache = not stream
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        api_model = _resolve_model(selectedModel)
        messages = [{"role": "system", "content": BASE_SYSTEM_MESSAGE}]
        messages.extend(_instruction_messages(selectedModel))
        if include_context:
            messages.extend(self._get_context(3))
        messages.append({"role": "user", "content": _normalize_user_input(userInput)})

        resolved_token_limit = token_limit if token_limit is not None else _resolve_token_limit(api_model)
        content, finish_reason, error = await self._execute_chat_request(
            messages,
            api_model,
            stream=stream,
            on_delta=on_delta,
            token_limit=resolved_token_limit,
        )
        if error:
            return error

        content = content or ""
        if content and (finish_reason == "length" or self._looks_incomplete_response(content)):
            content = await self._continue_response(
                messages,
                api_model,
                content,
                stream=stream,
                on_delta=on_delta,
                token_limit=resolved_token_limit,
            )

        if content:
            self.db.save_message("user", userInput)
            self.db.save_message("assistant", content)
            if use_cache:
                self._cache_set(cache_key, content)
            return content

        return "API returned no visible content."

    async def ask_gpt(self, text: str, mode: str = "chat", stream: bool = True, on_delta=None, fast: bool = False, selected_model: str = GPT4O_MODEL, include_context: bool | None = None) -> str | None:
        stream = CHAT_STREAMING_ENABLED
        if mode == "hint":
            prompt = _hint_prompt(text)
            messages = [
                {"role": "system", "content": _model_prompt(GPT_HINT_MODEL)},
                {"role": "user", "content": prompt},
            ]
            payload = {
                "model": GPT_HINT_MODEL,
                "messages": messages,
                "stream": stream,
            }
            payload.update(_token_limit_field(GPT_HINT_MODEL, HINT_MAX_TOKENS))
            payload.update(_sampling_field(GPT_HINT_MODEL))

            session = await self._ensure_session()
            async with session.post(GPT_URL, json=payload) as resp:
                if resp.status != 200:
                    try:
                        error_payload = await resp.json()
                    except Exception:
                        error_payload = await resp.text()
                    return f"API error {resp.status}: {error_payload}"

                if stream:
                    content, finish_reason = await self._read_stream_response(resp, on_delta)
                    if content and (finish_reason == "length" or self._looks_incomplete_response(content)):
                        content = await self._continue_response(
                            messages,
                            GPT_HINT_MODEL,
                            content,
                            stream=stream,
                            on_delta=on_delta,
                            token_limit=HINT_MAX_TOKENS,
                        )
                    return content

                result = await resp.json()
                if "error" in result:
                    message = result["error"].get("message")
                    return f"API error: {message or result['error']}"

                choice = result.get("choices", [{}])[0]
                return choice.get("message", {}).get("content")

        api_model = _resolve_model(selected_model)
        if selected_model == GPT5_MODEL and fast:
            token_limit = GPT5_FAST_MAX_TOKENS
        else:
            token_limit = _resolve_token_limit(api_model)
        if include_context is None:
            include_context = mode == "chat"
        return await self.generateResponse(text, selected_model, stream=stream, on_delta=on_delta, include_context=include_context, token_limit=token_limit)

    async def _read_stream_response(self, resp, on_delta):
        content = ""
        finish_reason = None
        while True:
            line_bytes = await resp.content.readline()
            if not line_bytes:
                break

            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data:"):
                continue

            payload = json.loads(line[len("data:"):].strip())
            if "error" in payload:
                message = payload["error"].get("message")
                return f"API error: {message or payload['error']}", "error"

            choices = payload.get("choices")
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta", {}).get("content", "")
            if delta:
                content += delta
                if on_delta:
                    on_delta(delta)

        return content, finish_reason

    async def stream_gpt(self, text: str):
        """Async generator that yields partial response tokens as they arrive.

        Usage:
            async for token in bridge.stream_gpt("your question"):
                print(token, end="", flush=True)
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _on_delta(delta: str) -> None:
            queue.put_nowait(delta)

        async def _run_request() -> None:
            try:
                await self.generateResponse(text, GPT4O_MODEL, stream=True, on_delta=_on_delta, include_context=False)
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(_run_request())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
            await task
        finally:
            if not task.done():
                task.cancel()

    async def process(self, wav_path: str) -> tuple[str | None, str | None]:
        transcript = await self.transcribe(wav_path)
        if not transcript:
            return None, None

        reply = await self.ask_gpt(transcript, mode="hint")
        return transcript, reply