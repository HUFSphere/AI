import json
import os
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, field_validator

load_dotenv()

app = FastAPI(title="Onboarding RAG QnA (AI part)")

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
ASK_TOP_K = 5

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY=sk-... 를 추가하세요."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def _not_blank(v: str) -> str:
    if not v or not v.strip():
        raise ValueError("must not be blank")
    return v


class Chunk(BaseModel):
    source_type: str
    item_type: str
    title: str
    url: str
    text: str


class SourceMeta(BaseModel):
    source_type: str
    item_type: str
    title: str
    url: str


class QnaResponse(BaseModel):
    answer: str
    sources: list[SourceMeta]


class QnaRequest(BaseModel):
    question: str
    lang: str
    chunks: list[Chunk] = Field(default_factory=list)

    @field_validator("question", "lang")
    @classmethod
    def not_blank(cls, v: str) -> str:
        return _not_blank(v)


# ---- in-memory vector store -------------------------------------------------
# 나중에 FAISS/Chroma 등 파일 저장 방식으로 교체할 때는 이 섹션(저장·검색 함수)만
# 갈아끼우면 된다. /index, /search, /ask, /qna 쪽 코드는 그대로 둔다.

_store: list[tuple[Chunk, np.ndarray]] = []


def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        resp = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    except (OpenAIError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=f"임베딩 생성 실패: {e}") from e
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]


def add_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    embeddings = embed_texts([c.text for c in chunks])
    for chunk, emb in zip(chunks, embeddings):
        _store.append((chunk, np.array(emb, dtype=np.float32)))
    return len(chunks)


def search_chunks(question: str, top_k: int) -> list[tuple[float, Chunk]]:
    if not _store:
        return []
    query = np.array(embed_texts([question])[0], dtype=np.float32)
    matrix = np.stack([emb for _, emb in _store])

    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    scores = matrix_norm @ query_norm

    order = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), _store[i][0]) for i in order]


# ---- shared answer generation (used by /qna and /ask) ----------------------

SYSTEM_PROMPT_TEMPLATE = """당신은 GitHub, Notion, Figma 등 협업 기록을 근거로 신규 합류자의 질문에 답하는 어시스턴트입니다.

반드시 지켜야 할 규칙:
1. 아래 제공된 근거(chunks) 안에 있는 내용만 사용해서 답하세요. 근거에 없는 내용은 추측하거나 지어내지 마세요.
2. 근거만으로 답을 찾을 수 없다면 "관련 기록을 찾지 못했습니다"에 해당하는 취지의 답변을 하세요.
3. 답변 언어: ISO 639-1 코드 "{lang}"에 해당하는 언어로만 답변을 작성하세요. 질문이나 근거가 다른 언어(예: 한국어)로 되어 있어도, 최종 answer 필드는 반드시 "{lang}" 언어로 번역해서 작성해야 합니다. 예: lang="vi"면 베트남어로만, lang="en"이면 영어로만, lang="ko"면 한국어로만 작성합니다. 다른 언어를 절대 섞지 마세요.
4. 답변을 작성하는 데 실제로 사용한 근거의 번호를 used_indices 배열에 넣으세요. 사용하지 않은 근거는 넣지 마세요. 근거를 전혀 사용하지 못했다면 빈 배열을 반환하세요.

반드시 지정된 JSON 스키마로만 응답하세요."""

RESPONSE_JSON_SCHEMA = {
    "name": "qna_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "used_indices": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["answer", "used_indices"],
        "additionalProperties": False,
    },
}


def build_user_prompt(question: str, chunks: list[Chunk], lang: str) -> str:
    lines = [f"질문: {question}", f"(answer 필드는 반드시 언어 코드 '{lang}'로만 작성)", "", "근거 목록:"]
    if not chunks:
        lines.append("(제공된 근거 없음)")
    for i, c in enumerate(chunks):
        lines.append(
            f"[{i}] source_type={c.source_type} item_type={c.item_type} "
            f"title={c.title}\n{c.text}"
        )
    return "\n".join(lines)


def generate_answer(question: str, lang: str, chunks: list[Chunk]) -> QnaResponse:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(lang=lang)
    user_prompt = build_user_prompt(question, chunks, lang)

    try:
        completion = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": RESPONSE_JSON_SCHEMA},
        )
        raw = completion.choices[0].message.content
        parsed = json.loads(raw)
    except (OpenAIError, RuntimeError, json.JSONDecodeError, KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {e}") from e

    answer = parsed["answer"]
    used_indices = parsed.get("used_indices", [])

    sources: list[SourceMeta] = []
    seen: set[int] = set()
    for idx in used_indices:
        if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            c = chunks[idx]
            sources.append(
                SourceMeta(
                    source_type=c.source_type,
                    item_type=c.item_type,
                    title=c.title,
                    url=c.url,
                )
            )

    return QnaResponse(answer=answer, sources=sources)


@app.post("/qna", response_model=QnaResponse)
def qna(req: QnaRequest) -> QnaResponse:
    return generate_answer(req.question, req.lang, req.chunks)


# ---- indexing / search / ask endpoints --------------------------------------


class IndexRequest(BaseModel):
    chunks: list[Chunk] = Field(default_factory=list)


class IndexResponse(BaseModel):
    indexed: int


@app.post("/index", response_model=IndexResponse)
def index(req: IndexRequest) -> IndexResponse:
    count = add_chunks(req.chunks)
    return IndexResponse(indexed=count)


class SearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        return _not_blank(v)


class SearchResult(BaseModel):
    source_type: str
    item_type: str
    title: str
    url: str
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    ranked = search_chunks(req.question, req.top_k)
    results = [
        SearchResult(
            source_type=c.source_type,
            item_type=c.item_type,
            title=c.title,
            url=c.url,
            text=c.text,
            score=score,
        )
        for score, c in ranked
    ]
    return SearchResponse(results=results)


class AskRequest(BaseModel):
    question: str
    lang: str

    @field_validator("question", "lang")
    @classmethod
    def not_blank(cls, v: str) -> str:
        return _not_blank(v)


@app.post("/ask", response_model=QnaResponse)
def ask(req: AskRequest) -> QnaResponse:
    ranked = search_chunks(req.question, ASK_TOP_K)
    chunks = [c for _, c in ranked]
    return generate_answer(req.question, req.lang, chunks)


# ---- suggested questions -----------------------------------------------------

SUGGEST_CHUNK_LIMIT = 20
SUGGEST_TEXT_CHARS = 300

SUGGEST_SYSTEM_PROMPT_TEMPLATE = """당신은 GitHub, Notion, Figma 등 협업 기록을 바탕으로, 프로젝트에 새로 합류한 개발자가 팀의 맥락을 파악하기 위해 물어볼 만한 질문을 추천하는 어시스턴트입니다.

반드시 지켜야 할 규칙:
1. 아래 제공된 협업 기록 샘플에 실제로 등장하는 내용에 근거해서 질문을 만드세요. 기록에 없는 내용을 지어내지 마세요.
2. 질문은 단순 사실 나열이 아니라, 팀의 결정이나 맥락을 묻는 "왜/어떻게" 중심의 형태여야 합니다.
3. 제공된 기록이 비어 있다면, 신규 합류자가 일반적으로 물어볼 법한 온보딩 질문 3개를 대신 생성하세요.
4. 정확히 3개의 질문만 반환하세요.
5. 반드시 ISO 639-1 언어 코드 "{lang}"에 해당하는 언어로만 질문을 작성하세요. 자료가 다른 언어로 되어 있어도 "{lang}" 언어로 번역해서 작성해야 합니다. 다른 언어를 절대 섞지 마세요.

반드시 지정된 JSON 스키마로만 응답하세요."""

SUGGEST_RESPONSE_JSON_SCHEMA = {
    "name": "suggested_questions",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}


def build_suggestion_material(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(저장된 협업 기록 없음)"
    lines = ["저장된 협업 기록 샘플:"]
    for i, c in enumerate(chunks):
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > SUGGEST_TEXT_CHARS:
            snippet = snippet[:SUGGEST_TEXT_CHARS] + "..."
        lines.append(
            f"[{i}] source_type={c.source_type} item_type={c.item_type} title={c.title}\n{snippet}"
        )
    return "\n".join(lines)


def generate_suggested_questions(lang: str) -> list[str]:
    sample = [c for c, _ in _store[:SUGGEST_CHUNK_LIMIT]]
    material = build_suggestion_material(sample)
    system_prompt = SUGGEST_SYSTEM_PROMPT_TEMPLATE.format(lang=lang)

    try:
        completion = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": material},
            ],
            response_format={"type": "json_schema", "json_schema": SUGGEST_RESPONSE_JSON_SCHEMA},
        )
        raw = completion.choices[0].message.content
        parsed = json.loads(raw)
    except (OpenAIError, RuntimeError, json.JSONDecodeError, KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {e}") from e

    questions = parsed.get("questions", [])
    return [q for q in questions if isinstance(q, str) and q.strip()][:3]


class SuggestQuestionsRequest(BaseModel):
    lang: str

    @field_validator("lang")
    @classmethod
    def not_blank(cls, v: str) -> str:
        return _not_blank(v)


class SuggestQuestionsResponse(BaseModel):
    questions: list[str]


@app.post("/suggest-questions", response_model=SuggestQuestionsResponse)
def suggest_questions(req: SuggestQuestionsRequest) -> SuggestQuestionsResponse:
    questions = generate_suggested_questions(req.lang)
    return SuggestQuestionsResponse(questions=questions)


# ---- work item extraction ----------------------------------------------------

EXTRACT_CHUNK_LIMIT = 100
EXTRACT_TEXT_CHARS = 1500

WORK_ITEM_STATUSES = ["todo", "in_progress", "review", "done", "blocked"]

EXTRACT_SYSTEM_PROMPT_TEMPLATE = """당신은 GitHub, Notion, Figma 등 협업 기록 조각(chunk)을 하나의 "작업(work item)"으로 구조화하는 어시스턴트입니다.

아래 번호가 매겨진 chunk 목록이 주어집니다. 각 chunk마다 정확히 하나의 결과 항목을 만들어 반환하세요.

반드시 지켜야 할 규칙:
1. 각 chunk의 실제 내용만 근거로 판단하세요. chunk에 없는 내용을 지어내지 마세요.
2. summary_brief: 해당 chunk가 무엇에 관한 것인지 1~2문장으로 간결하게 요약하세요.
3. status는 아래 다섯 값 중 하나로 판정하세요.
   - todo: 아직 시작하지 않았거나 상태를 알 수 없음(기본값)
   - in_progress: 진행 중이라는 근거가 있음
   - review: 리뷰/검토 중이라는 근거가 있음
   - done: 완료/병합/해결되었다는 근거가 있음
   - blocked: 막혀 있다는 근거가 있음
   명확한 근거가 없으면 반드시 todo로 판정하세요.
4. 반드시 ISO 639-1 언어 코드 "{lang}"에 해당하는 언어로 summary_brief를 작성하세요. chunk 원문이 다른 언어여도 "{lang}" 언어로 번역해서 작성해야 합니다.
5. 입력으로 주어진 chunk 각각에 대해, 그 chunk의 index를 그대로 결과의 index 필드에 넣으세요. 순서를 바꾸거나 누락하지 마세요.

반드시 지정된 JSON 스키마로만 응답하세요."""


def build_extract_material(chunks: list[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks):
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > EXTRACT_TEXT_CHARS:
            snippet = snippet[:EXTRACT_TEXT_CHARS] + "..."
        lines.append(
            f"[{i}] source_type={c.source_type} item_type={c.item_type} title={c.title}\n{snippet}"
        )
    return "\n".join(lines)


def build_extract_json_schema(n: int) -> dict:
    return {
        "name": "work_item_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": n,
                    "maxItems": n,
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "summary_brief": {"type": "string"},
                            "status": {"type": "string", "enum": WORK_ITEM_STATUSES},
                        },
                        "required": ["index", "summary_brief", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


class WorkItem(BaseModel):
    source_type: str = Field(alias="sourceType")
    item_type: str = Field(alias="itemType")
    title: str
    summary_brief: str = Field(alias="summaryBrief")
    status: str
    url: str

    model_config = {"populate_by_name": True}


def generate_work_items(lang: str) -> list[WorkItem]:
    chunks = [c for c, _ in _store[:EXTRACT_CHUNK_LIMIT]]
    if not chunks:
        return []

    material = build_extract_material(chunks)
    system_prompt = EXTRACT_SYSTEM_PROMPT_TEMPLATE.format(lang=lang)
    json_schema = build_extract_json_schema(len(chunks))

    try:
        completion = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": material},
            ],
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
        raw = completion.choices[0].message.content
        parsed = json.loads(raw)
    except (OpenAIError, RuntimeError, json.JSONDecodeError, KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {e}") from e

    by_index: dict[int, dict] = {}
    for item in parsed.get("items", []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in by_index:
            by_index[idx] = item

    work_items: list[WorkItem] = []
    for i, c in enumerate(chunks):
        item = by_index.get(i)
        status = item["status"] if item and item.get("status") in WORK_ITEM_STATUSES else "todo"
        summary_brief = (
            item["summary_brief"]
            if item and isinstance(item.get("summary_brief"), str) and item["summary_brief"].strip()
            else ""
        )
        work_items.append(
            WorkItem(
                sourceType=c.source_type,
                itemType=c.item_type,
                title=c.title,
                summaryBrief=summary_brief,
                status=status,
                url=c.url,
            )
        )
    return work_items


class ExtractWorkItemsRequest(BaseModel):
    lang: str

    @field_validator("lang")
    @classmethod
    def not_blank(cls, v: str) -> str:
        return _not_blank(v)


class ExtractWorkItemsResponse(BaseModel):
    work_items: list[WorkItem] = Field(alias="workItems")

    model_config = {"populate_by_name": True}


@app.post("/extract-work-items", response_model=ExtractWorkItemsResponse)
def extract_work_items(req: ExtractWorkItemsRequest) -> ExtractWorkItemsResponse:
    items = generate_work_items(req.lang)
    return ExtractWorkItemsResponse(workItems=items)


# ---- work item linking --------------------------------------------------------

LINK_CHUNK_LIMIT = 60
LINK_TEXT_CHARS = 300
LINK_SIMILARITY_THRESHOLD = 0.4

LINK_SYSTEM_PROMPT_TEMPLATE = """당신은 GitHub, Notion, Figma 등 협업 기록으로 구조화된 여러 "작업(work item)" 사이의 의미적 연관 관계를 판단하는 어시스턴트입니다.

아래에 각 작업(from)과, 임베딩 유사도로 미리 선별된 연결 후보(candidates) 목록이 주어집니다.

반드시 지켜야 할 규칙:
1. 후보로 주어졌다고 해서 무조건 연결하지 마세요. 두 작업의 실제 내용을 비교해서 실제로 관련이 있을 때만 연결하세요.
2. 관련이 약하거나 근거를 댈 수 없으면 해당 후보는 결과에서 제외하세요(빈 배열 허용). 억지로 관계를 지어내지 마세요.
3. 연결로 판단한 각 후보에 대해, 왜 관련이 있는지 두 작업의 실제 내용에 근거해서 1문장으로 설명(link_reason)하세요.
4. 반드시 ISO 639-1 언어 코드 "{lang}"에 해당하는 언어로 link_reason을 작성하세요. 원문이 다른 언어여도 "{lang}" 언어로 번역해서 작성해야 합니다.
5. 각 작업(from_index)에 대해, 주어진 후보 목록에 있는 to_index만 사용하세요. 후보에 없는 index를 만들어내지 마세요.

반드시 지정된 JSON 스키마로만 응답하세요."""


def compute_link_candidates(chunks: list[Chunk], embeddings: list[np.ndarray], top_k: int) -> dict[int, list[tuple[int, float]]]:
    if len(chunks) < 2:
        return {}
    matrix = np.stack(embeddings)
    norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    sims = norm @ norm.T
    np.fill_diagonal(sims, -1.0)

    candidates: dict[int, list[tuple[int, float]]] = {}
    for i in range(len(chunks)):
        order = np.argsort(-sims[i])[:top_k]
        candidates[i] = [(int(j), float(sims[i][j])) for j in order if sims[i][j] >= LINK_SIMILARITY_THRESHOLD]
    return candidates


def _link_snippet(c: Chunk) -> str:
    s = c.text.strip().replace("\n", " ")
    return s[:LINK_TEXT_CHARS] + ("..." if len(s) > LINK_TEXT_CHARS else "")


def build_link_material(chunks: list[Chunk], items_with_candidates: dict[int, list[tuple[int, float]]]) -> str:
    lines = []
    for i in sorted(items_with_candidates.keys()):
        c = chunks[i]
        lines.append(f"### 작업 [{i}] source_type={c.source_type} item_type={c.item_type} title={c.title}\n{_link_snippet(c)}")
        lines.append("후보:")
        for j, _score in items_with_candidates[i]:
            cc = chunks[j]
            lines.append(
                f"  - [{j}] source_type={cc.source_type} item_type={cc.item_type} title={cc.title}\n    {_link_snippet(cc)}"
            )
        lines.append("")
    return "\n".join(lines)


def build_link_json_schema(n: int) -> dict:
    return {
        "name": "work_item_links",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": n,
                    "maxItems": n,
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_index": {"type": "integer"},
                            "linked": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "to_index": {"type": "integer"},
                                        "link_reason": {"type": "string"},
                                    },
                                    "required": ["to_index", "link_reason"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["from_index", "linked"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


class LinkedItem(BaseModel):
    to_index: int = Field(alias="toIndex")
    to_title: str = Field(alias="toTitle")
    to_source_type: str = Field(alias="toSourceType")
    to_url: str = Field(alias="toUrl")
    link_reason: str = Field(alias="linkReason")
    score: float

    model_config = {"populate_by_name": True}


class WorkItemLinks(BaseModel):
    from_index: int = Field(alias="fromIndex")
    from_title: str = Field(alias="fromTitle")
    linked_items: list[LinkedItem] = Field(alias="linkedItems")

    model_config = {"populate_by_name": True}


def generate_work_item_links(lang: str, top_k: int) -> list[WorkItemLinks]:
    store_slice = _store[:LINK_CHUNK_LIMIT]
    if len(store_slice) < 2:
        return []

    chunks = [c for c, _ in store_slice]
    embeddings = [emb for _, emb in store_slice]
    candidates = compute_link_candidates(chunks, embeddings, top_k)

    items_with_candidates = {i: cands for i, cands in candidates.items() if cands}
    if not items_with_candidates:
        return [
            WorkItemLinks(fromIndex=i, fromTitle=c.title, linkedItems=[])
            for i, c in enumerate(chunks)
        ]

    material = build_link_material(chunks, items_with_candidates)
    system_prompt = LINK_SYSTEM_PROMPT_TEMPLATE.format(lang=lang)
    json_schema = build_link_json_schema(len(items_with_candidates))

    try:
        completion = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": material},
            ],
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
        raw = completion.choices[0].message.content
        parsed = json.loads(raw)
    except (OpenAIError, RuntimeError, json.JSONDecodeError, KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"OpenAI 호출 실패: {e}") from e

    reasons: dict[tuple[int, int], str] = {}
    for entry in parsed.get("items", []):
        from_idx = entry.get("from_index")
        if not isinstance(from_idx, int) or from_idx not in items_with_candidates:
            continue
        valid_to = {j for j, _ in items_with_candidates[from_idx]}
        for link in entry.get("linked", []):
            to_idx = link.get("to_index")
            reason = link.get("link_reason")
            if isinstance(to_idx, int) and to_idx in valid_to and isinstance(reason, str) and reason.strip():
                reasons[(from_idx, to_idx)] = reason

    results: list[WorkItemLinks] = []
    for i, c in enumerate(chunks):
        linked_items: list[LinkedItem] = []
        for j, score in candidates.get(i, []):
            reason = reasons.get((i, j))
            if reason is None:
                continue
            target = chunks[j]
            linked_items.append(
                LinkedItem(
                    toIndex=j,
                    toTitle=target.title,
                    toSourceType=target.source_type,
                    toUrl=target.url,
                    linkReason=reason,
                    score=score,
                )
            )
        results.append(WorkItemLinks(fromIndex=i, fromTitle=c.title, linkedItems=linked_items))
    return results


class LinkWorkItemsRequest(BaseModel):
    lang: str
    top_k: int = Field(default=4, ge=1)

    @field_validator("lang")
    @classmethod
    def not_blank(cls, v: str) -> str:
        return _not_blank(v)


class LinkWorkItemsResponse(BaseModel):
    links: list[WorkItemLinks]


@app.post("/link-work-items", response_model=LinkWorkItemsResponse)
def link_work_items(req: LinkWorkItemsRequest) -> LinkWorkItemsResponse:
    links = generate_work_item_links(req.lang, req.top_k)
    return LinkWorkItemsResponse(links=links)


# ---- GitHub PR ingestion -----------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"


def _github_headers(access_token: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "hufsphere-ai-ingest",
    }
    token = access_token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_get(url: str, params: dict | None = None, access_token: str | None = None) -> list | dict:
    try:
        resp = httpx.get(url, headers=_github_headers(access_token), params=params, timeout=15.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"GitHub API 호출 실패: {e}") from e

    if resp.status_code >= 400:
        try:
            message = resp.json().get("message", resp.text)
        except json.JSONDecodeError:
            message = resp.text
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API 오류 ({resp.status_code}): {message}",
        )
    return resp.json()


def fetch_recent_pull_requests(
    owner: str, name: str, months: int, access_token: str | None = None
) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)
    pulls: list[dict] = []
    page = 1
    while True:
        data = _github_get(
            f"{GITHUB_API_BASE}/repos/{owner}/{name}/pulls",
            params={
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
            access_token=access_token,
        )
        if not data:
            break

        reached_cutoff = False
        for pr in data:
            updated_at = datetime.fromisoformat(pr["updated_at"].replace("Z", "+00:00"))
            if updated_at < cutoff:
                reached_cutoff = True
                break
            pulls.append(pr)

        if reached_cutoff or len(data) < 100:
            break
        page += 1

    return pulls


def fetch_pr_comments_text(
    owner: str, name: str, pr_number: int, access_token: str | None = None
) -> str:
    texts: list[str] = []

    review_comments = _github_get(
        f"{GITHUB_API_BASE}/repos/{owner}/{name}/pulls/{pr_number}/comments",
        access_token=access_token,
    )
    for c in review_comments:
        body = (c.get("body") or "").strip()
        if body:
            texts.append(body)

    issue_comments = _github_get(
        f"{GITHUB_API_BASE}/repos/{owner}/{name}/issues/{pr_number}/comments",
        access_token=access_token,
    )
    for c in issue_comments:
        body = (c.get("body") or "").strip()
        if body:
            texts.append(body)

    return "\n".join(texts)


def pr_to_chunk(owner: str, name: str, pr: dict, access_token: str | None = None) -> Chunk:
    title = pr.get("title") or "(제목 없음)"
    body = (pr.get("body") or "").strip()
    comments_text = fetch_pr_comments_text(owner, name, pr["number"], access_token)

    parts = [title]
    if body:
        parts.append(body)
    if comments_text:
        parts.append(comments_text)

    return Chunk(
        source_type="github",
        item_type="pr",
        title=title,
        url=pr["html_url"],
        text="\n\n".join(parts),
    )


class IngestGithubRequest(BaseModel):
    repo: str
    months: int = Field(default=3, ge=1)
    access_token: str | None = None

    @field_validator("repo")
    @classmethod
    def valid_repo(cls, v: str) -> str:
        v = _not_blank(v)
        parts = v.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("repo must be in 'owner/name' format")
        return v


class IngestGithubResponse(BaseModel):
    repo: str
    indexed: int


@app.post("/ingest/github", response_model=IngestGithubResponse)
def ingest_github(req: IngestGithubRequest) -> IngestGithubResponse:
    owner, name = req.repo.split("/")
    prs = fetch_recent_pull_requests(owner, name, req.months, req.access_token)
    chunks = [pr_to_chunk(owner, name, pr, req.access_token) for pr in prs]
    count = add_chunks(chunks)
    return IngestGithubResponse(repo=req.repo, indexed=count)


# ---- Notion page ingestion ---------------------------------------------------


class NotionPage(BaseModel):
    title: str
    url: str
    text: str = ""
    item_type: str = "meeting"

    @field_validator("title")
    @classmethod
    def not_blank(cls, v: str) -> str:
        return _not_blank(v)


def notion_page_to_chunk(page: NotionPage) -> Chunk:
    parts = [page.title]
    if page.text.strip():
        parts.append(page.text)

    return Chunk(
        source_type="notion",
        item_type=page.item_type,
        title=page.title,
        url=page.url,
        text="\n\n".join(parts),
    )


class IngestNotionRequest(BaseModel):
    pages: list[NotionPage] = Field(default_factory=list)


class IngestSourceResponse(BaseModel):
    source: str
    indexed: int


@app.post("/ingest/notion", response_model=IngestSourceResponse)
def ingest_notion(req: IngestNotionRequest) -> IngestSourceResponse:
    chunks = [notion_page_to_chunk(p) for p in req.pages]
    count = add_chunks(chunks)
    return IngestSourceResponse(source="notion", indexed=count)


# ---- Figma comment ingestion --------------------------------------------------


class FigmaComment(BaseModel):
    frameName: str
    url: str
    text: str = ""

    @field_validator("frameName")
    @classmethod
    def not_blank(cls, v: str) -> str:
        return _not_blank(v)


def figma_comment_to_chunk(comment: FigmaComment) -> Chunk:
    parts = [comment.frameName]
    if comment.text.strip():
        parts.append(comment.text)

    return Chunk(
        source_type="figma",
        item_type="design",
        title=comment.frameName,
        url=comment.url,
        text="\n\n".join(parts),
    )


class IngestFigmaRequest(BaseModel):
    comments: list[FigmaComment] = Field(default_factory=list)


@app.post("/ingest/figma", response_model=IngestSourceResponse)
def ingest_figma(req: IngestFigmaRequest) -> IngestSourceResponse:
    chunks = [figma_comment_to_chunk(c) for c in req.comments]
    count = add_chunks(chunks)
    return IngestSourceResponse(source="figma", indexed=count)
