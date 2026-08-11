# Onboarding RAG QnA — AI part (FastAPI)

> ⚠️ **`OPENAI_API_KEY`는 절대 이 레포에 올리지 않는다.** `.env` 파일, 커밋, PR, 이슈, 스크린샷 등
> 어떤 형태로도 커밋·푸시·업로드 금지. 키는 [BE](https://github.com/HUFSphere/BE)와 별개로
> 팀장이 개별적으로(카톡/슬랙 DM 등) 전달한다. 새로 받은 키는 로컬 `.env`에만 적어두면 되고,
> `.env`는 이미 `.gitignore`에 포함되어 있어 `git add`해도 커밋되지 않는다(아래 "실행 방법" 참고).

GitHub·Notion·Figma 협업 기록(chunk)을 임베딩해 인메모리에 저장하고, 질문이 오면
관련 chunk를 코사인 유사도로 검색해 `gpt-4o-mini`로 근거 기반 답변을 생성한다.
답변은 제공된 근거 안에서만 작성되며, 근거가 없으면 모른다고 답한다(할루시네이션 방지).

## BE 팀원 필독 — 이 레포(AI)도 같이 클론해야 하는 이유

이 프로젝트는 [BE](https://github.com/HUFSphere/BE)(Spring Boot, linkboard)와 이 AI(FastAPI) 레포,
총 두 개의 저장소로 나뉘어 있다. BE만 클론해서 실행하면 "source 연동 → QnA" 흐름이 전부 502/타임아웃으로
막힌다 — BE가 소스 동기화·QnA 요청 시 내부적으로 이 AI 서버(`http://localhost:8000`)를 호출하기 때문이다.
즉 BE 혼자서는 절반짜리 프로젝트만 돌아간다. **BE와 AI 두 레포를 각각 클론해서 로컬에서 동시에 띄워야
전체 흐름을 테스트할 수 있다.**

```
어딘가/
├─ BE/   (git clone https://github.com/HUFSphere/BE.git)
└─ AI/   (git clone https://github.com/HUFSphere/AI.git)  ← 이 레포
```

두 서버는 포트가 겹치지 않는다 — BE는 `:8080`, AI는 `:8000`. BE가 AI 서버 주소를 바라보는 설정값
(예: `application.yml`의 AI 서버 base URL)이 `http://localhost:8000`을 가리키고 있는지 확인하고,
아래 "실행 방법"대로 AI 서버를 먼저(또는 BE와 상관없이 아무 순서로나) `:8000`에 띄워두면 BE가 정상적으로
호출할 수 있다. 두 서버를 각각 다른 터미널 창에서 띄워두면 된다 — 하나가 다른 하나를 실행시켜주지 않는다.

- 임베딩 모델: `text-embedding-3-small`
- 생성 모델: `gpt-4o-mini`
- 저장소: 파이썬 프로세스 메모리(list) — **서버를 재시작하면 인덱스가 사라진다.** 지금은 데모/개발 단계라 의도된 동작이며,
  나중에 FAISS/Chroma 같은 파일 기반 저장소로 교체할 때는 `main.py`의 "in-memory vector store" 섹션
  (`add_chunks`, `search_chunks`, `embed_texts`)만 갈아끼우면 된다. 그 외 엔드포인트는 그대로 둔다.

## 엔드포인트

| | 메서드/경로 | 역할 |
|---|---|---|
| 1 | `POST /index` | chunk들을 임베딩해서 메모리에 저장 |
| 2 | `POST /search` | 질문과 가장 관련 있는 저장된 chunk를 top_k개 검색 (점수 포함) |
| 3 | `POST /ask` | question·lang만 받아 내부에서 검색 + 답변 생성까지 한 번에 |
| 4 | `POST /qna` | chunk를 직접 받아 답변만 생성 (기존 구현, 검색 없음) |
| 5 | `POST /ingest/github` | 공개 GitHub 레포의 최근 PR을 긁어와 chunk로 만들어 `/index`와 동일하게 저장 |
| 6 | `POST /ingest/notion` | Notion 페이지 목록을 받아 chunk로 만들어 `/index`와 동일하게 저장 |
| 7 | `POST /ingest/figma` | Figma 코멘트 목록을 받아 chunk로 만들어 `/index`와 동일하게 저장 |

`/ask`와 `/qna`는 같은 답변 생성 로직(`generate_answer`)을 공유한다 — 근거 안에서만 답하고,
없으면 모른다고 `lang`으로 답하며, 실제로 사용한 근거만 `sources`로 반환하는 규칙은 동일하다.

> **AI 서버 구현 vs BE 연동 상태.** `/ingest/notion`, `/ingest/figma`는 이 AI 서버에는 이미 구현되어
> 있다(4~7행). 다만 [BE](https://github.com/HUFSphere/BE)가 아직 이 두 엔드포인트를 호출하도록
> 연동되어 있지 않아서, BE에 Notion/Figma를 소스로 등록하면 BE가 400을 반환한다 — 이건 BE 쪽 미연동
> 문제이지 AI 서버가 안 만들어진 게 아니다. GitHub 파이프라인(`/ingest/github`)만 BE↔AI 전체 흐름이
> 실제로 연동·검증되어 있다(`pypa/sampleproject`로 2026-08-11 e2e 테스트 통과 — 아래 "테스트 흐름:
> GitHub 레포 자동 수집 → ask" 참고).

## 실행 방법

> 이 PC의 anaconda3 Python은 `DLLs\_socket.pyd`가 없어(설치 손상) `pip`조차 실행되지 않는다.
> `anaconda3\python.exe` 대신 다른 정상 Python(예: `AppData\Local\Programs\Python\Python313\python.exe`)으로
> 프로젝트 전용 venv를 만들어 쓴다. anaconda를 계속 쓰고 싶으면 Anaconda를 Repair/재설치해야 한다.

`OPENAI_API_KEY`는 프로젝트 루트의 `.env` 파일에 적어두면 `main.py`가 시작할 때
`python-dotenv`로 자동으로 읽는다. `.env`는 `.gitignore`에 들어있어 `git add .`를 해도 커밋되지 않지만,
**절대 강제로(`git add -f`) 커밋하거나 다른 방식(Slack 코드블록, 스크린샷, PR 설명 등)으로도 공유하지 말 것.**
키는 팀장이 개별적으로 전달하며, 레포에는 `.env.example`처럼 값이 비어있는 템플릿만 존재해야 한다.

`GITHUB_TOKEN`(선택)도 같은 `.env`에 넣어두면 `/ingest/github`가 인증된 요청을 보낸다 —
비공개 레포를 볼 필요는 없지만, GitHub 비인증 요청은 시간당 60회로 제한되므로 여러 레포/큰 레포를
반복 수집할 계획이면 [Personal Access Token](https://github.com/settings/tokens)(별도 권한 없이 `public_repo` 정도로 충분)을
넣어두는 걸 권장한다. 없어도 동작은 한다.

```powershell
# Windows PowerShell (프로젝트 루트: F:\HUFSphere\AI)
& "C:\Users\pjyng\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Copy-Item .env.example .env
# .env 파일을 열어 OPENAI_API_KEY=sk-... 실제 키로 채워넣기

.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

```bash
# bash (git bash 등)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY=sk-... 실제 키로 채워넣기

./.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
```

`.env`가 없거나 `OPENAI_API_KEY`가 비어 있으면 호출 시 502로 명확한 안내 메시지를 반환한다.

`requirements.txt`의 `openai>=2.53.0`은 필수 하한이다: `openai==1.51.0` + 최신 `httpx`(0.28+) 조합은
`Client.__init__() got an unexpected keyword argument 'proxies'`로 서버 기동 중 500 에러가 나는
알려진 비호환 문제가 있어, 실제로 이 환경에서 재현 후 상한 없는 `openai>=2.53.0`으로 고정했다.

## 테스트 흐름: index → search/ask

> Windows PowerShell에서 `curl`은 `Invoke-WebRequest`의 별칭이라 한글을 기본 인코딩으로 보내면서
> 깨질 수 있다. 아래처럼 UTF-8 바이트로 명시해서 보내는 게 안전하다. bash(git bash 등)에서는
> 일반 `curl`이 UTF-8을 그대로 보내므로 이 문제가 없다.

**1) chunk 저장 (`/index`)**

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {
        "source_type": "github", "item_type": "pr",
        "title": "Add JWT auth", "url": "https://github.com/org/repo/pull/142",
        "text": "세션은 서버 상태를 요구해서 확장이 어렵다고 판단, stateless JWT로 결정함"
      },
      {
        "source_type": "notion", "item_type": "meeting",
        "title": "로그인 방식 회의", "url": "https://notion.so/abc",
        "text": "회의 결론: 모바일 확장성 위해 토큰 기반 인증 채택."
      }
    ]
  }'
# -> {"indexed": 2}
```

**2) 관련 chunk 검색만 (`/search`, 선택)**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "왜 JWT를 선택했나요?", "top_k": 3}'
# -> {"results": [{"source_type": "...", ..., "text": "...", "score": 0.49}, ...]}
```

**3) 검색 + 답변을 한 번에 (`/ask`)**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "왜 세션 대신 JWT를 썼어요?", "lang": "vi"}'
```

기대 응답 형태 (`/ask`, `/qna` 공통):

```json
{
  "answer": "(vi로 된 모국어 답변)",
  "sources": [
    { "source_type": "github", "item_type": "pr", "title": "Add JWT auth", "url": "https://github.com/org/repo/pull/142" }
  ]
}
```

Windows PowerShell에서 위 3단계를 한 번에 확인하려면:

```powershell
function Post-Json($url, $obj) {
  $bytes = [System.Text.Encoding]::UTF8.GetBytes(($obj | ConvertTo-Json -Depth 10))
  $r = Invoke-WebRequest -Uri $url -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes -UseBasicParsing
  [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray())
}
Post-Json "http://localhost:8000/index" @{ chunks = @(@{source_type="github"; item_type="pr"; title="Add JWT auth"; url="https://github.com/org/repo/pull/142"; text="세션은 서버 상태를 요구해서 확장이 어렵다고 판단, stateless JWT로 결정함"}) }
Post-Json "http://localhost:8000/ask" @{ question = "왜 세션 대신 JWT를 썼어요?"; lang = "vi" }
```

브라우저로 `http://localhost:8000/docs`를 열면 Swagger UI에서 같은 흐름을 "Try it out"으로도 확인할 수 있다.

## 테스트 흐름: GitHub 레포 자동 수집 → ask

`/index`에 chunk를 손으로 넣는 대신, 실제 공개 레포의 PR을 긁어와 자동으로 넣을 수 있다.

```bash
curl -X POST http://localhost:8000/ingest/github \
  -H "Content-Type: application/json" \
  -d '{"repo": "pypa/sampleproject", "months": 3}'
# -> {"repo": "pypa/sampleproject", "indexed": 6}

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Python 3.14 지원이 추가됐나요?", "lang": "ko"}'
# -> {"answer": "네, ... Add Python 3.14 support ...", "sources": [{"url": "https://github.com/pypa/sampleproject/pull/240", ...}]}
```

(위 `pypa/sampleproject`, 6개 PR, 실제 응답은 이 흐름으로 직접 검증됨 — PR 제목·본문·리뷰/이슈 코멘트까지 텍스트로 합쳐져 인덱싱된다.)

PowerShell에서는 앞서 정의한 `Post-Json` 함수를 그대로 재사용하면 된다:

```powershell
Post-Json "http://localhost:8000/ingest/github" @{ repo = "pypa/sampleproject"; months = 3 }
Post-Json "http://localhost:8000/ask" @{ question = "Python 3.14 지원이 추가됐나요?"; lang = "ko" }
```

## 규칙 요약

- `/index`: chunk의 `text`를 `text-embedding-3-small`로 임베딩해 메모리에 저장. `chunks`가 비어 있으면 `indexed: 0`.
- `/search`: `question`을 같은 모델로 임베딩 후 저장된 벡터들과 코사인 유사도(numpy)로 비교, 상위 `top_k`(기본 5)개를 점수와 함께 반환. 저장된 게 없으면 빈 배열.
- `/ask`: 내부적으로 `/search`와 동일한 로직으로 상위 5개 chunk를 찾은 뒤 `/qna`와 같은 규칙으로 답변 생성.
- `question`/`lang`이 비어 있으면 422.
- 임베딩·답변 생성 API 호출이 실패하면 502.
- `/ask`, `/qna` 최종 응답의 `sources`에는 실제로 답변에 쓰인 근거의 메타(`source_type`/`item_type`/`title`/`url`)만 들어가고 `text` 원문은 들어가지 않는다. (`/search`는 검색 결과 확인용이라 `text`를 포함한다.)
- `/ingest/github`: `{repo: "owner/name", months: 3(기본)}`을 받아 최근 `months`개월 내 생성/수정된 PR만 GitHub REST API로 수집(페이지네이션 처리, `updated_at` 기준 필터). PR만 수집하고 이슈·커밋 단독 수집은 하지 않는다. 각 PR은 제목+본문+리뷰/이슈 코멘트를 이어붙여 하나의 chunk(`source_type="github"`, `item_type="pr"`)로 만들고, 본문·코멘트가 없어도 제목만으로 chunk를 만든다(건너뛰지 않음). 만들어진 chunk는 기존 `add_chunks`로 그대로 임베딩·저장된다. 응답은 `{repo, indexed}`뿐이고 PR 원문은 되돌려주지 않는다. GitHub API 실패(레포 없음·rate limit 등)는 502 + GitHub가 준 메시지를 그대로 담아 반환한다.
- `/ingest/notion`: `{pages: [{title, url, text?, item_type?(기본 "meeting")}]}`을 받아 페이지마다 `title + text`를 이어붙여 chunk(`source_type="notion"`)를 만들고 `add_chunks`로 임베딩·저장. 응답은 `{source: "notion", indexed}`. `pages`가 비어 있으면 `indexed: 0`. **BE 연동은 아직 없음** — 위 "AI 서버 구현 vs BE 연동 상태" 참고.
- `/ingest/figma`: `{comments: [{frameName, url, text?}]}`을 받아 코멘트마다 `frameName + text`를 이어붙여 chunk(`source_type="figma"`, `item_type="design"`)를 만들고 `add_chunks`로 임베딩·저장. 응답은 `{source: "figma", indexed}`. `comments`가 비어 있으면 `indexed: 0`. **BE 연동은 아직 없음** — 위 "AI 서버 구현 vs BE 연동 상태" 참고.
