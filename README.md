# rag-docs-chatbot

문서 업로드 → 벡터 검색(Chroma) → 출처 기반 답변을 제공하는 Flask RAG 챗봇 MVP.

## 기능

| 단계 | 경로 | 설명 |
|------|------|------|
| 업로드 | `/` | `.txt` 업로드, 목록·삭제 |
| 인덱싱 | `/index` | 청크 집계(GET), 임베딩·Chroma 저장(POST 버튼) |
| 질문 | `/ask` | 유사 청크 검색 + OpenAI 답변 + 출처 표시 |

## 기술 스택

- Python 3.13, Flask
- OpenAI API (임베딩 `text-embedding-3-small`, 채팅 `gpt-4o-mini`)
- ChromaDB (로컬 `chroma_data/`)

## 실행 방법 (Windows / PowerShell)

```powershell
cd rag-mvp
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` 파일 생성 (`.env.example` 참고):

```env
OPENAI_API_KEY=sk-...
```

```powershell
python app.py
```

브라우저: http://127.0.0.1:5000

## 사용 순서

1. **홈**에서 `.txt` 업로드
2. **인덱싱** 페이지 → **인덱싱 실행** 클릭
3. **질문하기**에서 질문 입력 → 답변·출처 확인

파일을 추가·삭제한 뒤에는 반드시 **재인덱싱**하세요.

## 폴더 구조

```
rag-mvp/
  app.py              # Flask 앱
  templates/          # HTML
  uploads/            # 업로드 문서
  chroma_data/        # 벡터 DB (git 제외)
  requirements.txt
  .env                # API 키 (git 제외)
```

## 환경 변수

| 변수 | 필수 | 설명 |
|------|------|------|
| `OPENAI_API_KEY` | 예 | OpenAI API 키 |
| `SECRET_KEY` | 아니오 | Flask flash용 (미설정 시 개발용 기본값) |

## 검색 품질

- `RETRIEVAL_TOP_K`: 가져올 청크 상한 (기본 5)
- `MAX_RETRIEVAL_DISTANCE`: 코사인 거리 임계값 (기본 0.75). 이보다 먼 청크는 LLM에 넣지 않음.

