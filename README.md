# DiffusionGemma Local

로컬(Apple Silicon)에서 [DiffusionGemma](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)를
MLX로 구동하고, 웹 채팅 UI와 SSE 스트리밍 서버로 여러 기기에서 함께 쓰는 구성.

- 모델: [mlx-community/diffusiongemma-26B-A4B-it-4bit](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit)
  (26B MoE / 활성 3.8B, 4-bit 양자화 ≈ 16.5GB)
- 런타임: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — 디퓨전 디코딩(`diffusion_gemma`) 네이티브 지원
- 검증 환경: MacBook Pro M5 Max / 36GB

## 설치

```bash
python3 -m venv .venv
.venv/bin/pip install -U mlx-vlm huggingface_hub jinja2
.venv/bin/hf download mlx-community/diffusiongemma-26B-A4B-it-4bit
```

## 웹 서버 (권장)

```bash
.venv/bin/python server.py
```

실행하면 액세스 토큰이 포함된 LAN URL을 출력한다. 같은 네트워크의 폰·태블릿·다른 맥에서
그 주소를 열면 바로 쓸 수 있다.

```
Open on any device on this network:
  http://192.168.0.10:8842/?token=Xc9f...
```

주요 옵션: `--port` (기본 8842), `--host`, `--max-context` (기본 96,000),
`--max-tokens` (기본 1024), `--token` (고정 토큰; 생략 시 매 실행마다 랜덤 발급).
`DIFFUSIONGEMMA_TOKEN` 환경변수로도 지정할 수 있다.

### 기능

- 첨부 화면과 같은 대화 목록 사이드바 (검색, 새 채팅, 삭제)
- SSE 토큰 스트리밍, 대기열 순번 실시간 표시, 생성 중지
- 추론 채널(`생각하는 과정 표시`) 접기/펼치기
- 마크다운 + 코드블록 렌더링, LaTeX 수식 평문 변환
- 대화 기록은 서버 `conversations.db`(SQLite)에 저장되어 기기 간 공유

## CLI 채팅

```bash
.venv/bin/python chat.py
```

`/reset`(히스토리 초기화), `/stats`(속도 통계 토글), `/quit`.

## OOM 방지 설계

모델은 상주 17GB, 긴 컨텍스트에서 28GB까지 쓴다. 36GB 머신에서 **두 번째 생성이 동시에
돌면 느려지는 게 아니라 그냥 죽는다.** 그래서 동시성은 병렬 실행이 아니라 대기열로 처리한다.

| 장치 | 동작 |
|---|---|
| 단일 워커 직렬화 | 생성은 언제나 한 번에 하나. 나머지는 큐에서 대기하며 순번을 SSE로 받는다 |
| 대기열 상한 8 | 초과 요청은 503으로 즉시 거부 (무한 적체 방지) |
| 중복 실행 차단 | PID 락파일 + `mlx_vlm.server` 프로세스 스캔. 두 인스턴스는 ~34GB라 하드 스톱 |
| 포트 사전 점검 | 모델 로드(3초) 전에 바인드 가능 여부를 먼저 확인 |
| 컨텍스트 예산 | 기본 96K, 하드 상한 120K. 초과 시 오래된 턴부터 자동 제외 |
| 청크 프리필 | `prefill_step_size=2048`. 없으면 ~16K에서 Metal 버퍼 한계로 OOM |
| 캐시 정리 | 작업이 끝날 때마다 `mx.clear_cache()` |

> **주의**: `mlx_vlm.server`(OpenAI 호환 API)를 쓰려면 `server.py`를 먼저 내려야 한다.
> 둘을 동시에 띄우면 모델이 두 번 로드되어 즉시 OOM이다. `server.py`는 이를 감지해
> 시작을 거부하지만, 반대 순서로 띄우면 막아주지 못한다.

OpenAI 호환 API가 필요할 때 (openclaw 등 외부 클라이언트 연동):

```bash
.venv/bin/python -m mlx_vlm.server --model mlx-community/diffusiongemma-26B-A4B-it-4bit --port 8080
```

## 측정 결과

### 생성 품질·속도 (`bench.py`, 8종 프롬프트)

사실 질문·한국어·추론·코드·산수·요약·창작·JSON 8개 전부 정상 출력.
생성 속도 **51~208 tok/s**, 평균 약 120 tok/s. 자기회귀 모델과 반대로 **출력이 길수록 빨라진다**
(256토큰 블록을 병렬 생성하는 디퓨전 디코딩 특성). 424토큰 추론 답변이 3.4초.

### 입력 길이 (`limits.py`, 출력 1024토큰 기준)

| 입력 토큰 | 피크 메모리 | 총 소요 | 생성 속도 |
|---|---|---|---|
| 77K | 24.6GB | 66초 | 34 tok/s |
| 96K | 26.5GB | 94초 | 27 tok/s |
| 116K | 27.9GB | 131초 | 25 tok/s |

회귀식 `피크GB ≈ 18.0 + 0.085 × (입력K)`.
모델 스펙 최대는 262K지만, 36GB 기준 실질 한계는 **~120K**이며 기본값을 96K로 두는 이유는:

1. `iogpu.wired_limit_mb` 기본값이 약 27GB(36GB의 75%)라 116K는 이미 그 선을 넘는다
2. 120K면 macOS와 다른 앱에 8GB 남짓만 남는다
3. 메모리보다 지연시간이 먼저 문제 — 116K는 한 턴에 2분, 속도는 8배 느려진다

여유 메모리를 더 쓰고 싶다면 wired limit을 직접 올릴 수 있다(재부팅 시 초기화):

```bash
sudo sysctl iogpu.wired_limit_mb=30720
```

### 동시 요청 (실측)

4개 기기 동시 요청 → 순번 `3 → 2 → 1 → 시작`으로 직렬 처리, **피크 17.63GB 유지**.
9개 큐 적체 상태에서도 피크 17.92GB, 스왑 증가 0.

## 파일

| 파일 | 역할 |
|---|---|
| [server.py](server.py) | SSE 서버, 단일 워커 큐, SQLite, 토큰 인증 |
| [web/](web/) | 채팅 UI (`index.html`, `style.css`, `app.js`) |
| [context_guard.py](context_guard.py) | 컨텍스트 상한·히스토리 트리밍 (CLI/서버 공용) |
| [chat.py](chat.py) | 터미널 채팅 |
| [bench.py](bench.py) | 품질·속도 벤치마크 → `bench_results.json` |
| [limits.py](limits.py) | 입력 길이 한계 **탐색 프로브** (런타임 경로 아님) |

## 디퓨전 전용 생성 옵션

`stream_generate(...)`에 넘길 수 있는 주요 파라미터:

- `max_denoising_steps` — 디노이징 반복 상한 (속도↔품질)
- `block_length` / `diffusion_max_canvas_length` — 병렬 생성 캔버스 길이
- `diffusion_sampler` — 기본 `confidence-threshold`
