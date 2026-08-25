# trolley 연동 메모

같은 워크스페이스의 자매 프로젝트 `../trolley`에 대해, 이 저장소 쪽에서 알아야 할 것만
적는다. 아직 두 프로젝트는 **연결돼 있지 않다.**

## 저쪽이 무엇인가

macOS 접근성 API(AXUIElement 트리)로 GUI 앱을 조작하는 Swift 도구다. 스크린샷도
좌표도 비전 모델도 쓰지 않는다.

| | |
| --- | --- |
| 위치 | `/Users/markhub/Desktop/workspace/llm/trolley` |
| 원격 | `github.com/minsuRob/trolley` |
| 실행 | `trolley mcp` — stdio로 JSON-RPC 2.0(MCP)을 말하는 서버 |
| 설치 경로 | `~/bin/trolley` (접근성 권한이 부여된 경로) |
| 툴 | 11종 — 아래 참고 |

Claude Code에는 이미 등록돼 있다(`claude mcp add trolley -- ~/bin/trolley mcp`,
trolley 프로젝트 스코프).

### 툴 표면

`check_permissions`, `list_apps`, `launch_app`, `snapshot`, `find_elements`,
`click`, `focus`, `type_text`, `press_key`, `set_ax_value`, `wait_for_element`.

`snapshot`이 AX 트리를 LLM이 읽기 좋은 JSON으로 주고 노드마다 `e1`, `e2` 같은 ID를
붙인다. 이후 동작 툴은 그 ID로 요소를 지목한다. 서버가 살아 있는 동안 ID가 유지되므로
"방금 찾은 그 버튼을 눌러"가 성립한다.

## 지금 연동이 안 되는 이유

**이 저장소에 툴 콜링이 없다.** `server.py`는 모델을 인프로세스로 로드해 텍스트 생성과
이미지 읽기만 하고, 에이전트 루프도 함수 호출도 MCP 클라이언트도 없다. 모델 출력이
동작으로 디스패치되는 경로가 존재하지 않는다.

trolley 쪽은 준비돼 있다. 우리가 MCP 클라이언트를 말할 수만 있으면 붙는다.

## 붙이려면

두 갈래가 있고 각각 대가가 다르다.

**A. `server.py`에 MCP 클라이언트 + 에이전트 루프를 넣는다.**
`~/bin/trolley mcp`를 서브프로세스로 띄우고 stdin/stdout으로 개행 구분 JSON을 주고받으면
된다. 프로토콜이 얇다 — `initialize`, `notifications/initialized`, `tools/list`,
`tools/call` 넷이면 충분하고, 툴 실패는 예외가 아니라 `isError: true` 결과로 온다.

주의할 점은 **우리 쪽 단일 워커 제약**이다. `Engine`이 모델 하나·워커 스레드 하나로
모든 요청을 직렬화하므로(중복 실행 시 ~34GB로 OOM), 에이전트 루프가 도는 동안 채팅
요청은 큐에서 대기한다(상한 8, 초과 시 503). 루프가 툴을 여러 번 호출하면 그만큼
점유 시간이 길어진다. SSE로 진행 상황을 흘려보내는 기존 패턴을 그대로 쓰면 사용자에게
"멈춘 것처럼" 보이지는 않게 할 수 있다.

**B. OpenAI 호환 API를 쓴다.**
README에 적힌 대로 `server.py` 대신 `python -m mlx_vlm.server --port 8080`을 띄우면
기존 툴 콜링 클라이언트를 그대로 붙일 수 있다. **단 `server.py`와 동시에 띄우면 모델이
두 번 로드되어 즉시 OOM이다.** 웹 UI와 추출 파이프라인을 포기하는 셈이라 트레이드오프가
크다.

어느 쪽이든 trolley 쪽 수정은 필요 없다.

## 붙일 때 알아야 할 trolley 쪽 제약

- **접근성 권한은 trolley 바이너리 자체에, 경로 단위로 부여된다.** 부모 프로세스와 무관
  하므로 파이썬이 서브프로세스로 띄워도 그 경로가 승인돼 있어야 한다. 먼저
  `check_permissions`를 호출해 확인할 것. 재빌드로 바이너리가 바뀌면 재승인이 필요할 수 있다.
- **stdout은 JSON-RPC 전용이다.** 진단은 전부 stderr로 나가므로 파이프를 분리해야 한다.
- **trolley도 한 번에 한 요청만 처리한다.** AX 호출이 동기라 의도적으로 직렬이다.
  즉 양쪽 모두 직렬이므로 지연은 두 병목의 합이다.
- **`type_text`의 `method`를 이해해야 한다.** 기본 `paste`가 한글·이모지를 모두 처리한다
  (클립보드를 쓰고 원래대로 복원한다). `keys`는 ASCII 전용이며 입력원을 잠시 바꾼다.
  `unicode`는 이 머신에서 전달되지 않으니 쓰지 말 것.
- **`verification: "unverifiable"`은 성공이 아니라 모름이다.** Chromium·리치텍스트 뷰가
  값을 보고하지 않아 흔히 나온다. 시스템 프롬프트에서 모델에게 이를 성공으로 해석하지
  말라고 명시해야 한다. 같은 이유로 trolley는 자동 재시도를 하지 않는다 — 검증 불가 상태에서
  재시도하면 텍스트가 두 번 들어갈 수 있기 때문이다.
- **Chromium/Electron 웹 콘텐츠는 AX 트리가 안 열리는 경우가 있다.** `thorough=true`로도
  안 되면 그 영역은 AX 방식의 구조적 한계다. 우리에게는 오히려 기회다 — 이 저장소는 이미
  이미지를 읽는 모델을 갖고 있으므로, 네이티브 UI는 trolley의 AX로, 웹 콘텐츠는
  스크린샷 + DiffusionGemma의 비전으로 처리하는 하이브리드가 자연스럽다.

## 반대 방향

trolley 쪽에서 필요한 정보는 그 저장소의 `docs/DiffusionGemma-local-연동.md`에 있다.
