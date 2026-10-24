# yumin_solver_4

**핵심 키워드:** Minimum Piercing · POMO Routing · Load Balancing · Iterative Refinement

1. 3번의 Minimum Piercing 커버와 POMO 기반 경로 탐색을 출발점으로 사용합니다.
2. 커버가 끝난 뒤에도 부품을 처음 지정된 FOV에 고정하지 않고, 더 일찍 방문하는 호환 FOV로 재배정합니다.
3. 이 재배정은 촬영 위치를 크게 바꾸지 않으면서 검사 코어의 부하와 마지막 처리 구간을 줄이는 역할을 합니다.
4. 부품 이동으로 비게 된 FOV는 제거하고, 남은 FOV 중심은 허용 범위 안에서 이동시켜 Stage 이동 시간을 줄입니다.
5. 변경된 FOV 집합을 다시 라우팅하고 Tail Lightening·Local Search로 병목 구간을 추가 개선합니다.
6. `Cover → Route → Refine → Re-route`를 반복하되 실제 채점 목적식이 좋아지는 후보만 채택합니다.

실행: `python 2_run.py --solver yumin_solver_4`
