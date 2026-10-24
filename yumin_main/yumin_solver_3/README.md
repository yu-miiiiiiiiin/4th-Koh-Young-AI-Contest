# yumin_solver_3

**핵심 키워드:** Minimum Piercing · POMO · Multi-start · Symmetry Augmentation · LPT

1. 스텝별 Minimum Piercing으로 FOV 커버를 만들며, Fiducial은 규칙에 따라 전용 FOV에 배치합니다.
2. Barcode와 일반 부품은 허용되는 경우 같은 FOV에 넣어 촬영 횟수를 줄입니다.
3. 생성된 FOV의 방문 순서는 POMO 방식의 Multi-start Rollout으로 탐색합니다. 서로 다른 시작 FOV에서 여러 경로를 동시에 만들고 가장 좋은 경로를 남깁니다.
4. 회로를 회전·반전한 8개 대칭 관점에서도 후보를 만들어 특정 방향에 치우친 경로가 선택되는 것을 줄입니다.
5. 모든 후보는 실제 채점기와 같은 `CT + 10 × 코어 수` 목적식으로 평가하며, 마지막에 2-opt·Or-opt Local Search로 경로를 보정합니다.
6. 한 FOV 안의 부품은 검사시간이 긴 순서인 LPT로 배치해 검사 코어의 마지막 작업이 과도하게 늦어지는 것을 막습니다.
7. 여러 seed 평가로 POMO 초기 가중치와 탐색시간을 조정하되, 매 회로에서 다시 보상 기반 탐색을 수행합니다.

실행: `python 2_run.py --solver yumin_solver_3`
