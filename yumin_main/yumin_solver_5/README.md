# yumin_solver_5

**핵심 키워드:** Machine Learning · Feature Clustering · Adaptive Budget · Exact Set Cover · LNS

1. 부품 수·회로 밀도·Side 비율·부품 크기·검사시간·최대 코어 수 등의 특징을 회로에서 추출합니다.
2. 학습된 경량 Cluster 모델이 새 회로와 가장 가까운 유형을 찾고, 그 유형에 맞는 경로 탐색 시간 배율을 선택합니다.
3. 선택된 시간 예산으로 4번의 `Cover → Route → Refine` 파이프라인을 실행합니다. 즉 머신러닝은 FOV를 직접 만들지 않고 탐색량을 조절합니다.
4. 기본 `solve()`는 실행시간도 점수에 포함된다는 점을 고려해, 검증된 짧은 탐색 예산과 4번 커버를 사용합니다.
5. 선택 기능에서는 작은 커버 문제를 Exact Set Cover와 Branch-and-Bound로 풀고, 큰 문제는 LNS로 일부 영역을 반복 재구성할 수 있습니다.
6. 커버 개선으로 얻는 점수보다 계산시간 손해가 클 수 있어 Exact Cover는 기본 경로에서는 비활성화되어 있습니다.

실행: `python 2_run.py --solver yumin_solver_5`
