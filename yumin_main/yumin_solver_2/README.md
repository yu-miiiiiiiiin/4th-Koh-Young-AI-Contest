# yumin_solver_2

**핵심 키워드:** Spatial Clustering · Minimum Piercing · Helly Property · Merge/Dissolve/Absorb

1. 1번처럼 전체 영역을 격자로 덮지 않고, 실제 부품이 존재하는 셀만 연결해 공간 클러스터를 만듭니다.
2. Top과 Side 부품은 가능한 경우 같은 촬영 위치를 공유해 중복 이동과 FOV 수를 줄입니다.
3. 각 부품을 완전히 촬영할 수 있는 FOV 중심의 범위를 `Feasible-center Box`라는 사각형으로 계산합니다.
4. 여러 사각형이 겹치는 한 점에 FOV를 놓으면 여러 부품을 동시에 촬영할 수 있으므로, 문제를 최소 개수의 점을 찾는 `Minimum Piercing`으로 바꿉니다.
5. 8가지 축·방향으로 Sweep한 후보 중 가장 적은 FOV를 선택하고, Merge·Dissolve·Absorb로 불필요한 FOV를 추가 제거합니다.
6. 클러스터끼리는 Nearest Neighbor, 클러스터 내부는 Snake 순서로 연결한 뒤 FOV 중심을 이동 가능한 범위에서 미세 조정합니다.
7. 같은 FOV 안의 부품은 검사시간이 긴 순서로 정렬해 코어의 마지막 처리 지연을 줄입니다.

실행: `python 2_run.py --solver yumin_solver_2`
