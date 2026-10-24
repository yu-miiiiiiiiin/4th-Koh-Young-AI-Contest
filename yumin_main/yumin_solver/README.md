# yumin_solver (버전 1)

**핵심 키워드:** Riemann Tiling · Greedy Absorption · Snake Ordering · Nearest Neighbor

1. 부품을 `type → step → side` 순서로 그룹화해 Fiducial, Barcode, 일반 부품의 검사 순서 규칙을 먼저 고정합니다.
2. 각 그룹의 전체 영역을 FOV 크기의 격자로 나누고, 실제 부품이 들어 있는 타일에만 FOV를 배치합니다. 격자 원점을 여러 위치에서 시험해 타일 경계에 걸리는 부품과 필요한 FOV 수가 적은 위치를 고릅니다.
3. Side 부품을 절반 크기 FOV로 먼저 타일링하고, 그 FOV에 함께 들어오는 Top 부품을 흡수해 중복 촬영을 줄입니다.
4. 타일 경계에 걸려 한 FOV에 들어오지 않는 부품에는 중심 FOV를 추가하고, 함께 들어오는 잔여 부품도 Greedy 방식으로 흡수합니다.
5. FOV보다 큰 부품은 여러 FOV로 분할해 각 촬영 영역의 합집합이 부품 전체를 덮도록 합니다.
6. Barcode FOV에 호환 일반 부품을 흡수한 뒤 남은 부품만 타일링합니다. 격자는 Snake, 추가 FOV는 Nearest Neighbor 순회합니다.

실행: `python 2_run.py --solver yumin_solver`
