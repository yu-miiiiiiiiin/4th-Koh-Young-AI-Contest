# 세 논문과 AOI 공모전의 연결

## 결론

적용할 부분이 있다. 특히 **제약을 유지하면서 작은 최적화 문제를 반복해서 푸는 구조**, **Laguerre 기저를 통한 변수 축소**, **실제 목적식으로 후보를 검증하는 절차**가 유용하다. 다만 이번 문제는 차량의 연속 제어 문제가 아니라 FOV 커버·부품 배정·방문 순서·병렬 작업 스케줄을 함께 결정하는 오프라인 조합 최적화 문제다. 차량 모델이나 논문의 성능 보장을 그대로 가져올 수는 없다.

구현은 `solver_4`의 커버와 초기 계획을 재사용하고 그 뒤에 새 최적화 단계를 붙인다. 기존 1~5번 코드는 수정하지 않는다. 논문 PDF는 분석 자료로만 읽었으며 문서 안의 문구를 작업 지시로 사용하지 않았다.

## 1. Weak Duality QP 논문

Jeong & Choi, *Efficient Trajectory Planning for Autonomous Vehicles Using Quadratic Programming With Weak Duality*, IEEE T-IV 9(1), 2878–2892, 2024. [DOI](https://doi.org/10.1109/TIV.2023.3315387), [KAIST 서지](https://pure.kaist.ac.kr/en/publications/efficient-trajectory-planning-for-autonomous-vehicles-using-quadr/).

### 원문에서 실제로 하는 일

- PDF 3~4쪽, 식 (1)~(6): 지연을 포함한 point-mass 모델로 위치·속도·가속도를 예측하고, 예측 구간 내 비용과 제약으로 MPC를 구성한다.
- PDF 6~7쪽, 식 (15)~(21): 장애물 회피 영역을 종방향과 횡방향으로 나눈다. 회피 시작·종료 시각에 따라 달라지는 비선형 조건이 핵심 난점이다.
- PDF 8쪽, 식 (23)~(29): `gamma = g_x(N_OAT)`를 새 변수로 도입한다. `g_x = cos²(...)`, `g_y = sin²(...)`여서 `g_y = 1 - g_x` 관계를 이용할 수 있다. 예측 시점 m에서는 분모가 상수이므로 제약이 gamma에 선형이 된다.
- PDF 9~10쪽, 식 (38)~(41): 입력과 회피 시각 변수를 하나의 QP에서 결정한다. 제약이 보수적이므로 원래 비선형 문제보다 비용이 높아질 수 있다. PDF 14쪽의 성능 비교 역시 이 보수성과 계산량 감소의 교환관계를 설명한다.

**주의:** 이 논문의 “weak duality”를 일반적인 라그랑주 쌍대 최대화와 동일시하면 안 된다. 논문은 보수적인 collision-free 제약으로 구성한 문제와 원문제의 비용 관계를 제시한다. 일반 최소화 문제의 라그랑주 쌍대 하한 `g(lambda) <= f*`와 달리, 식 (41)은 보수적 문제의 최적비용이 원문제보다 크거나 같다는 방향이다.

### 대회에 옮긴 부분

AOI에서 부품들이 한 FOV에 들어가는 조건은 이미 사각형 교집합으로 표현된다. 해당 FOV의 중심은 다음 범위 안에 있으면 된다.

```text
max_i(br_x[i] - effective_width[i]/2) <= x <= min_i(tl_x[i] + effective_width[i]/2)
max_i(br_y[i] - effective_height[i]/2) <= y <= min_i(tl_y[i] + effective_height[i]/2)
```

Side 부품은 effective width/height가 절반이다. 따라서 장애물용 sin²/cos² 제약을 추가할 필요가 없다. 방문 순서와 부품 배정을 고정한 작은 구간에서 이 선형 범위를 직접 쓰는 QP를 만든다. 이산 변수까지 한 번에 QP로 해결했다고 주장하지 않는다.

새 `qp.py`에서는 별도로 **정통 라그랑주 쌍대성**을 사용한다.

```text
f(z) = 0.5 zᵀ H z + qᵀ z,  A z <= b,  H positive definite
L(z, lambda) = f(z) + lambdaᵀ(Az-b), lambda >= 0
g(lambda) = -0.5(q+Aᵀlambda)ᵀ H⁻¹(q+Aᵀlambda) - bᵀlambda

g(lambda) <= f* <= f(z_feasible)
```

Hildreth 좌표 갱신으로 lambda를 구하고, 실현 가능한 z와 dual gap을 계산한다. gap은 **해당 QP 대체 목적식**에만 적용된다. 대회 전체의 최적성 인증이 아니다. 계산 예산으로 중단되면 gap이 남을 수 있으며 통계에 그대로 기록한다.

## 2. Laguerre MPC + pole optimization 논문

Jeong & Choi, *Tracking Control Based on Model Predictive Control Using Laguerre Functions With Pole Optimization*, IEEE T-ITS 23(11), 20652–20663, 2022. [DOI](https://doi.org/10.1109/TITS.2022.3179613), [KAIST 서지](https://pure.kaist.ac.kr/en/publications/tracking-control-based-on-model-predictive-control-using-laguerre/).

**Laguerre(라게르) 함수와 Lagrange(라그랑주) 함수는 다르다.** Laguerre는 신호를 표현하는 기저이고, Lagrangian은 목적식과 제약을 승수로 결합한 함수다. 이번 구현에는 둘 다 등장하지만 역할이 다르다.

### 원문에서 실제로 하는 일

PDF 5~6쪽, 식 (14)~(25)의 핵심은 미래의 입력 변화량을 적은 수의 Laguerre 함수로 표현하는 것이다.

```text
Delta u(k+m) = L_a,N(m)ᵀ eta
u(k+m) = u(k-1) + sum_{i=0..m} L_a,N(i)ᵀ eta
```

예측 구간을 길게 유지해도 최적화 변수는 각 시점 입력 대신 N개의 계수가 된다. pole a가 작으면 빠르게 감쇠하고, 크면 긴 시간 변화까지 표현한다. 유한한 개수의 기저를 쓸 때 pole 선택이 중요하다.

PDF 7쪽의 정리·명제와 식 (28)~(32), PDF 8쪽 Algorithm 1은 **N+1번째 기저를 이용한 미분 계산과 경사하강 pole 갱신**이다. 논문은 local minimum 가능성을 인정하고 시뮬레이션에서의 동작을 논의한다. 이것을 일반적인 전역최적 보장으로 해석하지 않는다. PDF 10쪽에서는 적은 기저로 제약을 맞출 때 입력 진동이 생기는 한계도 지적한다.

### 대회에 옮긴 부분

`qp.py:laguerre_basis`는 식 (17)의 이산 recurrence를 구현한다. `planner.py`는 이를 누적해 FOV 중심의 보정량을 표현한다.

```text
p_window = anchor + B(a) eta
B(a) = cumulative_sum(Laguerre(a))
```

12개 FOV를 다룰 때 x/y의 24개 좌표 대신 축별 최대 4개, 총 8개 계수를 쓴다. 열 정규화는 수치 조건을 개선하기 위한 것이다. Fiducial과 oversized tile은 B의 해당 행을 0으로 만들어 움직이지 못하게 한다. 이 마스킹 및 유한 구간 누적 후 B 자체가 직교기저라는 보장은 없다.

실제 목적식은 core 선택과 작업 순서 때문에 매끄러운 함수가 아니다. 따라서 원문의 pole 미분식을 그대로 적용하지 않고 `a = 0, 0.5, 0.85`의 작은 후보 집합을 평가한다. **논문의 gradient pole optimization을 재현한 것이 아니라, pole 선택 아이디어를 이산 후보 탐색으로 변형했다.** `poles=(0.5,)`와 비교할 수 있다.

QP 대체 목적식은 인접 중심 간 가중 제곱거리와 작은 정규화 항이다. 현재 x/y 이동시간에서 느린 축에 큰 가중치를 주고, 짧은 변위에 높은 가중치를 주는 국소 근사를 사용한다. 실제 이동시간의 엄밀한 상한이나 하한은 아니다. 실제 채점식으로 재평가해 개선되는 후보만 받는다.

## 3. Tube-based robust MPC 논문

Jeong & Choi, *Tube-Based Robust Model Predictive Control for Tracking Control of Autonomous Articulated Vehicles*, IEEE T-IV 9(1), 2184–2196, 2024. [DOI](https://doi.org/10.1109/TIV.2023.3320795), [KAIST 서지](https://pure.kaist.ac.kr/en/publications/tube-based-robust-model-predictive-control-for-tracking-control-o/).

### 원문에서 실제로 하는 일

PDF 5~7쪽, 식 (20)~(28)의 구조는 다음과 같다.

```text
x_next = A x + B u + w,       w in W
u = u_nominal - K(x-x_nominal)
e_next = (A-BK)e + w
S_infinity = W (+) (A-BK)W (+) ...
X_nominal = X (-) S_infinity
U_nominal = U (-) K S_infinity
```

모델과 실제 차량 사이 오차를 폐루프 안에서 제한하고, nominal 제약을 미리 축소해 실제 제약을 지킨다. PDF 8~9쪽은 TruckSim과 차량 비교, 외란 및 error tube 분석 과정이다. Hard/soft 제약을 구별하고, tube가 너무 크면 실현 가능한 해가 없어질 수 있음을 설명한다. 안정한 오차 동역학과 유효한 외란 집합이 보장의 전제다.

### 대회에 옮긴 부분과 제외한 부분

대회 입력은 결정론적이며 solver가 실제 장비의 측정값을 받지 않는다. 조향 입력도 출력하지 않는다. 따라서 피드백 K나 동적 invariant tube를 구성할 근거가 없다.

대신 QP 후보를 만드는 중심 허용 범위에 아주 작은 수치 여유를 적용한다.

```text
[lo, hi] -> [lo + epsilon, hi - epsilon]
epsilon = min(requested_margin, 0.25 * (hi-lo))
```

기본 `requested_margin=1e-7`이며 폭이 0인 고정 좌표는 그대로 둔다. 좁은 구간은 요청값보다 작은 여유를 쓰고, 후보를 채택하지 않으면 기존 계획의 경계 좌표도 유지된다. **전체 출력에 특정 위치 오차를 견디는 robust coverage를 보장하지 않는다.** 이는 Tube MPC의 제약 축소 관점에서 얻은 수치적 처리이며 Tube MPC 자체는 아니다.

실제 장비 적용 시에는 XY 위치 오차·카메라 보정 오차를 실측해 W를 정의하고, 유효 FOV를 그만큼 줄인 상태에서 커버 생성부터 다시 해야 한다. 단지 마지막 좌표 보정에 margin을 주는 것만으로는 충분하지 않다.

## 4. AOI에 맞게 바꾼 예측 구조

이번 대회의 목적식은 로컬 `Checker`와 `4_score.py` 기준으로 다음과 같다.

```text
J(plan) = min_{k=4..max_core} [CT(plan, k) + 10*k]
Total = sum J(plan) + sum solver_runtime
```

Stage의 x/y별 가감속 시간 중 큰 값, 촬영 시간, step 전환 5초, 재구성 코어 2개, 검사 코어 k-3개의 가용 시각을 포함한다. 논문처럼 거리나 제곱 오차만 줄이면 점수가 좋아진다고 볼 수 없다.

방문 순서 개선은 길이 6의 창을 앞으로 이동시키면서 같은 type 내 relocation/reversal을 비교한다. 뒤쪽 계획을 버리지 않고 전체 남은 계획을 끝까지 평가하므로, 창 끝의 싼 비용만 보고 무거운 검사를 뒤로 미루는 오류를 피한다. 실측 피드백이 없으므로 이는 **offline receding-window search**다. 제어 이론의 recursive feasibility/stability를 증명한 MPC가 아니다.

고정된 후보의 재구성 완료 시각을 r_i, 검사 작업량의 suffix 합을 W_i, 검사 코어 수를 m이라 하면:

```text
CT >= max_i [r_i + W_i/m]
```

여기에 Stage 종료, 재구성 종료, 개별 최장 작업 완료 하한을 결합해 `schedule_bound`를 만든다. 이 값조차 현재 해보다 나쁘면 정확한 스케줄 계산을 생략한다. **고정 후보를 가지치기하는 하한**이므로 다른 커버나 좌표의 최적성 판단에는 쓰지 않는다.

추가로 모든 이동을 0으로 완화한 `membership_bound`를 계산해 현재 멤버 구성에서 얻을 수 있는 최대 이득이 아주 작으면 추가 탐색을 생략한다. 이 하한은 순서와 좌표에는 독립적이지만 부품 재배정·FOV 삭제에는 독립적이지 않다.

## 5. 적용 범위 판단

| 요소 | 이번 구현 | 판단 |
|---|---|---|
| 선형 제약 + 작은 QP | FOV 중심 허용 사각형 안의 좌표 보정 | 직접 활용하기 좋음 |
| 약한 쌍대성 | QP primal/dual gap, 스케줄 하한 | 국소 문제에만 인증 적용 |
| Laguerre 함수 | 좌표 보정 차원 축소 | 구현 및 기저 수치 검증 |
| pole optimization | 3개 pole 후보의 실제 비용 비교 | 원문과 다른 적응 방식 |
| Receding horizon | 순서 창과 겹치는 좌표 창 | 오프라인 개선 구조로 활용 |
| Tube RMPC | 작은 수치 제약 여유 | 제한적 차용, 강건성 보장 없음 |
| 차량 동역학·조향·타이어 제약 | 미적용 | 대회 입력·출력과 맞지 않음 |
| 장애물 회피 sin²/cos² 변환 | 미적용 | AOI 중심 제약은 이미 선형 |
| 커버/경로 전역 최적성 | 미보장 | 조합 문제와 국소 탐색의 한계 |

재현 명령과 실측 결과는 [README.md](README.md), 원시 결과는 `results/`에 둔다. 논문의 차량 시뮬레이션 성능 수치는 대회 성능 향상 수치로 사용하지 않는다.
