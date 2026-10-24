# yumin_solver_7

시뮬레이터 피드백으로 학습한 모델이 부품 이동·교환·FOV 병합 후보를 정렬하고, 정확한 채점 시뮬레이터가 최종 채택 여부를 결정하는 solver다. 논문 확인 범위와 대회에 맞춘 변경은 [PAPER_ANALYSIS.md](PAPER_ANALYSIS.md)에 기록했다. 전기방사 논문은 공개 원문을 읽었으며, 메타물질 논문은 확보 가능한 초록·서론의 방법론을 참고했다.

## 실행

저장소 루트에서 실행한다.

```bash
python3 2_run.py --solver yumin7
python3 4_score.py yumin7
```

`solve(folder) -> FOV 개수` 인터페이스를 지원한다. 입력 폴더에 `output_fov.csv`를 작성한다. `2_run.py`를 이용하면 기존 solver 결과와 별도 폴더에 저장된다. NumPy/Pandas와 기존 `yumin_solver_4`, `yumin_solver_5`를 재사용하므로 이 폴더만 떼어 제출하는 독립 패키지는 아니다. 추론에 GPU, PyTorch 또는 scikit-learn은 필요 없다.

## 알고리즘

1. solver_5와 같은 예산 선택기를 이용해 solver_4 초기 해를 생성한다.
2. 위치·검사 부하·촬영 비용·후반 부하 등 23개 상태/행동 특징을 계산한다.
3. 부품 이동, 두 부품 교환, 두 FOV 병합 중 기하학적으로 가능한 후보를 만든다. 인접 최대 8개 FOV와 일부 대표 부품을 사용하며, 휴리스틱 사전 정렬 후 최대 240개를 남긴다.
4. 학습한 깊이 3의 회귀 트리 32개가 즉시 보상을 예측한다. 상위 24개만 정확하게 평가하고 `min_k(CT + 10k)`가 낮아지는 최선의 후보를 채택한다.
5. 변경된 상태에서 다시 예측한다. 최대 5라운드, 추가 탐색 0.4초의 소프트 시간 한도를 사용한다. 예측 보상이 전부 0 이하이거나 직전 라운드의 실제 이득이 시간 비용보다 작으면 중단한다.

경로 중심점 전체를 다시 이동시키지 않고, 배정이 바뀐 FOV만 새 허용 구간으로 투영한다. fiducial 및 oversized 부품이 들어간 FOV는 추가 탐색에서 고정한다. 이동하는 부품의 step과 barcode 순서 블록도 보존한다. 최종 CSV의 FOV 내 부품은 검사시간 내림차순이다.

이는 `gamma=0`의 contextual bandit에 해당하는 즉시 보상 회귀다. DDQN 또는 메타물질 논문의 세부 알고리즘을 그대로 재현한 것은 아니다. 모델이 직접 채점하거나 제약조건을 판정하지 않는다.

## 학습과 검증

- 학습: `robustness_data/train_seed_31001`, 96회로, 51,821개 후보 피드백.
- 모델: [reward_model.json](reward_model.json). 추론은 NumPy 배열 연산만 사용한다.
- 기록: [training/manifest.json](training/manifest.json), [training/fit_report.json](training/fit_report.json), `training/feedback.npz`.
- 설정 선택: `robustness_data/validation_seed_41001`의 96회로에서 수행했다. 짧은 탐색과 긴 탐색의 비용을 포함해 비교한 뒤 5라운드/24개/0.4초를 선택했다.
- 최종 평가: 기본 `simulation_data`와 새로 생성한 `robustness_data/solver7_holdout_seed_91007`. 최종 평가 결과를 보고 모델이나 설정을 다시 학습하지 않는다.

학습 재실행에는 scikit-learn이 필요하다. 기존 모델 파일을 갱신하는 명령이다.

```bash
python3 -m pip install -r yumin_solver_7/requirements-train.txt
python3 -m yumin_solver_7.train
# 저장된 피드백으로 모델만 다시 학습
python3 -m yumin_solver_7.train --reuse
```

학습 라벨은 후보를 실제로 계산한 시간에 의존하므로 완전히 동일한 가중치를 재생성한다고 보장하지 않는다. 학습 회로의 ID나 이름은 특징으로 사용하지 않는다. 기존 solver_5의 예산 모델은 그대로 재사용한다.

## 평가 재현

```bash
python3 -m unittest yumin_solver_7.test_solver -v
python3 -m yumin_solver_7.benchmark --data simulation_data --label practice --repeat 2
python3 -m yumin_solver_7.benchmark --data robustness_data/solver7_holdout_seed_91007 --label holdout --repeat 2
python3 4_score.py yumin_solver_7/results/practice/summary_yumin_solver_7_repeat0.csv
```

미사용 회로를 새로 생성하려면 저장소의 Rtree 의존성이 필요하다.

```bash
python3 1_generate_data.py --root robustness_data/solver7_holdout_seed_91007 --seed-base 91007 --count 12
```

각 비교 실행은 solver 5·6·7의 실행 순서를 회로마다 순환시킨다. 입력 파싱과 CSV 저장까지 시간을 측정하고 공식 Checker 계산시간은 제외한다. 병렬로 solver를 실행하지 않는다. 커버리지/검사 순서/등록순 fiducial 중심을 검사하고, 내부 `J`와 공식 `CT + 10 × core`가 일치하는지 확인한다. 위반이 나오면 보정 solver로 덮지 않고 실패한다.

`comparison.csv`는 회로별 결과, `totals.csv`는 반복별 합계, `summary_*.csv`는 `4_score.py` 호환 입력이다. 각 최종 평가 폴더의 `manifest.json`에는 실행 환경과 소스·모델 SHA-256을 기록한다. 입력 CSV와 이전 solver 결과는 변경하지 않는다.

## 실험 기록의 해석

`results/validation_pilot`은 초기 24회로 모델의 탐색 실험이며 현재 배포 모델과 다르다. `results/validation_final`은 최종 96회로 모델의 설정 비교다. 이 폴더에서 `learned`는 3라운드/16개/0.18초, `learned_long`은 현재 기본 설정인 5라운드/24개/0.4초를 뜻한다. 검증에서 긴 학습 탐색은 초기 해보다 1.787978점 낮았지만 개선 폭은 작다. 최종 점수와 반복별 변동은 [RESULTS.md](RESULTS.md)에 기록한다.
