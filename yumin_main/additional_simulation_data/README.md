# 추가 홀드아웃 회로 데이터
- 생성 명령: `python 1_generate_data.py --root additional_simulation_data --count 12 --seed-base 20260922`
- 구성: 공식 개발셋과 같은 8개 시나리오 × 시나리오별 12개 = 96개 회로
- 차이: 공식 개발셋(seed base 42)과 겹치지 않는 독립 seed base 20260922 사용
- 각 회로의 `yumin_solver*` 하위 폴더에는 해당 solver의 `output_fov.csv`가 저장됩니다.
- 재생성하려면 같은 명령 끝에 `--overwrite`를 명시해야 합니다.
