"""Build a self-contained visual report for the final AOI planner."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
BOARD = "S00_sparse_fast_010"


def compact_records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    return frame[columns].round(5).to_dict(orient="records")


def build_data() -> dict:
    board_dir = ROOT / "simulation_data" / BOARD
    comp = pd.read_csv(board_dir / "input_component.csv")
    size = pd.read_csv(board_dir / "input_size.csv")
    greedy_fov = pd.read_csv(board_dir / "greedy_solver" / "output_fov.csv")
    final_fov = pd.read_csv(board_dir / "yumin_solver_5" / "output_fov.csv")

    components = []
    for row in comp.itertuples():
        components.append({
            "x": round(float(row.tl_x), 4),
            "y": round(float(row.tl_y), 4),
            "w": round(float(row.br_x - row.tl_x), 4),
            "h": round(float(row.br_y - row.tl_y), 4),
            "type": int(row.type),
            "side": int(row.side),
        })

    def fovs(frame: pd.DataFrame) -> list[dict]:
        result = []
        for row in frame.itertuples():
            members = json.loads(row.comp_idx)
            result.append({
                "x": round(float(row.x), 4),
                "y": round(float(row.y), 4),
                "side": int(any(int(comp.iloc[i].side) == 1 for i in members)),
            })
        return result

    summaries = {}
    for name in ("greedy", "yumin_solver_5"):
        frame = pd.read_csv(ROOT / f"summary_{name}.csv").set_index("dataset")
        row = frame.loc[BOARD]
        summaries[name] = {
            "n_fov": int(row.n_fov),
            "core": int(row.core),
            "ct": round(float(row.ct), 2),
            "runtime": round(float(row.solve_s), 2),
            "score": round(float(row.ct + row.solve_s + 10 * row.core), 2),
        }

    score = pd.read_csv(ROOT / "robustness_results" / "score_comparison.csv")
    group_order = [
        "simulation_data",
        "additional_simulation_data",
        "robustness_data/train_seed_31001",
        "robustness_data/validation_seed_41001",
        "robustness_data/holdout_seed_51001",
        "robustness_data/stress_seed_61001",
        "robustness_data/stress_seed_71001",
        "robustness_data/stress_seed_81001",
    ]
    label_map = {
        "simulation_data": "원본 · 42",
        "additional_simulation_data": "추가 · 20260922",
        "robustness_data/train_seed_31001": "학습 · 31001",
        "robustness_data/validation_seed_41001": "검증 · 41001",
        "robustness_data/holdout_seed_51001": "홀드아웃 · 51001",
        "robustness_data/stress_seed_61001": "스트레스 · 61001",
        "robustness_data/stress_seed_71001": "스트레스 · 71001",
        "robustness_data/stress_seed_81001": "스트레스 · 81001",
    }
    groups = []
    for group in group_order:
        greedy = score[(score.dataset_group == group) & (score.solver == "greedy")].iloc[0]
        final = score[(score.dataset_group == group) & (score.solver == "yumin_solver_5")].iloc[0]
        groups.append({
            "label": label_map[group],
            "greedy": round(float(greedy.score), 2),
            "final": round(float(final.score), 2),
            "improvement": round(float(final.improvement_pct), 2),
        })

    totals = {}
    for name in ("greedy", "yumin_solver_5"):
        rows = score[score.solver == name]
        totals[name] = {
            "score": round(float(rows.score.sum()), 2),
            "ct": round(float(rows.ct.sum()), 2),
            "runtime": round(float(rows.runtime.sum()), 2),
            "core_cost": round(float(rows.cores.sum() * 10), 2),
        }
    totals["improvement"] = round(
        100 * (totals["greedy"]["score"] - totals["yumin_solver_5"]["score"])
        / totals["greedy"]["score"], 2
    )

    practice = score[score.dataset_group == "simulation_data"].set_index("solver")
    evolution = []
    names = [
        ("greedy", "Greedy"),
        ("yumin_solver", "Solver 1"),
        ("yumin_solver_2", "Solver 2"),
        ("yumin_solver_3", "Solver 3"),
        ("yumin_solver_4", "Solver 4"),
        ("yumin_solver_5", "Solver 5"),
    ]
    for key, label in names:
        evolution.append({"label": label, "score": round(float(practice.loc[key].score), 2)})

    return {
        "board": {
            "name": BOARD,
            "width": float(size.loc[size.axis == "x", "pcb_size"].iloc[0]),
            "height": float(size.loc[size.axis == "y", "pcb_size"].iloc[0]),
            "fov_width": float(size.loc[size.axis == "x", "fov_size"].iloc[0]),
            "fov_height": float(size.loc[size.axis == "y", "fov_size"].iloc[0]),
            "components": components,
            "greedy_fovs": fovs(greedy_fov),
            "final_fovs": fovs(final_fov),
            "summary": summaries,
        },
        "groups": groups,
        "totals": totals,
        "evolution": evolution,
    }


TEMPLATE = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="ICCAS 2026 AOI 검사 경로 최적화 알고리즘 시각화">
<title>AOI Planner · 알고리즘 시각화</title>
<style>
:root{--ink:#171a1d;--muted:#626970;--paper:#f3f5f5;--panel:#fff;--line:#d9dddc;--blue:#2f67a9;--blue-soft:#e8f0f8;--orange:#b86732;--orange-soft:#f4e8df;--green:#38815e;--green-soft:#e5f1ea;--violet:#7557a6;--shadow:0 18px 45px rgba(18,28,35,.08);font-family:Inter,Pretendard,"Noto Sans KR",system-ui,-apple-system,sans-serif;color:var(--ink);background:var(--paper)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);line-height:1.65}button{font:inherit}.wrap{width:min(1120px,calc(100% - 40px));margin:auto}.hero{padding:104px 0 82px;border-bottom:1px solid var(--line);background:radial-gradient(circle at 80% 15%,rgba(47,103,169,.12),transparent 28%),linear-gradient(180deg,#eef2f3 0%,var(--paper) 100%)}
.eyebrow{font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);font-weight:700}.hero h1{font-size:clamp(42px,7vw,82px);letter-spacing:-.065em;line-height:1.02;margin:22px 0 28px;max-width:900px}.hero h1 em{font-style:normal;color:var(--blue)}.lead{font-size:clamp(17px,2vw,21px);color:var(--muted);max-width:720px;margin:0}.hero-meta{display:flex;gap:24px;flex-wrap:wrap;margin-top:40px;font-size:13px}.hero-meta span{display:flex;gap:8px;align-items:center}.dot{width:8px;height:8px;border-radius:50%;background:var(--green)}
section{padding:86px 0;border-bottom:1px solid var(--line)}.section-head{display:grid;grid-template-columns:170px 1fr;gap:24px;margin-bottom:38px}.kicker{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:700}.section-head h2{font-size:clamp(30px,4vw,48px);line-height:1.13;letter-spacing:-.045em;margin:0}.section-head p{grid-column:2;color:var(--muted);max-width:720px;margin:-10px 0 0}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.board-card{background:var(--panel);border:1px solid var(--line);border-radius:18px;overflow:hidden;box-shadow:var(--shadow)}.board-title{display:flex;justify-content:space-between;align-items:end;padding:18px 20px 14px}.board-title strong{font-size:15px}.board-count{font-size:34px;line-height:1;font-weight:750;letter-spacing:-.05em}.board-count small{font-size:12px;color:var(--muted);margin-left:4px}.board-wrap{position:relative;aspect-ratio:1.08;background:#f8f8f5;border-top:1px solid var(--line)}canvas{width:100%;height:100%;display:block}.compare-tools{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin:18px 0 0}.check{display:flex;gap:8px;align-items:center;font-size:14px}.legend{display:flex;gap:14px;flex-wrap:wrap;margin-left:auto;font-size:12px;color:var(--muted)}.swatch{display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:2px}.reduction{font-size:14px;color:var(--muted);margin-top:16px}.reduction strong{font-size:23px;color:var(--blue)}
.score-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:32px;align-items:start}.score-stack{display:grid;gap:18px}.score-row{display:grid;grid-template-columns:110px 1fr 82px;gap:14px;align-items:center}.score-row strong{text-align:right;font-variant-numeric:tabular-nums}.bar{height:38px;display:flex;background:#e3e7e8;border-radius:6px;overflow:hidden}.seg{height:100%;transition:width .45s ease}.seg.ct{background:var(--blue)}.seg.core{background:var(--orange)}.seg.runtime{background:var(--green)}.score-legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-left:124px}.formula{background:var(--ink);color:#fff;padding:26px 28px;border-radius:16px}.formula .mini{color:#aeb7bc;font-size:12px;letter-spacing:.12em}.formula code{display:block;font:600 20px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:14px 0}.formula p{color:#cbd1d4;margin:0;font-size:14px}.formula b{color:#fff}
.concept{display:grid;grid-template-columns:1fr 1fr;gap:42px;align-items:center}.concept-visual{min-height:390px;background:var(--panel);border:1px solid var(--line);border-radius:18px;position:relative;overflow:hidden}.part{position:absolute;left:43%;top:43%;width:17%;height:12%;background:var(--orange);opacity:.88}.fov-box{position:absolute;left:17%;top:20%;width:66%;height:60%;border:2px solid var(--blue);background:rgba(47,103,169,.07)}.region{position:absolute;left:34%;top:34%;width:35%;height:30%;background:rgba(56,129,94,.2);border:2px dashed var(--green)}.centre{position:absolute;width:15px;height:15px;border-radius:50%;background:var(--green);left:51%;top:48%;transform:translate(-50%,-50%);box-shadow:0 0 0 6px rgba(56,129,94,.18)}.anno{position:absolute;font-size:12px;font-weight:700}.anno.a{left:20%;top:14%;color:var(--blue)}.anno.b{left:44%;top:56%;color:#fff}.anno.c{left:35%;top:66%;color:var(--green)}.concept-copy h3{font-size:28px;line-height:1.25;letter-spacing:-.03em;margin:0 0 18px}.concept-copy p{color:var(--muted)}.concept-copy .callout{border-left:3px solid var(--blue);padding-left:18px;color:var(--ink);font-weight:650}
.pipeline{position:relative}.pipe-line{position:absolute;left:31px;top:38px;bottom:38px;width:1px;background:var(--line)}.stage{position:relative;display:grid;grid-template-columns:64px 1fr 180px;gap:18px;padding:22px 0;border-bottom:1px solid var(--line)}.stage:last-child{border:0}.stage-no{z-index:1;width:62px;height:62px;border-radius:50%;display:grid;place-items:center;background:var(--paper);border:1px solid var(--line);font-weight:750}.stage h3{margin:4px 0 7px;font-size:21px;letter-spacing:-.025em}.stage p{margin:0;color:var(--muted);max-width:670px}.tag{align-self:center;justify-self:end;padding:6px 10px;border-radius:999px;background:var(--blue-soft);color:var(--blue);font:700 11px ui-monospace,SFMono-Regular,monospace}.stage.optional{margin:20px 0;padding:22px;border:1px dashed #b6aaa1;border-radius:14px;background:var(--orange-soft)}.stage.optional .stage-no{background:var(--orange-soft);border-color:#d3b69f}.stage.optional .tag{background:#fff2e8;color:var(--orange)}
.ml-note{margin:28px 0 48px;padding:24px 28px;background:var(--ink);border-radius:16px;color:#eef1f2;display:grid;grid-template-columns:160px 1fr;gap:20px}.ml-note strong{color:#87b8ee;font-size:13px;letter-spacing:.08em}.ml-note p{margin:0}.loop{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;margin:28px 0}.loop span{padding:10px 15px;border-radius:999px;background:var(--panel);border:1px solid var(--line);font-weight:700;font-size:13px}.arrow{color:var(--muted)}
.result-top{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:36px}.stat{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px}.stat span{font-size:12px;color:var(--muted)}.stat strong{display:block;font-size:34px;letter-spacing:-.05em;margin-top:4px}.stat small{color:var(--green)}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:26px}.chart-title{display:flex;justify-content:space-between;gap:18px;align-items:baseline;margin-bottom:24px}.chart-title h3{margin:0;font-size:20px}.chart-title span{font-size:12px;color:var(--muted)}.group-row{display:grid;grid-template-columns:145px 1fr 68px;gap:12px;align-items:center;margin:13px 0;font-size:13px}.group-bars{display:grid;gap:5px}.thinbar{height:10px;background:#e7e9e9;border-radius:999px;overflow:hidden}.thinbar i{display:block;height:100%;border-radius:999px}.thinbar.greedy i{background:#aeb4b7}.thinbar.final i{background:var(--blue)}.pct{text-align:right;color:var(--green);font-weight:750}.chart-legend{display:flex;gap:18px;font-size:12px;color:var(--muted);margin:18px 0 0 157px}.evolution{margin-top:18px;display:flex;align-items:flex-end;gap:10px;height:230px;padding-top:30px}.evo-col{flex:1;display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end;gap:8px}.evo-bar{width:min(54px,70%);background:var(--blue);border-radius:7px 7px 2px 2px;min-height:8px}.evo-col:first-child .evo-bar{background:#aeb4b7}.evo-value{font-size:12px;font-weight:700}.evo-label{font-size:11px;color:var(--muted);white-space:nowrap}
.proof{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.proof article{padding:22px 20px;border-top:3px solid var(--blue);background:var(--panel)}.proof h3{font-size:18px;margin:0 0 8px}.proof p{font-size:14px;color:var(--muted);margin:0}.foot{padding:38px 0 56px;font-size:12px;color:var(--muted);display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
@media(max-width:820px){.section-head{grid-template-columns:1fr}.section-head p{grid-column:1;margin:0}.compare-grid,.score-grid,.concept{grid-template-columns:1fr}.stage{grid-template-columns:64px 1fr}.stage .tag{grid-column:2;justify-self:start}.ml-note{grid-template-columns:1fr}.result-top,.proof{grid-template-columns:1fr}.legend{margin-left:0}.group-row{grid-template-columns:112px 1fr 58px}.chart-legend{margin-left:124px}}
@media(max-width:520px){.wrap{width:min(100% - 24px,1120px)}.hero{padding:72px 0 58px}section{padding:62px 0}.compare-grid{grid-template-columns:1fr}.score-row{grid-template-columns:78px 1fr 66px;font-size:12px}.score-legend{margin-left:92px}.stage{grid-template-columns:48px 1fr;gap:12px}.stage-no{width:46px;height:46px}.pipe-line{left:23px}.group-row{grid-template-columns:1fr 54px}.group-bars{grid-column:1}.pct{grid-column:2;grid-row:1 / span 2}.evo-label{writing-mode:vertical-rl}.chart{padding:18px}}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
  <div class="eyebrow">ICCAS 2026 · KOH YOUNG AI CONTEST</div>
  <h1>기판을 덮고, 길을 만들고,<br><em>실제 점수만 남긴다</em></h1>
  <p class="lead">수천 개 부품을 가장 적은 촬영 위치로 묶고, 카메라 이동과 검사 코어의 병목을 함께 줄이는 AOI 계획 알고리즘입니다.</p>
  <div class="hero-meta"><span><i class="dot"></i> 768개 독립 회로 검증</span><span>최종 solver · yumin_solver_5</span><span>공식 채점식 그대로 최적화</span></div>
</div></header>

<main>
<section><div class="wrap">
  <div class="section-head"><div class="kicker">01 · 한눈에 보기</div><h2>같은 323개 부품,<br>67장이 16장이 되었다</h2><p>실제 원본 데이터의 같은 기판입니다. 일반 부품은 회색, 옆면 촬영 부품은 파랑, 기준점은 초록으로 표시했습니다.</p></div>
  <div class="compare-grid">
    <article class="board-card"><div class="board-title"><strong>greedy 기본안</strong><div class="board-count">67<small>장</small></div></div><div class="board-wrap"><canvas id="greedyBoard" aria-label="greedy 촬영 위치 67개가 표시된 기판"></canvas></div></article>
    <article class="board-card"><div class="board-title"><strong>yumin_solver_5</strong><div class="board-count" style="color:var(--blue)">16<small>장</small></div></div><div class="board-wrap"><canvas id="finalBoard" aria-label="최종 알고리즘 촬영 위치 16개가 표시된 기판"></canvas></div></article>
  </div>
  <div class="compare-tools"><label class="check"><input id="showFov" type="checkbox" checked> 촬영 범위</label><label class="check"><input id="showRoute" type="checkbox"> 방문 순서</label><div class="legend"><span><i class="swatch" style="background:#8e979b"></i>일반</span><span><i class="swatch" style="background:#2f67a9"></i>옆면</span><span><i class="swatch" style="background:#38815e"></i>기준점</span><span><i class="swatch" style="background:#b86732"></i>촬영 중심</span></div></div>
  <p class="reduction"><strong>−76.1%</strong> 촬영 횟수 · 이 기판의 점수는 127.62에서 74.50으로 감소</p>
</div></section>

<section><div class="wrap">
  <div class="section-head"><div class="kicker">02 · 목적 함수</div><h2>사진 수만 줄여서는<br>좋은 답이 아니다</h2><p>검사 시간, 사용 코어 수, 알고리즘 실행시간을 모두 비용으로 냅니다. 그래서 각 개선 단계는 근사 지표가 아니라 실제 채점식 J로 합격 여부를 결정합니다.</p></div>
  <div class="score-grid">
    <div class="score-stack" id="scoreBars"></div>
    <aside class="formula"><div class="mini">MINIMIZE</div><code>J = CT + 10 × k + RT</code><p><b>CT</b> 검사 완료시간 · <b>k</b> 사용 코어 수 · <b>RT</b> 풀이 실행시간. 768개 회로에서 최종 점수는 greedy보다 <b>54.88%</b> 낮았습니다.</p></aside>
  </div>
</div></section>

<section><div class="wrap">
  <div class="section-head"><div class="kicker">03 · 기하 변환</div><h2>“카메라를 어디에 둘까”를<br>사각형 찌르기로 바꾼다</h2></div>
  <div class="concept"><div class="concept-visual" aria-label="부품, FOV, 가능한 카메라 중심 영역의 관계"><div class="fov-box"></div><div class="part"></div><div class="region"></div><div class="centre"></div><span class="anno a">FOV</span><span class="anno b">부품</span><span class="anno c">가능한 중심 R</span></div><div class="concept-copy"><h3>부품마다 “이 안에 중심을 두면 찍힌다”는 영역 R이 생깁니다.</h3><p>여러 부품의 R이 겹치면, 교집합의 점 하나에서 모두 찍을 수 있습니다. 따라서 연속 좌표 문제는 모든 R을 최소 개수의 점으로 찌르는 <b>minimum piercing</b> 문제로 변환됩니다.</p><p class="callout">옆면 부품은 유효 FOV가 가로·세로 절반이므로 R이 더 작습니다. 커버 수를 좌우하는 핵심 제약입니다.</p></div></div>
</div></section>

<section><div class="wrap">
  <div class="section-head"><div class="kicker">04 · 파이프라인</div><h2>Cover → Route → Refine,<br>좋아질 때만 다음 답이 된다</h2><p>각 단계는 다음 단계의 입력을 바꾸므로 순서가 중요합니다. 최종 solver는 검증된 짧은 예산 안에서 이 루프를 반복합니다.</p></div>
  <div class="ml-note"><strong>ML의 정확한 역할</strong><p>머신러닝이 FOV를 직접 생성하지 않습니다. 부품 수·밀도·Side 비율·검사량 같은 특징으로 회로 유형을 찾고, <b>경로 탐색 시간 배율만 선택</b>합니다.</p></div>
  <div class="pipeline">
    <div class="pipe-line"></div>
    <article class="stage"><div class="stage-no">01</div><div><h3>회로 특징을 읽는다</h3><p>부품 수, 보드 셀 수, Side·대형 부품 비율, 검사시간 분포, 재구성 시간, 최대 코어 수를 12개 특징으로 압축합니다.</p></div><div class="tag">board_features</div></article>
    <article class="stage"><div class="stage-no">02</div><div><h3>탐색 예산을 고른다</h3><p>오프라인에서 검증된 군집 중심과 새 회로를 비교해 0.55·0.70·0.80·0.95 중 경로 예산 배율을 선택합니다.</p></div><div class="tag">select_bscale</div></article>
    <article class="stage"><div class="stage-no">03</div><div><h3>최소 촬영 커버를 만든다</h3><p>Step 레벨별 feasible-centre box를 8개 방향 프레임에서 훑고, merge·dissolve·absorb로 겹치는 촬영을 정리합니다. Fiducial과 대형 부품 타일은 규칙대로 고정합니다.</p></div><div class="tag">build_cover</div></article>
    <article class="stage"><div class="stage-no">04</div><div><h3>실제 점수로 경로를 찾는다</h3><p>POMO 다중 시작, 8배 좌표 증강, 국소 이동을 사용합니다. 후보 순서는 이동거리만이 아니라 촬영·Z 전환·검사 tail·코어 비용까지 포함한 J로 평가합니다.</p></div><div class="tag">route · POMO</div></article>
    <article class="stage"><div class="stage-no">05</div><div><h3>부품을 더 이른 촬영으로 옮긴다</h3><p>호환 FOV로 부품을 재배정하고, 빈 FOV를 지우고, 마지막 촬영의 무거운 작업을 앞쪽으로 당겨 검사 코어의 drain time을 줄입니다.</p></div><div class="tag">refine_plan</div></article>
    <article class="stage"><div class="stage-no">06</div><div><h3>중심과 순서를 다시 다듬는다</h3><p>허용 영역 안에서 FOV 중심을 움직이고 다시 라우팅합니다. 직전 답보다 J가 낮아진 후보만 채택합니다.</p></div><div class="tag">re-route</div></article>
    <article class="stage"><div class="stage-no">07</div><div><h3>긴 작업부터 내보낸다</h3><p>한 촬영 안의 부품은 검사시간이 긴 순서로 기록합니다. 채점기가 이 순서대로 코어에 배정하므로 LPT가 코어 부하를 고르게 만듭니다.</p></div><div class="tag">emit · LPT</div></article>
    <article class="stage optional"><div class="stage-no">＋</div><div><h3>정확 커버와 학습 유도 LNS는 선택 기능</h3><p>작은 독립 영역은 Branch-and-Bound로 정확히 풀고, 큰 영역은 온라인 ranker가 고른 8개 FOV 이웃을 재구성할 수 있습니다. 하지만 실행시간 비용이 개선 이득보다 큰 경우가 있어 공식 점수 경로에서는 기본 비활성화했습니다.</p></div><div class="tag">exact_cover · LNS</div></article>
  </div>
  <div class="loop"><span>Cover</span><b class="arrow">→</b><span>Route</span><b class="arrow">→</b><span>Refine</span><b class="arrow">→</b><span>Re-route</span><b class="arrow">↺ J가 좋아질 때만</b></div>
</div></section>

<section><div class="wrap">
  <div class="section-head"><div class="kicker">05 · 검증 결과</div><h2>768개 회로에서<br>평균이 아니라 반복해서 이겼다</h2><p>원본·추가·학습·검증·홀드아웃·스트레스 seed의 8개 세트, 각 96개 회로를 동일 조건으로 재실행했습니다.</p></div>
  <div class="result-top"><div class="stat"><span>총 회로</span><strong>768</strong><small>8 × 96 boards</small></div><div class="stat"><span>greedy 대비 총점</span><strong>−54.88%</strong><small>234,944 → 106,001</small></div><div class="stat"><span>규칙 위반</span><strong>0</strong><small>모든 세트 유효</small></div></div>
  <div class="chart"><div class="chart-title"><h3>독립 seed별 총점 비교</h3><span>낮을수록 좋음 · 각 행 96회로</span></div><div id="groupChart"></div><div class="chart-legend"><span><i class="swatch" style="background:#aeb4b7"></i>greedy</span><span><i class="swatch" style="background:var(--blue)"></i>yumin_solver_5</span></div></div>
  <div class="chart" style="margin-top:18px"><div class="chart-title"><h3>알고리즘 진화 · 원본 96회로</h3><span>각 버전의 실제 총점</span></div><div class="evolution" id="evolutionChart"></div></div>
</div></section>

<section><div class="wrap">
  <div class="section-head"><div class="kicker">06 · 핵심 판단</div><h2>더 복잡한 알고리즘보다<br>점수에 남는 알고리즘</h2></div>
  <div class="proof"><article><h3>FOV 수가 먼저다</h3><p>촬영은 이동·촬상·검사 release를 동시에 만듭니다. 기하 커버를 먼저 강하게 줄이는 이유입니다.</p></article><article><h3>경로와 부하는 함께 본다</h3><p>가까운 순서가 항상 빠른 완료시간은 아닙니다. 마지막 무거운 부품이 검사 tail을 길게 만들 수 있습니다.</p></article><article><h3>실행시간도 비용이다</h3><p>정확 커버가 더 작은 답을 찾아도 그 계산시간이 점수 개선보다 크면 사용하지 않습니다.</p></article></div>
</div></section>
</main>

<footer><div class="wrap foot"><span>실측 데이터: robustness_results/score_comparison.csv · 2026-09-23</span><span>알고리즘 근거: yumin_solver_4 · yumin_solver_5 · utils/checker.py</span></div></footer>
<script>
const DATA = __REPORT_DATA__;
const css = getComputedStyle(document.documentElement);
const colors = {ink:'#171a1d', line:'#d9dddc', normal:'#8e979b', side:'#2f67a9', fid:'#38815e', fov:'#b86732'};

function drawBoard(canvas, fovs){
  const b=DATA.board, box=canvas.parentElement.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2);
  canvas.width=Math.max(1,Math.floor(box.width*dpr)); canvas.height=Math.max(1,Math.floor(box.height*dpr));
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr); const W=box.width,H=box.height,pad=16;
  const s=Math.min((W-2*pad)/b.width,(H-2*pad)/b.height), ox=(W-b.width*s)/2, oy=(H-b.height*s)/2;
  ctx.fillStyle='#f8f8f5';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#cbd0cf';ctx.strokeRect(ox,oy,b.width*s,b.height*s);
  if(document.getElementById('showFov').checked){
    ctx.fillStyle='rgba(184,103,50,.07)';ctx.strokeStyle='rgba(184,103,50,.45)';ctx.lineWidth=1;
    for(const f of fovs){const fw=(f.side?b.fov_width/2:b.fov_width)*s,fh=(f.side?b.fov_height/2:b.fov_height)*s;ctx.fillRect(ox+f.x*s-fw/2,oy+f.y*s-fh/2,fw,fh);ctx.strokeRect(ox+f.x*s-fw/2,oy+f.y*s-fh/2,fw,fh)}
  }
  for(const c of b.components){ctx.fillStyle=c.type===2?colors.fid:(c.side?colors.side:colors.normal);ctx.fillRect(ox+c.x*s,oy+c.y*s,Math.max(2,c.w*s),Math.max(2,c.h*s))}
  if(document.getElementById('showRoute').checked){ctx.beginPath();fovs.forEach((f,i)=>{const x=ox+f.x*s,y=oy+f.y*s;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle='rgba(47,103,169,.75)';ctx.lineWidth=1.8;ctx.stroke()}
  ctx.strokeStyle=colors.fov;ctx.lineWidth=1.6;
  for(const f of fovs){const x=ox+f.x*s,y=oy+f.y*s;ctx.beginPath();ctx.moveTo(x-4,y);ctx.lineTo(x+4,y);ctx.moveTo(x,y-4);ctx.lineTo(x,y+4);ctx.stroke()}
}
function redraw(){drawBoard(document.getElementById('greedyBoard'),DATA.board.greedy_fovs);drawBoard(document.getElementById('finalBoard'),DATA.board.final_fovs)}
document.getElementById('showFov').addEventListener('change',redraw);document.getElementById('showRoute').addEventListener('change',redraw);new ResizeObserver(redraw).observe(document.querySelector('.compare-grid'));

const totals=DATA.totals,maxScore=totals.greedy.score;
document.getElementById('scoreBars').innerHTML=['greedy','yumin_solver_5'].map(key=>{const d=totals[key],label=key==='greedy'?'greedy':'solver 5';return `<div class="score-row"><span>${label}</span><div class="bar" aria-label="${label} 점수 ${d.score}"><i class="seg ct" style="width:${d.ct/maxScore*100}%"></i><i class="seg core" style="width:${d.core_cost/maxScore*100}%"></i><i class="seg runtime" style="width:${Math.max(.5,d.runtime/maxScore*100)}%"></i></div><strong>${Math.round(d.score).toLocaleString()}</strong></div>`}).join('')+'<div class="score-legend"><span><i class="swatch" style="background:var(--blue)"></i>CT</span><span><i class="swatch" style="background:var(--orange)"></i>10 × cores</span><span><i class="swatch" style="background:var(--green)"></i>runtime</span></div>';

const groupMax=Math.max(...DATA.groups.map(d=>d.greedy));
document.getElementById('groupChart').innerHTML=DATA.groups.map(d=>`<div class="group-row"><span>${d.label}</span><div class="group-bars"><div class="thinbar greedy"><i style="width:${d.greedy/groupMax*100}%"></i></div><div class="thinbar final"><i style="width:${d.final/groupMax*100}%"></i></div></div><span class="pct">−${d.improvement.toFixed(1)}%</span></div>`).join('');
const evoMax=Math.max(...DATA.evolution.map(d=>d.score)),evoMin=Math.min(...DATA.evolution.map(d=>d.score))*0.88;
document.getElementById('evolutionChart').innerHTML=DATA.evolution.map(d=>`<div class="evo-col"><span class="evo-value">${Math.round(d.score).toLocaleString()}</span><div class="evo-bar" style="height:${Math.max(8,(d.score-evoMin)/(evoMax-evoMin)*165)}px"></div><span class="evo-label">${d.label}</span></div>`).join('');
</script>
</body></html>'''


def main() -> None:
    data = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE.replace("__REPORT_DATA__", data)
    (ROOT / "algorithm-visualization.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
