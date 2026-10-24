"""Write a Korean result report from completed, unrounded paired benchmarks."""
from pathlib import Path
import json
import pandas as pd


def main():
    root=Path(__file__).parent
    lines=['# 최종 평가 결과','',
           '점수는 `ΣCT + Σ실행시간 + 10 × Σ코어 수`이며 낮을수록 좋다. 아래는 각 96회로 전체를 순차 실행한 2회 평균이다. 공식 대회 비공개 테스트 점수가 아니라 로컬 평가 점수다. 입력 파싱과 출력 CSV 저장 시간을 포함한다.','',
           '| 평가 세트 | solver | 평균 점수 | 1회차 | 2회차 | 평균 CT 합 | 평균 실행시간 합 | 평균 코어 합 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    combined=[];notes=[]
    for label,title in [('practice','기본 96회로'),('holdout','새 seed 91007의 96회로')]:
        folder=root/'results'/label
        df=pd.read_csv(folder/'comparison.csv')
        expected=len(json.loads((folder/'manifest.json').read_text())['boards'])
        assert expected==96
        assert len(df)==96*3*2
        assert df.cover.all() and df.order.all()
        sums=df.groupby(['repeat','solver'])[['ct','solve_s','core','total']].sum()
        mean=sums.groupby('solver').mean()
        for solver in ['yumin_solver_5','yumin_solver_6','yumin_solver_7']:
            row=mean.loc[solver]
            reps=[sums.loc[(i,solver),'total'] for i in range(2)]
            lines.append(f'| {title} | {solver} | {row.total:.6f} | {reps[0]:.6f} | {reps[1]:.6f} | {row.ct:.6f} | {row.solve_s:.6f} | {row.core:.1f} |')
            combined.append(dict(data=label,solver=solver,**row.to_dict(),repeat0=reps[0],repeat1=reps[1]))
        table=df.groupby(['dataset','solver']).total.mean().unstack()
        v7=df[df.solver=='yumin_solver_7']
        diff5=mean.loc['yumin_solver_5','total']-mean.loc['yumin_solver_7','total']
        diff6=mean.loc['yumin_solver_6','total']-mean.loc['yumin_solver_7','total']
        notes.extend(['',f'## {title}','',
            f'- solver_5 대비 {diff5:+.6f}점 절감({diff5/mean.loc["yumin_solver_5","total"]*100:+.6f}%). solver_6 대비 {diff6:+.6f}점 절감. 음수는 악화다.',
            f'- 회로별 2회 평균에서 solver_5보다 낮은 점수: {int((table.yumin_solver_7<table.yumin_solver_5).sum())}/96. solver_6보다 낮은 점수: {int((table.yumin_solver_7<table.yumin_solver_6).sum())}/96.',
            f'- 동일한 초기 해에 대한 추가 단계의 평균 J 절감 합: {v7.groupby("repeat").gain.sum().mean():.6f}; 추가 단계 시간 합: {v7.groupby("repeat").refinement_s.sum().mean():.6f}초.',
            f'- 2회 총 채택: 이동 {int(v7.move_accepted.sum())}, 교환 {int(v7.swap_accepted.sum())}, 병합 {int(v7.merge_accepted.sum())}.',
            f'- 원시 커버리지/순서 검사 및 내부 목적함수와 공식 Checker 점수 일치: {len(df)}/{len(df)} 통과. 패널티 보정 미사용.',
            f'- [회로별 결과](results/{label}/comparison.csv), [실행별 합계](results/{label}/totals.csv), [실행 환경·소스 해시](results/{label}/manifest.json).'])
    lines+=notes+['','## 해석상의 제한','',
        '설정은 별도 validation_seed_41001에서 선택한 뒤 고정했다. 기본 세트와 새로운 seed 91007 결과를 보고 재학습하지 않았다. 학습한 후보 선택기는 전체 solver를 대체하지 않고 기존 solver_4/5 초기 해를 개선하는 추가 단계다. 2회 반복은 시간 변동을 확인하기 위한 최소 점검이며 통계적 우월성의 증명은 아니다. 기존 초기 해도 시간 예산에 의존하므로 다른 컴퓨터나 부하에서 계획과 점수가 달라질 수 있다.',
        '', '전기방사 논문은 공개 PDF 원문을 확인했으며, 메타물질 논문은 공개 초록·서론까지만 확인했다. 구현을 두 논문의 완전한 재현으로 해석하면 안 된다. [논문 분석과 적용 범위](PAPER_ANALYSIS.md)를 참고한다.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    frame=pd.DataFrame(combined);frame.to_csv(root/'results'/'aggregate.csv',index=False)
    print(frame.to_string(index=False))

if __name__=='__main__': main()
