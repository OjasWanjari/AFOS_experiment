"""Audit imported pooled retraining results without training or changing inputs."""
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'paper_full_results/retrain_runs/pooled_top15_updated'
PREVIOUS = ROOT / 'analysis/paper_full_results_updated_seed4'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan = json.loads((RUN / 'plan.json').read_text(encoding='utf-8'))
assert digest(RUN / 'plan.json') == (RUN / 'plan.sha256').read_text().strip()
assert digest(RUN / 'split_indices.npz') == plan['split_sha256']
for name, expected in plan['source_sha256'].items():
    assert digest(RUN / 'source_snapshot' / name) == expected, name
selected = read_csv(RUN / 'selected_top15.csv')
assert selected == read_csv(PREVIOUS / 'pooled_top15.csv')
source_checks = []
for source in plan['sources']:
    seed = source['seed']
    folder = 'paper_full_seeds3 and 4' if seed == 3 else f'paper_full_seeds{seed}'
    raw = ROOT / 'paper_full_results' / folder
    assert digest(raw / 'results' / f'evaluation_results_seed_{seed}.csv') == source['csv_sha256']
    assert digest(raw / 'manifest.json') == source['manifest_sha256']
    assert digest(raw / 'split_indices.npz') == plan['split_sha256']
    source_checks.append({'seed': seed, 'local_source': str(raw.relative_to(ROOT)), 'verified': True})

rows = read_csv(RUN / 'results/retraining_results.csv')
saved_summary = {r['function']: r for r in read_csv(RUN / 'results/retraining_summary.csv')}
planned = {r['trial_id']: r for r in plan['trials']}
assert len(rows) == len(planned) == 105
assert len({r['trial_id'] for r in rows}) == 105
assert {r['trial_id'] for r in rows} == set(planned)
failures, model_count, log_counts, envs = [], 0, Counter(), set()
for row in rows:
    trial = row['trial_id']
    attempt = RUN / 'trials' / trial / row['attempt']
    result = json.loads((attempt / 'result.json').read_text(encoding='utf-8'))
    for key in ['function', 'activation_expression', 'status', 'trial_id']:
        assert row[key] == result[key], (trial, key)
    for key in ['function', 'activation_expression', 'seed', 'chromosome', 'baseline']:
        assert result[key] == planned[trial][key], (trial, key)
    for key in ['test_accuracy', 'test_loss', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy']:
        assert (None if row[key] == '' else float(row[key])) == result.get(key), (trial, key)
    history = read_csv(attempt / 'histories' / f'{trial}.csv')
    assert len(history) == int(row['epochs_completed']) == result['epochs_completed']
    assert [int(h['epoch']) for h in history] == list(range(1, len(history) + 1))
    env = json.loads((attempt / 'environment.json').read_text(encoding='utf-8'))
    assert env['runtime'] == plan['runtime']
    envs.add(json.dumps({'packages': env['packages'], 'runtime': env['runtime']}, sort_keys=True))
    log = (attempt / 'worker.log').read_text(encoding='utf-8', errors='replace')
    for label, pattern in [('oom', r'out of memory|RESOURCE_EXHAUSTED|ResourceExhaustedError|CUDA_ERROR_OUT_OF_MEMORY'),
                           ('traceback', r'Traceback \(most recent call last\)'),
                           ('layout_warning', r'layout failed: INVALID_ARGUMENT')]:
        if re.search(pattern, log, re.I):
            log_counts[label] += 1
    if row['status'] == 'completed':
        assert len(history) == 50
        assert all(math.isfinite(float(h[k])) for h in history for k in ['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        assert math.isfinite(float(row['test_accuracy'])) and math.isfinite(float(row['test_loss']))
        assert row['model_save_status'] == 'saved' and (attempt / 'model.keras').is_file()
        model_count += 1
    else:
        last = history[-1]
        first_bad = next(h for h in history if any(not math.isfinite(float(h[k])) for k in ['loss', 'accuracy', 'val_loss', 'val_accuracy']))
        failures.append({'trial_id': trial, 'function': row['function'], 'seed': int(row['seed']),
                         'status': row['status'], 'epochs_completed': len(history),
                         'first_bad_epoch': int(first_bad['epoch']), 'last_history': last,
                         'history_path': str((attempt / 'histories' / f'{trial}.csv').relative_to(ROOT))})

groups = []
for function, saved in saved_summary.items():
    trials = sorted([r for r in rows if r['function'] == function], key=lambda r: int(r['seed']))
    assert [int(r['seed']) for r in trials] == [0, 1, 2, 3, 4]
    good = [r for r in trials if r['status'] == 'completed']
    scores = [float(r['test_accuracy']) * 100 for r in good]
    complete = len(good) == 5
    assert int(saved['completed']) == len(good)
    assert saved['status'] == ('complete' if complete else 'incomplete')
    if complete:
        assert abs(statistics.mean(scores) / 100 - float(saved['test_accuracy_mean'])) < 1e-12
        assert abs(statistics.stdev(scores) / 100 - float(saved['test_accuracy_std'])) < 1e-12
    else:
        assert saved['test_accuracy_mean'] == saved['test_accuracy_std'] == ''
    groups.append({'function': function, 'expression': trials[0]['activation_expression'],
                   'completed': len(good), 'mean_pct': statistics.mean(scores) if complete else None,
                   'sd_pp': statistics.stdev(scores) if complete else None,
                   'seed_scores_pct': [float(r['test_accuracy']) * 100 if r['status'] == 'completed' else None for r in trials]})

discovered = sorted([g for g in groups if g['function'].startswith('discovered_') and g['completed'] == 5], key=lambda g: -g['mean_pct'])
baseline = [g for g in groups if not g['function'].startswith('discovered_')]
by_name = {g['function']: g for g in groups}
paper = {r['function']: r for r in read_csv(PREVIOUS / 'baseline_paper_comparison.csv')}
legacy = []
for path in sorted((ROOT / 'paper_full_results/runs/results').glob('evaluation_results_seed_*.csv')):
    lr = read_csv(path)
    legacy.append({'file': str(path.relative_to(ROOT)), 'records': len(lr),
                   'max_generation': max(int(r['generation']) for r in lr), 'columns': list(lr[0])})
status = Counter(r['status'] for r in rows)
assert status == json.loads((RUN / 'status.json').read_text())['status_counts']
best = discovered[0]
gelu = by_name['gelu']
relu = by_name['relu']
audit = {'analysis_date': '2026-10-22', 'input_run': str(RUN.relative_to(ROOT)),
         'plan_sha256': digest(RUN / 'plan.json'), 'plan_and_snapshot_verified': True,
         'all_source_hashes_and_split_verified': source_checks, 'updated_top15_exact_match': True,
         'trials': len(rows), 'statuses': dict(status), 'completed_models_present': model_count,
         'unique_process_ids': len({r['pid'] for r in rows}), 'environments': [json.loads(e) for e in envs],
         'worker_log_matches': {k: log_counts[k] for k in ['oom', 'traceback', 'layout_warning']}, 'failures': failures, 'groups': groups,
         'best_function': best['function'], 'best_mean_pct': best['mean_pct'],
         'best_minus_gelu_pp': best['mean_pct'] - gelu['mean_pct'],
         'best_minus_relu_pp': best['mean_pct'] - relu['mean_pct'],
         'complete_candidates_above_gelu_mean': sum(g['mean_pct'] > gelu['mean_pct'] for g in discovered),
         'best_minus_gelu_by_seed_pp': [a-b for a,b in zip(best['seed_scores_pct'],gelu['seed_scores_pct'])],
         'minimum_completed_test_accuracy_pct': min(float(r['test_accuracy'])*100 for r in rows if r['status']=='completed'),
         'legacy_runs_excluded': legacy}
(OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

lines = ['# 新しい統合Top 15の再学習結果', '', '分析日：2026-10-22。追加された保存結果を検査・再集計した。追加学習や元データの変更は行っていない。', '',
         '**105試行すべてに終了結果があり、99試行で50エポックの学習・テスト評価が完了した。6試行はNaNで停止した。発見15候補のうち14候補で5シードがそろい、最高平均テスト精度は83.582±0.460%だった。**', '',
         '## 入力と検証', '',
         '- 対象：`paper_full_results/retrain_runs/pooled_top15_updated`。新seed 4を含む全5探索から選定したTop 15と既存6関数を、各seed 0〜4で評価した。',
         '- `selected_top15.csv`は前回の更新版分析のTop 15と完全一致。計画のハッシュ、保存コード、探索元CSV・manifest、訓練・検証分割のハッシュを照合した。',
         '- 105件の集計行を各試行の`result.json`・学習履歴と照合し、平均・標本標準偏差を再計算した。99件は全50エポックの指標が有限で、保存モデルも99個存在する。',
         '- 105個の異なるプロセスID。実行設定はGPU、JIT無効、memory growth有効。TensorFlow 2.21.0 / Keras 3.12.4。',
         f'- workerログのOOM該当：{log_counts["oom"]}試行、Traceback：{log_counts["traceback"]}試行。layout optimizerの警告は{log_counts["layout_warning"]}試行にあるが、試行終了状態とは区別する。',
         '- `paper_full_results/runs`は別形式の探索記録で、seed 0〜2各50評価・10世代。共通15関数を各5シードで50エポック再学習した結果ではないため合算しない。', '',
         '検査記録：[audit.json](audit.json)。', '', '## 完了状況', '',
         '| 対象 | 予定 | 完了 | 非有限値で停止 |', '|---|---:|---:|---:|',
         '| 発見15候補 | 75 | 74 | 1 |', '| 既存6関数 | 30 | 25 | 5 |', '| 合計 | 105 | 99 | 6 |', '',
         '`all_evaluations_completed=false`は全件成功でないことを示す。未着手・実行中の試行はない。5回そろった候補は0/15から14/15に更新された。', '',
         '## 発見関数の結果', '', '平均±SDは同じ関数の再学習seed 0〜4に対するテスト精度。SDは標本標準偏差。候補番号は探索時の検証精度順位で、下表は再学習後の平均順。', '',
         '| 平均順位 | 探索候補 | 関数 | 完了 | 平均±SD (%) |', '|---:|---|---|---:|---:|']
for i,g in enumerate(discovered,1):
    rank = 1 + sum(other['mean_pct'] > g['mean_pct'] for other in discovered)
    lines.append(f'| {rank} | {g["function"]} | `{g["expression"]}` | 5/5 | {g["mean_pct"]:.3f} ± {g["sd_pp"]:.3f} |')
g=by_name['discovered_13']
lines += [f'| — | {g["function"]} | `{g["expression"]}` | 4/5 | 未完了 |', '',
          f'最高候補は`{best["expression"]}`。seed 0〜4は'+', '.join(f'{x:.2f}%' for x in best['seed_scores_pct'])+'。', '',
          f'完了14候補はすべてReLUの平均を上回り、うち{audit["complete_candidates_above_gelu_mean"]}候補はGELUの平均を上回った。最高候補の差はReLU比+{audit["best_minus_relu_pp"]:.3f}ポイント、最良baselineのGELU比+{audit["best_minus_gelu_pp"]:.3f}ポイント。これは観測差で、有意差検定による結論ではない。', '',
          '候補7と10は数学的に同一だが数式文字列が異なり、別候補として実行されている。15個の数学的に異なる関数と解釈しない。', '',
          '## 既存関数と論文の比較', '', '| 関数 | 今回 平均±SD (%) | 論文 平均±SD (%) | 平均差 (pt) |', '|---|---:|---:|---:|']
for g in baseline:
    ref=paper[g['function']]
    obs=f'{g["mean_pct"]:.3f} ± {g["sd_pp"]:.3f}' if g['completed']==5 else f'未完了 ({g["completed"]}/5)'
    delta=f'{g["mean_pct"]-float(ref["paper_mean_pct"]):+.3f}' if g['completed']==5 else '—'
    lines.append(f'| {g["function"]} | {obs} | {float(ref["paper_mean_pct"]):.2f} ± {float(ref["paper_sd_pp"]):.2f} | {delta} |')
lines += ['', '論文参照：Activation Function Optimization Scheme for Image Classification, arXiv:2409.04915v1, Table 6 Base CNN。両者とも完了関数は5回だが、論文と同じ分割・実行環境まで保証されない。', '',
          '今回最高83.582%は論文EELU-4の82.74±0.14%より数値上0.842ポイント高い。ただし異なる関数・実行環境の比較であり、EELU-1〜4を今回の環境で再学習した結果はない。ReLU・SERF・Mishの論文との差も残るため、論文に対する優位性は未確定。', '',
          '## 失敗した6試行', '', '| 試行 | 最初の非有限値エポック | 訓練損失 | 検証損失 |', '|---|---:|---|---|']
for failure in failures:
    lines.append(f'| {failure["trial_id"]} | {failure["first_bad_epoch"]} | {failure["last_history"]["loss"]} | {failure["last_history"]["val_loss"]} |')
lines += ['', '候補13のseed 1は49エポックまで有限で、50エポック目にNaNになった。50という履歴行数だけで成功と判断してはいけない。ELUは全seedで1エポック目にNaNとなった。いずれもテスト評価は保存されていない。', '']
partial=by_name['discovered_13']['seed_scores_pct']
lines += ['候補13の成功分（seed 0,2,3,4）は'+', '.join(f'{v:.2f}%' for v in partial if v is not None)+'。失敗を除いた4回平均を5回の正式な平均として報告しない。', '',
          f'完了99試行の最低テスト精度は{audit["minimum_completed_test_accuracy_pct"]:.2f}%で、以前の有限値10%崩壊は今回の完了試行にはない。NaNの発生源となった中間活性・勾配は保存されていないため、ELUや候補13の根本原因は未確定。', '',
          '## 次に確認する点', '',
          '1. 候補13 seed 1とELUの失敗原因を、最初に非有限値になる演算・勾配の記録で調べる。条件を変える場合は別実験として区別する。',
          '2. EELU-1〜4を固定比較対象として同じ環境・同じ5シードで評価し、今回の候補と直接比較する。',
          '3. ReLUなどの基準関数が論文より低い理由と、短期探索順位から50エポック評価順位への変化を調べる。', '',
          '## 元ファイル', '',
          '- [試行別結果](../../paper_full_results/retrain_runs/pooled_top15_updated/results/retraining_results.csv)',
          '- [関数別集計](../../paper_full_results/retrain_runs/pooled_top15_updated/results/retraining_summary.csv)',
          '- [実行状態](../../paper_full_results/retrain_runs/pooled_top15_updated/status.json)',
          '- [固定した候補](../../paper_full_results/retrain_runs/pooled_top15_updated/selected_top15.csv)',
          '- [再検査スクリプト](analyze.py)', '']
(OUT / 'README_JA.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({k:audit[k] for k in ['trials','statuses','worker_log_matches','best_function','best_mean_pct','best_minus_gelu_pp','best_minus_relu_pp','complete_candidates_above_gelu_mean','best_minus_gelu_by_seed_pp']}, indent=2))
