"""Exercise the real process boundary with cheap deterministic training doubles."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_paper

ROOT = Path(__file__).resolve().parents[1]
WORKER = r'''
import os, sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
sys.path.insert(0, sys.argv.pop(1))
import run_paper
tf = ModuleType('tensorflow')
visible = ['test GPU']
tf.config = SimpleNamespace(
    set_visible_devices=lambda devices, kind: visible.clear(),
    get_visible_devices=lambda kind=None: visible)
current_seed = [0]
tf.keras = SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None),
    utils=SimpleNamespace(set_random_seed=lambda n: current_seed.__setitem__(0, n)))
model = ModuleType('paper_afos.model')
class FakeModel:
    def save(self, path, **kwargs): Path(path).write_text('model', encoding='utf-8')
    def evaluate(self, *args, **kwargs): return 0.5, 0.8
    def summary(self, print_fn): print_fn('test model')
model.build_model = lambda *args, **kwargs: FakeModel()
model.load_data = lambda *args, **kwargs: ((None, None), (None, None), (None, None), None)
def train(*args, **kwargs):
    if os.environ.get('AFOS_TEST_FAIL') == '1':
        raise RuntimeError('simulated OOM')
    score = 1 + current_seed[0] % 100 / 1000
    return dict(train_loss=0.5, train_accuracy=0.8, validation_loss=0.7 / score,
                validation_accuracy=0.7, epochs_completed=1, status='completed'), []
model.train_candidate = train
sys.modules['tensorflow'] = tf
sys.modules['paper_afos.model'] = model
def plots(root):
    (root / 'plots').mkdir(exist_ok=True)
    (root / 'plots' / 'test.png').write_bytes(b'test')
run_paper.plots = plots
run_paper.main()
'''


class SeedProcessTests(unittest.TestCase):
    def config(self):
        config = run_paper.load_config(ROOT / 'paper_parameters_adaptive.json')
        config['search'].update(seeds=[2, 3, 4], replications=3, generations=4,
                                population_size=6, selected_parents=2,
                                max_crossover_children=2, max_mutation_children=1,
                                minimum_random_individuals=1)
        config['post_search'].update(top_functions=1, seeds=[2, 3], replications=2, baselines=[])
        return config

    def setup_files(self, directory):
        root = Path(directory)
        fixture = root / 'fake_worker.py'
        fixture.write_text(WORKER, encoding='utf-8')
        config = root / 'config.json'
        config.write_text(json.dumps(self.config()), encoding='utf-8')
        return root, fixture, config

    def launcher(self, fixture, starts, fail_call=None):
        real_popen = subprocess.Popen
        def launch(command):
            import os
            starts.append(command)
            env = dict(os.environ, AFOS_TEST_FAIL='1' if len(starts) == fail_call else '0')
            # Run main() in a genuinely fresh process, replacing only TensorFlow/training.
            return real_popen([command[0], '-u', str(fixture), str(ROOT), *command[3:]],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return launch

    def test_three_seeds_separate_processes_then_pooled_retraining(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root, fixture, config = self.setup_files(directory)
            target = root / 'run with spaces'
            starts = []
            with patch('run_paper.subprocess.Popen', side_effect=self.launcher(fixture, starts)):
                run_paper.main(['--config', str(config), '--output', str(target), '--device', 'cpu', '--quiet'])
            state = json.loads((target / 'checkpoint.json').read_text())
            self.assertEqual([c[c.index('--_worker') + 1] for c in starts],
                             ['search', 'search', 'search', 'retrain', 'retrain', 'finalize'])
            self.assertEqual({r['seed'] for r in state['evaluations']}, {2, 3, 4})
            self.assertEqual(len(state['evaluations']), 72)
            self.assertEqual(len(state['generations']), 12)
            self.assertEqual(len(state['retraining']), 2)
            self.assertTrue(state['retraining_complete'])
            from paper_afos.search import select_top
            self.assertEqual(state['selected'], select_top(state['evaluations'], 1))
            from paper_afos.recording import read_results
            events = read_results(target / 'results/events.csv')
            pids = [r['detail']['pid'] for r in events if r['event'] == 'worker_started']
            self.assertEqual(len(set(pids)), 6)
            self.assertNotIn('tensorflow', sys.modules)
            self.assertEqual(json.loads((target / 'status.json').read_text())['status'], 'completed')
            starts.clear()
            with patch('run_paper.subprocess.Popen', side_effect=self.launcher(fixture, starts)):
                run_paper.main(['--config', str(config), '--resume', str(target), '--device', 'cpu'])
            self.assertEqual(len(starts), 1)  # Finalization only, no repeated training.
            self.assertEqual(json.loads((target / 'checkpoint.json').read_text()), state)

    def test_failure_stops_queue_and_resume_keeps_completed_seed(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root, fixture, config = self.setup_files(directory)
            target = root / 'run'
            starts = []
            with patch('run_paper.subprocess.Popen', side_effect=self.launcher(fixture, starts, fail_call=2)):
                with self.assertRaisesRegex(RuntimeError, 'Worker exited'):
                    run_paper.main(['--config', str(config), '--output', str(target), '--stage', 'search'])
            self.assertEqual(len(starts), 2)
            prior = json.loads((target / 'checkpoint.json').read_text())
            self.assertEqual(prior['seed_index'], 1)
            self.assertEqual(len(prior['evaluations']), 24)
            self.assertEqual(json.loads((target / 'status.json').read_text())['status'], 'failed')
            starts.clear()
            with patch('run_paper.subprocess.Popen', side_effect=self.launcher(fixture, starts)):
                run_paper.main(['--config', str(config), '--resume', str(target), '--stage', 'search'])
            self.assertEqual(len(starts), 3)  # Remaining two seeds, then plots.
            state = json.loads((target / 'checkpoint.json').read_text())
            self.assertEqual(state['evaluations'][:24], prior['evaluations'])
            self.assertEqual(len(state['evaluations']), 72)
            self.assertTrue(state['search_complete'])
            self.assertFalse(state['retraining_complete'])

    def test_interrupt_reaps_worker(self):
        from unittest.mock import Mock
        child = Mock()
        child.wait.side_effect = [KeyboardInterrupt, subprocess.TimeoutExpired('worker', 10), 1]
        child.poll.return_value = None
        with patch('run_paper.subprocess.Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                run_paper.wait_for_worker(['worker'])
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 3)

    def test_existing_output_is_not_touched(self):
        with tempfile.TemporaryDirectory() as directory, patch('run_paper.subprocess.Popen') as spawn:
            with self.assertRaises(FileExistsError):
                run_paper.main(['--output', directory, '--config', str(ROOT / 'paper_parameters_adaptive.json')])
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
