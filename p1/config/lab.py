"""Portable local checks and sequential classroom experiments; run from any cwd."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
P1 = ROOT / 'p1'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
FIELDS = {'adamw': 'learning_rate', 'muon': 'muon_learning_rate', 'sgdm': 'sgdm_learning_rate'}
DEFAULT_RATES = {'adamw': 3e-4, 'muon': 1e-2, 'sgdm': .1}


def run(*args):
    print('+', sys.executable, *map(str, args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=P1, check=True)


def write_config(method, kind, rate):
    source = (P1 / 'config/train_shakespeare_char.py').read_text(encoding='utf-8')
    replacements = {'optimizer_name': repr(method), 'out_dir': repr(f'runs/{method}-{kind}'),
                    FIELDS[method]: repr(rate)}
    lines = []
    for line in source.splitlines():
        key = line.split('=', 1)[0].strip()
        lines.append(f'{key} = {replacements[key]}' if key in replacements else line)
    destination = P1 / f'config/{method}-{kind}.py'
    if destination.exists():
        raise FileExistsError(f'Refusing to replace {destination}')
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def completed(folder):
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if summary['status'] != 'completed' or summary['tokens'] != config['max_tokens']:
        raise ValueError(f'Incomplete token budget: {folder}')
    return config, summary


def verify(gpu):
    import torch
    import numpy as np
    import matplotlib
    print(f'Python={sys.version}; torch={torch.__version__}; numpy={np.__version__}; matplotlib={matplotlib.__version__}')
    if not hasattr(torch.optim, 'Muon'):
        raise RuntimeError('Native torch.optim.Muon is required (PyTorch >= 2.9)')
    manifest = json.loads((ROOT / 'upstream-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Upstream data/check file changed: {name}')
    device = 'cuda' if gpu else 'cpu'
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; activate the course environment')
        print('GPU:', torch.cuda.get_device_name(0))
        x = torch.ones((64, 64), device='cuda')
        assert (x @ x).sum().item() == 64**3
        torch.cuda.synchronize()
    run('check_solution.py')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    for method in FIELDS:
        out = f'checks/{stamp}-{device}-{method}'
        run('train.py', 'config/smoke.py', f'--device={device}',
            f'--optimizer_name={method}', f'--out_dir={out}')
        completed(P1 / out)
        run('plot.py', f'--run-dir={out}')
        run('sample.py', f'--checkpoint={out}/last.pt', f'--device={device}', '--tokens=40')
    print('PASS: upstream files, official checks, three smoke runs, plots and checkpoint sampling.')


def train(stage):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Full experiments require the classroom CUDA environment')
    run('check_solution.py')
    kinds = ['default', 'other'] if stage == 'all' else [stage]
    planned = [(method, kind) for kind in kinds for method in FIELDS]
    for method, kind in planned:
        if not (P1 / f'config/{method}-{kind}.py').is_file():
            raise ValueError(f'Missing {method}-{kind} config. Choose other LRs after baseline curves.')
    for method, kind in planned:
        folder = P1 / f'runs/{method}-{kind}'
        if folder.exists():
            config, _ = completed(folder)
            sys.path.insert(0, str(P1))
            from configurator import load_config
            from train import DEFAULTS
            from model import resolve_model_preset
            expected = resolve_model_preset(load_config(DEFAULTS, [str(P1 / f'config/{method}-{kind}.py')]))
            if any(config.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Existing results use another configuration: {folder}')
            print('Already completed:', folder)
        else:
            run('train.py', f'config/{method}-{kind}.py')
        run('plot.py', f'--run-dir={folder}')


def collect():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rows = []
    for method in FIELDS:
        for kind in ('default', 'other'):
            folder = P1 / f'runs/{method}-{kind}'
            config, summary = completed(folder)
            if summary['tokens'] != 92160000:
                raise ValueError(f'Not a full course experiment: {folder}')
            with (folder / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
                metrics = list(csv.DictReader(stream))
            endpoint = metrics[-1]
            if int(endpoint['tokens']) != summary['tokens']:
                raise ValueError(f'Endpoint mismatch: {folder}')
            rows.append(dict(run=folder.name, optimizer=method, peak_lr=config[FIELDS[method]],
                             val_loss=endpoint['val_loss'], train_seconds=summary['train_seconds'],
                             wall_seconds=summary.get('wall_seconds'), tokens=summary['tokens']))
            for ax, field in zip(axes, ('train_loss', 'val_loss')):
                ax.plot([int(r['tokens']) for r in metrics], [float(r[field]) for r in metrics],
                        label=f'{method} LR={config[FIELDS[method]]:g}')
                ax.set(xlabel='Prediction tokens', ylabel='Loss (nats / character)', title=field)
            run('plot.py', f'--run-dir={folder}')
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(P1 / 'runs/six-run-train-validation.png', dpi=180)
    plt.close(fig)
    with (P1 / 'runs/endpoint-table.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run('plot.py', '--runs-dir=runs')
    run('sample.py', '--checkpoint=runs/adamw-default/last.pt', '--tokens=400')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run('export_results.py', '--runs-dir=runs', f'--output=outputs/part1-results-{stamp}.zip')
    print('Add your report PDF to the ZIP. Back up checkpoints separately before reboot.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('verify')
    check.add_argument('--gpu', action='store_true')
    training = sub.add_parser('train')
    training.add_argument('stage', choices=['default', 'other', 'all'])
    choose = sub.add_parser('choose')
    choose.add_argument('optimizer', choices=list(FIELDS))
    choose.add_argument('lr', type=float)
    choose.add_argument('--reason', required=True, help='Observation from the default loss curve')
    sub.add_parser('collect')
    args = parser.parse_args()
    if args.command == 'verify':
        verify(args.gpu)
    elif args.command == 'train':
        train(args.stage)
    elif args.command == 'collect':
        collect()
    else:
        if not math.isfinite(args.lr) or args.lr <= 0 or args.lr == DEFAULT_RATES[args.optimizer]:
            parser.error('Choose a finite positive LR different from the default')
        completed(P1 / f'runs/{args.optimizer}-default')
        run('plot.py', f'--run-dir=runs/{args.optimizer}-default')
        write_config(args.optimizer, 'other', args.lr)
        note = P1 / f'runs/{args.optimizer}-default/lr-choice.txt'
        note.write_text(args.reason + '\n', encoding='utf-8')
        print('Chosen config created. Inspect the baseline curve before running it.')


if __name__ == '__main__':
    main()
