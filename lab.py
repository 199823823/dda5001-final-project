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


def validate_experiment_config(config, reference, method, kind):
    """Preserve the course comparison, including Muon's auxiliary AdamW LR."""
    field = FIELDS[method]
    allowed = {'optimizer_name', 'out_dir', field}
    changed = sorted(k for k in set(config) | set(reference)
                     if k not in allowed and config.get(k) != reference.get(k))
    if changed:
        raise ValueError(f'{method}-{kind}: fixed settings changed: {changed}')
    if config.get('optimizer_name') != method or config.get('out_dir') != f'runs/{method}-{kind}':
        raise ValueError(f'{method}-{kind}: incorrect optimizer or output directory')
    rate = config.get(field)
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError(f'{method}-{kind}: invalid learning rate')
    if (kind == 'default' and rate != reference[field]) or (kind == 'other' and rate == reference[field]):
        raise ValueError(f'{method}-{kind}: learning rate does not match experiment kind')


def audit_experiments(require_others=False):
    sys.path.insert(0, str(P1))
    from configurator import load_config
    from train import DEFAULTS
    reference_path = P1 / 'config/train_shakespeare_char.py'
    manifest = json.loads((ROOT / 'upstream-manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != manifest['sha256']['p1/config/train_shakespeare_char.py']:
        raise ValueError('The official reference configuration changed; restore it before comparing runs.')
    reference = load_config(DEFAULTS, [str(reference_path)])
    plans = {}
    for method in FIELDS:
        for kind in ('default', 'other'):
            path = P1 / f'config/{method}-{kind}.py'
            if kind == 'other' and not path.exists() and not require_others:
                continue
            if not path.is_file():
                raise ValueError(f'Missing config: {path.name}; choose the other LR after baseline curves.')
            config = load_config(DEFAULTS, [str(path)])
            validate_experiment_config(config, reference, method, kind)
            plans[method, kind] = config
            print(f'AUDIT PASS: {method}-{kind}, peak LR={config[FIELDS[method]]:g}')
    return plans


def endpoint_statistics(config, summary, metrics):
    if not metrics or summary.get('status') != 'completed':
        raise ValueError('Only completed experiments with recorded metrics can be summarized')
    endpoint = metrics[-1]
    per_step = config['batch_size'] * config['block_size'] * config['gradient_accumulation_steps']
    if (summary['tokens'] != config['max_tokens'] or summary['tokens'] != summary['step'] * per_step
            or int(endpoint['tokens']) != summary['tokens'] or int(endpoint['step']) != summary['step']):
        raise ValueError('Endpoint metrics, completed updates and token budget must agree')
    train_loss, val_loss = float(endpoint['train_loss']), float(endpoint['val_loss'])
    losses = [float(row['val_loss']) for row in metrics]
    if not all(math.isfinite(value) for value in [train_loss, val_loss, *losses]):
        raise ValueError('Cannot summarize non-finite losses')
    best = min(losses)
    best_row = metrics[losses.index(best)]
    peak = summary.get('peak_memory_bytes')
    return dict(train_loss=train_loss, val_loss=val_loss, generalization_gap=val_loss-train_loss,
                best_val_loss=best, endpoint_minus_best=val_loss-best,
                best_recorded_step=int(best_row['step']), step=summary['step'],
                train_seconds=summary['train_seconds'], wall_seconds=summary.get('wall_seconds'),
                peak_memory_gib=peak / 1024**3 if peak is not None else None, tokens=summary['tokens'])


def verify(gpu):
    import torch
    import numpy as np
    import matplotlib
    print(f'Python={sys.version}; torch={torch.__version__}; numpy={np.__version__}; matplotlib={matplotlib.__version__}')
    if not hasattr(torch.optim, 'Muon'):
        raise RuntimeError('Native torch.optim.Muon is required (PyTorch >= 2.9)')
    audit_experiments()
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
    audit_experiments(require_others=stage in ('other', 'all'))
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
            if any(config.get(k) != v for k, v in expected.items() if k not in ('data_dir', 'resume')):
                raise ValueError(f'Existing results use another configuration: {folder}')
            print('Already completed:', folder)
        else:
            run('train.py', f'config/{method}-{kind}.py')
        run('plot.py', f'--run-dir={folder}')


def collect():
    plans = audit_experiments(require_others=True)
    from model import resolve_model_preset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rows = []
    for method in FIELDS:
        for kind in ('default', 'other'):
            folder = P1 / f'runs/{method}-{kind}'
            config, summary = completed(folder)
            expected = resolve_model_preset(plans[method, kind])
            # Absolute dataset paths can change when results move between machines.
            changed = sorted(k for k, v in expected.items() if k not in ('data_dir', 'resume') and config.get(k) != v)
            if changed:
                raise ValueError(f'{folder.name}: recorded settings differ from the plan: {changed}')
            if summary['tokens'] != 92160000:
                raise ValueError(f'Not a full course experiment: {folder}')
            with (folder / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
                metrics = list(csv.DictReader(stream))
            rows.append(dict(run=folder.name, optimizer=method, peak_lr=config[FIELDS[method]],
                             **endpoint_statistics(config, summary, metrics)))
            for ax, field in zip(axes, ('train_loss', 'val_loss')):
                ax.plot([int(r['tokens']) for r in metrics], [float(r[field]) for r in metrics],
                        color={'adamw':'tab:blue', 'muon':'tab:orange', 'sgdm':'tab:green'}[method],
                        linestyle='-' if kind == 'default' else '--',
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
    sub.add_parser('audit', help='Check fixed settings and optimizer-specific learning rates')
    args = parser.parse_args()
    if args.command == 'verify':
        verify(args.gpu)
    elif args.command == 'train':
        train(args.stage)
    elif args.command == 'collect':
        collect()
    elif args.command == 'audit':
        audit_experiments()
    else:
        audit_experiments()
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
