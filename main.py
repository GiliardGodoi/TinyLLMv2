import argparse
import yaml
from pathlib import Path

DEFAULT_ARGS = yaml.safe_load(Path('.', 'tiny', 'defaults.yaml').read_text())

DATASET_CONFIG = yaml.safe_load(Path('.', 'tiny', 'datasets.yaml').read_text())

def run(args):
    """
    """
    datapath = Path(args.datapath)
    assert datapath.exists()
    print(datapath, datapath.exists())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/flan-t5-large')
    parser.add_argument('--max_input_length', type=int, default=100)
    parser.add_argument('--grad_steps', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')

    args = parser.parse_args()

    if args.dataset in DATASET_CONFIG:
        args.datapath = DATASET_CONFIG[args.dataset].get('filepath', None)

    run(args)