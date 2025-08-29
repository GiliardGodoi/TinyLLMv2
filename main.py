import argparse
import sys
from pathlib import Path
from tiny import CONFIGS
from tiny.params import Params
from tiny.utils import (
    get_tokenizer_function,
    train_and_evaluate

)
from tiny.envinfo import env_report

datasets_options = list(CONFIGS['datasets'].keys())
default = Params(**CONFIGS['defaults'])

def run(params:Params):
    """
    """
    datapath = CONFIGS['datasets'][params.dataset]['filepath']
    datapath = Path(datapath)
    assert datapath.exists()
    params.to_yaml()

    env_report(params.base_folder.parent, with_torch_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description='Run tinyLLM fine tuning'
    )
    parser.add_argument('-d', '--dataset', type=str, choices=datasets_options, required=True)
    # parser.add_argument('--gamma', type=float, default=1)
    # parser.add_argument('--alpha', type=float, default=1)
    # parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--max_steps', type=int, default=default.max_steps)
    parser.add_argument('--eval_steps', type=int, default=default.eval_steps)
    parser.add_argument('--batch_size', type=int, default=default.batch_size)
    parser.add_argument('--optimizer_name', type=str, default=default.optimizer_name)
    parser.add_argument('--lr', type=float, default=default.lr)
    parser.add_argument('--run', type=int, default=default.run)
    parser.add_argument('--from_pretrained', type=str, default=default.from_pretrained)
    parser.add_argument('--max_input_length', type=int, default=default.max_input_length)
    parser.add_argument('--grad_steps', type=int, default=default.grad_steps)
    parser.add_argument('--local_rank', type=int, default=default.local_rank)
    parser.add_argument('--generation_max_length', type=int, default=default.generation_max_length)
    parser.add_argument('--parallelize', action='store_true' if default.parallelize else 'store_false')
    parser.add_argument('--bf16', action='store_true' if default.bf16 else 'store_false')
    parser.add_argument('--no_log', action='store_true' if default.no_log else 'store_false')
    parser.add_argument('--output_rationale', action='store_true' if default.output_rationale else 'store_false')

    args = parser.parse_args()
    params = default.update(**vars(args))
    run(params)
