import argparse
from pathlib import Path
from tiny import CONFIGS
from tiny.datasets import (
    ARCDatasetLoader,
    BioASQDatasetLoader,
    OBQADatasetLoader,
    PIQADatasetLoader,
    PubMedQADatasetLoader,
    RiddleDatasetLoader,
)
from tiny.params import Params
from tiny.envinfo import env_report


import os
import shutil
import logging
import torch
from datasets import Dataset
from transformers.trainer_utils import set_seed
from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            BitsAndBytesConfig,
            Seq2SeqTrainingArguments,
        )
from tiny.multi_teachers import (
    MultiTeacherDataCollatorForSeq2Seq,
    MultiTeacherSeq2SeqTrainer
)
from tiny.metrics import compute_metrics_text
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from itertools import product

from tiny.params import BaseParams
from typing import List
from dataclasses import dataclass, field

datasets_options = list(CONFIGS['datasets'].keys())
roles_keys = ['student', 'llama', 't5']

@dataclass
class Params(BaseParams):
    dataset: str = None
    from_pretrained: str = None
    student_weight: float = None
    teachers_weights: List[float] = field(default_factory=list)
    teachers_keys: List[str] = field(default_factory=list)
    max_steps: int = None
    num_train_epochs: int = None
    eval_steps: int = None
    optimizer_name: str = None
    lr: float = None
    run: int = None
    model_id: str = None
    max_input_length: int = None
    batch_size: int = None
    grad_steps: int = None
    local_rank: int = None
    generation_max_length: int = None
    parallelize: bool = None
    bf16: bool = None
    no_log: bool = None
    output_rationale: bool = None
    logging_strategy: str = None

    def __post_init__(self):
        if self.from_pretrained is None:
            self.model_id = None
        else:
            idx = self.from_pretrained.find('/') + 1
            self.model_id = self.from_pretrained[idx:]

        if self.no_log :
            self.logging_strategy = 'no'
        else:
            self.logging_strategy = 'steps'

    def template_folder_name(self):
        return "baselines/{hash_id}"

def get_tokenizer_function(tokenizer, max_length, roles_keys=roles_keys):

    def tokenizer_function(example, roles_keys=roles_keys):

        inputs_keys = ['input_ids', 'attention_mask', 'labels']
        model_inputs = { f'{role}_{key}' : [] for role, key in product(roles_keys, inputs_keys) }

        for role in roles_keys:
            input_enc = tokenizer(
                            example[role]['input'],
                            max_length=max_length,
                            truncation=True,
                            padding='max_length'
                        )

            model_inputs[f'{role}_input_ids'] = input_enc['input_ids']
            model_inputs[f'{role}_attention_mask'] = input_enc['attention_mask']
            # labels need to be text_target
            label_enc = tokenizer(
                            text_target=example[role]['label'],
                            max_length=150,
                            truncation=True,
                            padding='longest'
                        )
            model_inputs[f'{role}_labels'] = label_enc['input_ids']

        return model_inputs

    return tokenizer_function

def train_and_evaluate(params : Params, dataset: Dataset):
    """
    Sets up and runs the training and evaluation process for a sequence-to-sequence model.

    Args:
        args: Training configuration arguments.
        run: Integer representing the current run or seed for reproducibility.
        tokenizer: Tokenizer object for text preprocessing.
        tokenized_datasets: Tokenized datasets for training and evaluation.
        compute_metrics: Function to compute metrics during evaluation.

    This function initializes the model, sets up training arguments, data collator, and trainer,
    and then starts the training process.
    """
    set_seed(params.run)  # Ensure reproducibility

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        params.from_pretrained,
        device_map='auto',
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(params.from_pretrained, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    tokenizer_function = get_tokenizer_function(tokenizer=tokenizer, max_length=params.max_input_length)
    compute_metrics = compute_metrics_text(tokenizer)

    ds_tokenized = dataset.map(tokenizer_function, remove_columns=['student', 'llama', 't5'])
    ds_tokenized

    # Configuration directories for output and logging
    output_dir = params.base_folder
    checkpoints_dir = params.dir_checkpoints()
    logging_dir = None if params.no_log else params.dir_logs()

    # Clear existing checkpoint directory for a fresh start
    if os.path.exists(checkpoints_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(checkpoints_dir)

    # Setup training arguments for the Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        params.base_folder,
        bf16 = params.bf16,
        eval_strategy = 'steps',
        eval_steps = params.eval_steps,
        gradient_accumulation_steps = params.grad_steps,
        generation_max_length = params.generation_max_length,
        learning_rate = params.lr,
        local_rank = params.local_rank,
        logging_dir = logging_dir,
        logging_strategy = params.logging_strategy,
        logging_steps = params.eval_steps,
        max_steps = params.max_steps,
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size = params.batch_size,
        per_device_eval_batch_size = params.batch_size,
        predict_with_generate = True,
        prediction_loss_only = False,
        remove_unused_columns = False,
        save_strategy = 'no',
        save_steps = params.eval_steps,
        seed = params.run,
        report_to='none'
    )

    # Initialize the data collator for handling batching and tokenization
    data_collator = MultiTeacherDataCollatorForSeq2Seq(tokenizer, teachers_keys=params.teachers_keys)

    # Trainer setup with custom arguments for the training process
    trainer_kwargs = {
        'student_weight': params.student_weight,
        'teachers_weights': params.teachers_weights,
        'teachers_keys': params.teachers_keys,
        'output_rationale': params.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': ds_tokenized["train"],
        'eval_dataset': {'validation': ds_tokenized["test"], },
        'data_collator': data_collator,
        'processing_class': tokenizer,
        'compute_metrics': compute_metrics,
    }

    # Initialize and run the trainer
    trainer = MultiTeacherSeq2SeqTrainer(**trainer_kwargs)

    ## train_output = trainer.train() # retorna um namedtuple
    eval_output = trainer.evaluate(ds_tokenized['test'])

    trainer.save_metrics('baseline', eval_output)
    # trainer.save_metrics('train', train_output._asdict())

def run(params:Params):
    """
    """
    datapath = CONFIGS['datasets'][params.dataset]['filepath']
    datapath = Path(datapath)
    assert datapath.exists()
    params.to_yaml()

    if params.dataset == 'obqa':
        dataset_loader = OBQADatasetLoader()
        params.max_input_length = 100
    elif params.dataset == 'arc':
        dataset_loader = ARCDatasetLoader()
        params.max_input_length = 200
    elif params.dataset == 'piqa':
        dataset_loader = PIQADatasetLoader()
        params.max_input_length = 100
    elif params.dataset == 'riddle':
        dataset_loader = RiddleDatasetLoader()
        params.max_input_length = 100
    elif params.dataset == 'pubmedqa':
        dataset_loader = PubMedQADatasetLoader()
        params.max_input_length = 500
    elif params.dataset == 'bioasq':
        dataset_loader = BioASQDatasetLoader()
        params.max_input_length = 500
    else:
        raise ValueError()

    ds = dataset_loader.load_multiteacher_format()
    train_and_evaluate(params, ds)
    env_report(params.base_folder.parent, with_torch_info=True)


if __name__ == "__main__":
    default = Params(**CONFIGS['defaults'])
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
