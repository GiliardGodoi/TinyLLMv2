import argparse
import os
import shutil
import logging
import torch
import yaml

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets
)
from itertools import count
from pathlib import Path
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
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
from transformers.trainer_utils import set_seed
from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            BitsAndBytesConfig,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq
        )

DEFAULT_PARAMETERS = yaml.safe_load((Path('configs', 'experiment-004.yaml')).read_text())

import numpy as np
import logging

logger = logging.getLogger('transformers')

def compute_metrics_text(tokenizer):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        logger.info(f'compute_metrics: raw inputs {predictions.shape=}| {labels.shape=}')
        max_len = max(predictions.shape[-1], labels.shape[-1])

        preds_padded = np.pad(
                        predictions,
                        ((0, 0), (0, max_len - predictions.shape[-1])),
                        constant_values=tokenizer.pad_token_type_id
                    )

        labels_padded = np.pad(
                        labels,
                        ((0, 0), (0, max_len - labels.shape[-1])),
                        constant_values=tokenizer.pad_token_type_id
        )

        token_acc = (preds_padded == labels_padded).mean()

        if preds_padded.shape != labels_padded.shape:
            logger.info(f'compute_metrics: padded {preds_padded.shape=}| {labels_padded.shape=}')

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'token_accuracy': token_acc, 'accuracy' : acc }

    return compute_metrics

def get_data_processing_function(tokenizer, max_length):

    def data_processing_fn(example):
        inputs = tokenizer(
            example['input'],
            text_target=example['label'],
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        return inputs

    return data_processing_fn

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
    pass
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

    lora_config = LoraConfig(
        r=params.lora_rank,
        lora_alpha=32,
        target_modules=["q", "v", "wi_0", "wi_1", "wo"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

    logger.debug(model.print_trainable_parameters())

    tokenizer = AutoTokenizer.from_pretrained(params.from_pretrained, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_processing_fn = get_data_processing_function(tokenizer=tokenizer, max_length=params.max_input_length)
    compute_metrics = compute_metrics_text(tokenizer)

    ds_tokenized = dataset.map(data_processing_fn, remove_columns=['input', 'label']) # 'id',
    print(ds_tokenized)

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
        report_to='tensorboard'
    )

    # # Initialize the data collator for handling batching and tokenization
    # data_collator = MultiTeacherDataCollatorForSeq2Seq(tokenizer, teachers_keys=params.teachers_keys)
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # # Trainer setup with custom arguments for the training process
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': ds_tokenized["train"],
        'eval_dataset': {'validation': ds_tokenized["valid"], },
        'data_collator': data_collator,
        'processing_class': tokenizer,
        'compute_metrics': compute_metrics,
    }

    # # Initialize and run the trainer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    train_output = trainer.train() # retorna um namedtuple
    valid_output = trainer.evaluate(ds_tokenized["valid"])
    eval_output  = trainer.evaluate(ds_tokenized['test'])

    trainer.save_metrics('eval', eval_output)
    trainer.save_metrics('valid', valid_output)
    trainer.save_metrics('train', train_output._asdict())

def run(params : Params):
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
    # id_counter = count()
    # ds = ds.map(lambda _: {'id' : next(id_counter)})

    def extract_nested_features(example, key):
        return {
            'input' : example[key]['input'],
            'label' : example[key]['label'],
            # 'source': key,
            # 'id' : example['id']
        }

    def extract_student_features(example, key='student'):
        output = {
            'input' : example[key]['input'],
            'label' : example[key]['label'],
        }
        # if 'id' in example: output['id'] = example['id']
        # if 'source' in example: output['source'] = example['source']
        return output

    keys = ['llama', 't5', 'student']
    def unpack_training_dataset(ds):
        extracted_ds = [ ds['train'].map(extract_nested_features, fn_kwargs={'key' : k})
                        for k in keys
                    ]
        new_ds = concatenate_datasets(extracted_ds)
        return new_ds.remove_columns(keys)

    ds = DatasetDict({
        'train' : unpack_training_dataset(ds),
        'test'  : ds['test'].map(extract_student_features, remove_columns=keys),
        'valid' : ds['valid'].map(extract_student_features, remove_columns=keys)
    })

    ds = ds.filter(lambda example : len(example['input']) > 50 )

    train_and_evaluate(params, ds)
    env_report(params.base_folder.parent, with_torch_info=True)


if __name__ == "__main__":
    defaults = Params(**DEFAULT_PARAMETERS['parameters'])
    datasets_options = list(CONFIGS['datasets'].keys())

    parser = argparse.ArgumentParser(
        prog="main.py",
        description='Run tinyLLM fine tuning'
    )

    parser.add_argument('-d', '--dataset', type=str, choices=datasets_options, required=True)
    parser.add_argument('--max_steps', type=int, default=defaults.max_steps)
    parser.add_argument('--eval_steps', type=int, default=defaults.eval_steps)
    parser.add_argument('--batch_size', type=int, default=defaults.batch_size)
    parser.add_argument('--optimizer_name', type=str, default=defaults.optimizer_name)
    parser.add_argument('--lr', type=float, default=defaults.lr)
    parser.add_argument('--lora_rank', type=int, default=defaults.lora_rank)
    parser.add_argument('--run', type=int, default=defaults.run)
    parser.add_argument('--from_pretrained', type=str, default=defaults.from_pretrained)
    parser.add_argument('--max_input_length', type=int, default=defaults.max_input_length)
    parser.add_argument('--grad_steps', type=int, default=defaults.grad_steps)
    parser.add_argument('--local_rank', type=int, default=defaults.local_rank)
    parser.add_argument('--generation_max_length', type=int, default=defaults.generation_max_length)
    parser.add_argument('--parallelize', action='store_true' if defaults.parallelize else 'store_false')
    parser.add_argument('--bf16', action='store_true' if defaults.bf16 else 'store_false')
    parser.add_argument('--no_log', action='store_true' if defaults.no_log else 'store_false')
    parser.add_argument('--output_rationale', action='store_true' if defaults.output_rationale else 'store_false')

    args = parser.parse_args()
    params = defaults.update(**vars(args))
    run(params)
