
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
            TrainerCallback,
        )
from tiny.multi_teachers import (
    MultiTeacherDataCollatorForSeq2Seq,
    MultiTeacherSeq2SeqTrainer
)
from tiny.metrics import compute_metrics_text
from tiny.params import Params
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from itertools import product

roles_keys = ['student', 'llama', 't5']

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
        target_modules=["q", "v", "wi_0", "wi_1", "wo"],
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

    train_output = trainer.train() # retorna um namedtuple
    eval_output = trainer.evaluate(ds_tokenized['test'])

    trainer.save_metrics('eval', eval_output)
    trainer.save_metrics('train', train_output._asdict())