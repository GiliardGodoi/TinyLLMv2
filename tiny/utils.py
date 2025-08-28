
import os
import shutil
import logging
from transformers import Seq2SeqTrainingArguments
from transformers import T5ForConditionalGeneration
from transformers.trainer_utils import set_seed
from tiny.multi_teachers import MultiTeacherDataCollator, MultiTeacherTrainer
from itertools import product

roles_keys = ['student', 'llama', 't5']

def get_tokenizer_function(tokenizer, max_length):

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

def train_and_evaluate(params, run, tokenizer, ds_tokenized, compute_metrics):
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
    set_seed(run)  # Ensure reproducibility

    model = T5ForConditionalGeneration.from_pretrained(params.from_pretrained)

    # Configuration directories for output and logging
    config_dir = params.base_folder
    output_dir = params.dir_checkpoints()
    logging_dir = params.dir_logs()

    # Adjust logging strategy based on arguments
    if params.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # Clear existing checkpoint directory for a fresh start
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    # Setup training arguments for the Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns=False,
        evaluation_strategy='steps',
        eval_steps=params.eval_steps,
        save_strategy='no',
        save_steps=params.eval_steps,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=params.eval_steps,
        max_steps=params.max_steps,
        learning_rate=params.lr,
        gradient_accumulation_steps=params.grad_steps,
        per_device_train_batch_size=params.batch_size,
        per_device_eval_batch_size=params.batch_size,
        predict_with_generate=True,
        seed=run,
        local_rank=params.local_rank,
        bf16=params.bf16,
        generation_max_length=params.gen_max_len,
        prediction_loss_only=False,
    )

    # Initialize the data collator for handling batching and tokenization
    data_collator = MultiTeacherDataCollator(tokenizer=tokenizer, model=model)

    # Trainer setup with custom arguments for the training process
    trainer_kwargs = {
        'output_rationale': params.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': ds_tokenized["train"],
        'eval_dataset': {'test': ds_tokenized["test"], },
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }

    # Initialize and run the trainer
    trainer = MultiTeacherTrainer(**trainer_kwargs)

    trainer.train()