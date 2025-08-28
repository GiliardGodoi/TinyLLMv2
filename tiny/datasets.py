import re
import json
from datasets import Dataset, DatasetDict, load_dataset
from .tiny import CONFIGS

# DATASET_ROOT = 'datasets'
DATASET_ROOT = CONFIGS['root']


def transform(item):

    question = item['input']

    student = {
        'input' : f'Predict: {question}',
        'label' : item['label']
    }

    llama = {
        'input' : f'Rationale: {question}',
        'label' : item['llama']
    }

    t5 = {
        'input' : f'Explain: {question}',
        'label' : item['t5']
    }

    return { 'student' : student, 'llama' : llama, 't5' : t5 }

class DatasetLoader:
    """Base class for loading and processing specific datasets."""

    def __init__(
            self,
            dataset_name,
            has_valid,
            split_map,
            batch_size,
            train_batch_idxs,
            test_batch_idxs,
            valid_batch_idxs=None
        ):
        """
        Initializes the dataset loader.

        Args:
            dataset_name (str): Name of the dataset.
            has_valid (bool): Indicates if there's a validation set.
            split_map (dict): Maps dataset split names to their identifiers.
            batch_size (int): Size of each data batch.
            train_batch_idxs (list): Batch indices for training data.
            test_batch_idxs (list): Batch indices for test data.
            valid_batch_idxs (list, optional): Batch indices for validation data, if available.
        """
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.has_valid = has_valid
        self.split_map = split_map
        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        assert self.split_map is not None, "Split map cannot be None."

    def load_from_source(self):
        """Loads the dataset directly from its source."""
        datasets = load_dataset(self.dataset_name)
        return datasets

    def to_json(self, datasets):
        """Exports datasets to JSON files according to the split map."""
        for split_name, split_id in self.split_map.items():
            file_path = f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{split_name}.json'
            datasets[split_id].to_json(file_path)

    def load_from_json(self):
        """Loads the dataset from pre-exported JSON files."""
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }
        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json', })
        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets)
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx * self.batch_size, (idx + 1) * self.batch_size)
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])
        return datasets

    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        llamarationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)
            for output in outputs:
                rationale, label, llamarationale = self._parse_llm_output(output)
                rationales.append(rationale)
                labels.append(label)
                llamarationales.append(llamarationale)
        return rationales, labels, llamarationales

    def load_multiteacher_format(self, teachers = ['llama', 't5']):

        ds = self.load_from_json()
        train_teacher_rationales, _, train_second_teacher_rationales = self.load_llm_preds(split='train')
        test_teacher_rationales, _, test_second_teacher_rationales = self.load_llm_preds(split='test')

        ds['train'] = ds['train'].add_column('llama', train_teacher_rationales)
        ds['train'] = ds['train'].add_column('t5', train_second_teacher_rationales)
        ds['test'] = ds['test'].add_column('llama', test_teacher_rationales)
        ds['test'] = ds['test'].add_column('t5', test_second_teacher_rationales)

        if self.has_valid:
            valid_teacher_rationales, _, valid_second_teacher_rationales = self.load_llm_preds(split='valid')
            ds['valid'] = ds['valid'].add_column('llama', valid_teacher_rationales)
            ds['valid'] = ds['valid'].add_column('t5', valid_second_teacher_rationales)
        else:
            train_valid_ds = ds['train'].train_test_split(test_size=0.1, seed=0)
            ds = DatasetDict({
                'train' : train_valid_ds['train'],
                'valid' : train_valid_ds['test'],
                'test' : ds['test']
            })

        ds = ds.map(transform, remove_columns=['input', 'label'])

        return ds

class OBQADatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'obqa'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(10)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class ARCDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'arc'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(3)
        test_batch_idxs = range(3)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class PIQADatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'piqa'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(33)
        test_batch_idxs = range(2)
        valid_batch_idxs = range(2)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class RiddleDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'riddle'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(8)
        test_batch_idxs = range(2)
        valid_batch_idxs = range(2)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]
            c_4 = example['choices'][4]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}\n(e) {c_4}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class PubMedQADatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'pubmedqa'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(1)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class BioASQDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'bioasq'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}'

            example['input'] = input
            example['label'] = example['answer']
            print("example['label']")
            print(example['label'])
            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale