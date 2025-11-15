import hashlib
import json
import yaml
from collections.abc import Iterable
from dataclasses import (
    asdict,
    dataclass,
    field,
    fields
)
from datetime import datetime
from pathlib import Path
from typing import List

def choices(values):
    if not isinstance(values, Iterable):
        raise TypeError(f'Values options must e a iterable: got {type(values)}')
    return field(default_factory=lambda: values)

@dataclass
class BaseParams:

    _hash_id : str = field(default=None, init=False, repr=False, hash=False)
    _timestamp: str = field(default_factory=lambda: datetime.now().isoformat(), repr=False)

    def _check_attrs_names(self, **kwargs):

        valid_fields = {f.name for f in fields(self)}
        invalid_args = set(kwargs) - valid_fields
        if invalid_args:
            raise TypeError(
                f"Invalid argument(s) for {self.__class__.__name__}:"
                f"{invalid_args}"
                f"\n\tIt is not allowed to instantiate an object with an argument not defined in parameter class."
            )

    def update(self, **kwargs):
        """Cria uma nova instância com valores atualizados"""
        self._check_attrs_names(**kwargs)
        base_dict = self.to_dict()
        base_dict.update(kwargs)
        return self.__class__(**base_dict)

    # -------------------------
    # Leitura de arquivo
    # -------------------------
    @classmethod
    def from_dict(cls, config : dict) :
        return cls(**config)

    @classmethod
    def from_json(cls, filepath:'str'):
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)

    @classmethod
    def from_yaml(cls, filepath:'str'):
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    # -------------------------
    # Conversão
    # -------------------------
    def to_dict(self):
        return { f.name : getattr(self, f.name) for f in fields(self) if f.hash != False }

    def to_yaml(self):
        filepath = self.get_default_folder() / 'parameters.yaml'
        with open(filepath, 'w') as f:
            txt = yaml.dump(asdict(self))
            f.write(txt)
        return filepath

    def to_json(self):
        filepath = self.get_default_folder() / 'parameters.json'
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, sort_keys=True)
        return filepath

    # -------------------------
    # Hash dos argumentos
    # -------------------------
    def make_hash_id(self):
        args_dict = self.to_dict()
        args_txt = json.dumps(args_dict, sort_keys=True)
        digest = hashlib.md5(args_txt.encode()).hexdigest()
        return digest

    @property
    def hash_id(self):
        if self._hash_id is None:
            self._hash_id = self.make_hash_id()
        return self._hash_id

    # -------------------------
    # Diretório para os argumentos
    # -------------------------
    def template_folder_name(self):
        raise NotImplementedError()
        # return "{hassh_id}"

    def get_base_folder_name_from_arguments(self):
        template = self.template_folder_name()
        args = asdict(self)
        args['hash_id'] = self.hash_id
        return Path(template.format(**args))

    def get_default_folder(self, key=None, ensure_exists=True):
        if key is None:
            folder = self.get_base_folder_name_from_arguments()
        else:
            folder = self.get_base_folder_name_from_arguments() / key
        if not folder.exists() and ensure_exists:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def base_folder(self):
        return self.get_base_folder_name_from_arguments()

    # -------------------------
    # Pastas específicas
    # -------------------------
    def dir_checkpoints(self, ensure_exists=True):
        return self.get_default_folder('checkpoints', ensure_exists)

    def dir_logs(self, ensure_exists=True):
        return self.get_default_folder('logs', ensure_exists)

    def dir_metrics(self, ensure_exists=True):
        return self.get_default_folder('metrics', ensure_exists)

    def dir_predictions(self, ensure_exists=True):
        return self.get_default_folder('predictions', ensure_exists)

    def dir_images(self, ensure_exists=True):
        return self.get_default_folder('images', ensure_exists)

    def dir_results(self, ensure_exists=True):
        return self.get_default_folder('results', ensure_exists)

@dataclass
class Params(BaseParams):
    experiment : str = 'testing'
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
    lora_rank:int = None

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
        return "outputs/{experiment}/{hash_id}"