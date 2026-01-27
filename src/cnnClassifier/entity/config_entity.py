from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
 root_dir:Path
 source_URL:str
 local_data_file:Path
 unzip_dir:Path 


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    image_size: int
    learning_rate: float
    pretrained: bool
    channel: int
    classes: int
    batch_size: int
    epochs: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    training_data: Path
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    params_batch_size: int    
    num_classes: int  


