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