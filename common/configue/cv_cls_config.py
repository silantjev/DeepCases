from typing import Any
from pydantic import BaseModel, Field
from pydantic import field_validator, model_validator
from pydantic import ConfigDict

class TrainParams(BaseModel):
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=64, ge=1, le=1024)
    val_batch_size: int = Field(default=128, ge=1, le=1024)
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")
    load_images: bool = True

class CNNParams(BaseModel):
    image_size: int = Field(gt=0)


class AlexnetParams(CNNParams):
    fc_feat: int = 512
    conv_feats: tuple[int, int, int, int, int, int] | None = None
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator('conv_feats')
    @classmethod
    def validate_model(cls, v):
        if v is not None:
            if len(v) != 6:
                raise ValueError(f'Parameter "conv_feats" should an array, which contains exactly 6 integers')
            for dim in v:
                if dim <= 0:
                    raise ValueError(f'Elements of the array "conv_feats" should be positive integers')
        return v


class ResnetParams(CNNParams):
    frozen: int = 0 # Количество замороженных увовней

class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str
    train_params: TrainParams
    model_params: dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        allowed_models = {'alexnet', 'resnet'}
        v = v.lower()
        if v not in allowed_models:
            raise ValueError(f'Parameter "model" should be one of {allowed_models}')
        return v

    @model_validator(mode='after')
    def validate_model_params(self):  
        assert self.model in self.model_params, f"{self.model=}, {self.model_params=}"
        params = self.model_params[self.model]
        if self.model == 'alexnet':
            params = AlexnetParams(**params)
        else:
            assert self.model == 'resnet'
            params = ResnetParams(**params)
        self.model_params = {self.model: params}
        return self
