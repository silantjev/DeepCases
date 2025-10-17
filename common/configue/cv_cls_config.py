from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator

class Params(BaseModel):
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=64, ge=1, le=1024)
    val_batch_size: int = Field(default=128, ge=1, le=1024)
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")
    load_images: bool = True

class CNNParams(BaseModel):
    image_size: int
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class AlexnetParams(CNNParams):
    fc_feat: int = 512
    conv_feats: tuple[int] | None = None

class ResnetParams(CNNParams):
    frozen: int = 5

class Config(BaseModel):
    model: str
    train_params: TrainParams = Field(default_factory=TrainParams)
    model_params: dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        allowed_models = {'alexnat', 'resnet'}
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
