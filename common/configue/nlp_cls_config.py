from typing import Any
from pydantic import BaseModel, Field
from pydantic import field_validator, model_validator
from pydantic import ConfigDict


class TrainParams(BaseModel):
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=64, ge=1, le=1024)
    val_batch_size: int = Field(default=128, ge=1, le=1024)
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")

class NetParams(BaseModel):
    heads: int = Field(ge=0)
    layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    dropout_lin: float = Field(default=0.0, ge=0.0, le=1.0)

class RNNParams(NetParams):
    net_class: str
    bidirectional: bool = False
    hidden_dim: int = Field(default=128, ge=1)

    @field_validator('net_class')
    @classmethod
    def validate_class(cls, v):
        v = v.upper()
        allowed_classes = {'RNN', 'GRU', 'LSTM'}
        if v not in allowed_classes:
            raise ValueError(f'RNN-model class "net_class" should be one of {allowed_classes}')
        return v

class TransformerParams(NetParams):
    feedforward_dim: int = Field(default=1200, ge=1)

class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str
    train_params: TrainParams = Field(default_factory=TrainParams)
    model_params: dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        allowed_models = ['rnn', 'transformer']
        v = v.lower()
        if v not in allowed_models:
            raise ValueError(f'Parameter "model" should be one of {allowed_models}')
        return v

    @model_validator(mode='after')
    def validate_model_params(self):  
        assert self.model in self.model_params, f"{self.model=}, {self.model_params=}"
        params = self.model_params[self.model]
        if self.model == 'rnn':
            params = RNNParams(**params)
        else:
            assert self.model == 'transformer'
            params = TransformerParams(**params)
        self.model_params = {self.model: params}
        return self
