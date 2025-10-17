from pydantic import BaseModel, Field

class TrainParams(BaseModel):
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=64, ge=1, le=1024)
    val_batch_size: int = Field(default=128, ge=1, le=1024)
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")
    
