from pydantic import BaseModel, Field

class ProcessingVideoConfig(BaseModel):
    confidence: float = Field(0.3, gt=0.0, le=1.0)
    skip_frames: int = Field(5, ge=0)
    sharp_threshold: int = Field(20, ge=0, le=100)

class TrackerConfig(BaseModel):
    max_age: int = Field(30, gt=0.0)
    n_init: int = Field(3, ge=0)
    nn_budget: int = Field(100, ge=0)

class SimilarityConfig(BaseModel):
    alpha: float = Field(0.7, gt=0.0, le=1.0)

class CompareThreshold(BaseModel):
    threshold: float = Field(0.7, gt=0.0, le=1.0)


# Создаем основную схему конфига
class AppConfig(BaseModel):
    processing: ProcessingVideoConfig
    tracker: TrackerConfig
    similarity: SimilarityConfig
    compare: CompareThreshold
