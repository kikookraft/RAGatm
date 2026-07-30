import uuid
import os
from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')


class MinimalSource(StrictBaseModel):
    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError('file_path cannot be empty')
        try:
            if not os.path.exists(cleaned_value):
                raise ValueError(f'file_path does not exist: {cleaned_value}')
            with open(cleaned_value, 'r'):
                pass
        except FileNotFoundError:
            raise ValueError(f'file_path does not exist: {cleaned_value}')
        except Exception as e:
            raise ValueError(
                f'Error accessing file_path: {cleaned_value}. Error: {e}')
        return cleaned_value

    @model_validator(mode='after')
    def check_indices(self):
        if self.first_character_index >= self.last_character_index:
            raise ValueError(
                'last_character_index must be greater than '
                'first_character_index'
            )
        if self.last_character_index - self.first_character_index > 2000:
            raise ValueError('source span cannot exceed 2000 characters')
        return self


class UnansweredQuestion(StrictBaseModel):
    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str

    @field_validator('question')
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("Question cannot be empty")
        return cleaned_value


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("Answer cannot be empty")
        return cleaned_value


class RagDataset(StrictBaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(StrictBaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(StrictBaseModel):
    search_results: List[MinimalSearchResults]
    k: int = Field(ge=0)


class StudentSearchResultsAndAnswer(StrictBaseModel):
    search_results: List[MinimalAnswer]
    k: int = Field(ge=0)
