from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class PriorityTier(str, Enum):
    HIGH = "High Priority"
    MEDIUM = "Medium Priority"
    LOW = "Low / Unique"


class ExtractedQuestion(BaseModel):
    id: str
    paper_id: str
    paper_name: str
    question_number: str
    unit_number: Optional[str] = None
    marks: Optional[int] = None
    exact_text: str


class PaperInfo(BaseModel):
    paper_id: str
    filename: str
    paper_name: str
    exam_type: str = "Endsem"
    page_count: int
    used_ocr: bool = False
    ocr_warning: Optional[str] = None
    extracted_count: int = 0


class ExtractPaperResponse(BaseModel):
    paper: PaperInfo
    questions: List[ExtractedQuestion]
    warnings: List[str] = Field(default_factory=list)


class QuestionCluster(BaseModel):
    cluster_id: str
    topic_title: str
    unit_label: Optional[str] = None
    priority: PriorityTier
    unique_paper_count: int
    total_papers: int
    total_occurrences: int
    avg_similarity: float
    questions: List[ExtractedQuestion]


class AnalysisResponse(BaseModel):
    analysis_id: str
    subject_name: str
    similarity_threshold: float
    grouping_mode: str = "AI Concept Grouping"
    total_papers: int
    total_questions: int
    papers: List[PaperInfo]
    clusters: List[QuestionCluster]
    warnings: List[str] = Field(default_factory=list)


class GroupRequest(BaseModel):
    subject_name: str = "SPPU Subject Revision Notes"
    similarity_threshold: float = Field(default=0.56, ge=0.45, le=0.95)
    papers: List[PaperInfo]
    questions: List[ExtractedQuestion]
    warnings: List[str] = Field(default_factory=list)


class ReclusterStatelessRequest(BaseModel):
    analysis: AnalysisResponse
    similarity_threshold: float = Field(ge=0.45, le=0.95)


class MergeStatelessRequest(BaseModel):
    analysis: AnalysisResponse
    cluster_ids: List[str] = Field(min_length=2)
    custom_title: Optional[str] = None


class SplitStatelessRequest(BaseModel):
    analysis: AnalysisResponse
    cluster_id: str
    question_id: str
