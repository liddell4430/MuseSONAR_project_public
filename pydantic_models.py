# pydantic_models.py

"""
MuseSonar 분석 결과 데이터 구조를 정의하는 Pydantic 모델입니다.
"""

from pydantic import BaseModel, Field, NonNegativeInt, NonNegativeFloat
from typing import List, Optional

# LLM 검증 결과를 위한 모델
class LlmVerificationModel(BaseModel):
    """LLM 검증 결과 상세 모델"""
    status: Optional[str] = None  # 예: "Yes", "No", "Error", "Skipped", "Unclear"
    reason: Optional[str] = None  # LLM이 제공한 이유 또는 오류 메시지

# 상위 유사 결과를 위한 모델
class SimilarResultModel(BaseModel):
    """개별 상위 유사 결과 항목 모델"""
    rank: NonNegativeInt                                    # 순위 (0 이상 정수)
    similarity_percentage: float = Field(ge=0.0, le=100.0)  # 유사도 점수 (0 ~ 100 사이 실수)
    content_preview: str                                    # 내용 미리보기 (문자열)
    link: Optional[str] = None                              # 결과 링크 (Optional 문자열)
    source: str                                             # 출처 (예: "Google Search", "KIPRIS Patent")
    llm_verification: Optional[LlmVerificationModel] = None # LLM 검증 결과 (Optional 모델)

# 평가 지표(메트릭)를 위한 모델
class MetricModel(BaseModel):
    """분석 결과의 상세 평가 지표 모델"""
    concept_density_percentage: Optional[float] = Field(None, ge=0.0, le=100.0) # 웹/특허 개념 밀도 (평균 유사도, 0~100)
    evidence_discovery_rate_percentage: Optional[float] = Field(None, ge=0.0, le=100.0) # 구체적 구현 증거 발견율 (0~100)
    evidence_count: NonNegativeInt                         # 구체적 구현 증거 개수 (0 이상 정수)
    verification_attempts: NonNegativeInt                  # LLM 검증 시도 횟수 (0 이상 정수)
    relevant_search_results_count: NonNegativeInt          # 관련성 높은 검색 결과 개수 (0 이상 정수)
    combined_results_found: bool                           # 웹/특허 검색 결과 존재 여부 (True/False)
    verification_threshold_percentage: float = Field(ge=0.0, le=100.0) # LLM 검증 대상 선정 기준 유사도 (0~100)

class AnalysisResultModel(BaseModel):
    """MuseSonar 최종 분석 결과 모델"""
    # --- 기존 필드 ---
    rating: Optional[str] = None # 오류 시 값이 없을 수 있으므로 Optional로 변경
    score: Optional[int] = Field(None, ge=0, le=100)
    interpretation: Optional[str] = None # 오류 시 값이 없을 수 있으므로 Optional로 변경
    warning: Optional[str] = None
    metrics: Optional[MetricModel] = None # 오류 시 값이 없을 수 있으므로 Optional로 변경
    top_similar_results: List[SimilarResultModel] = []
    # --- 오류 처리용 필드 추가 ---
    error: Optional[str] = None # 오류 발생 시 메시지 저장

    # 모델 유효성 검사 예시 (선택 사항): 점수가 있는데 0~100 범위를 벗어나면 오류 발생시킴
    # from pydantic import validator
    # @validator('score')
    # def score_must_be_in_range(cls, v):
    #     if v is not None and not (0 <= v <= 100):
    #         raise ValueError('score must be between 0 and 100')
    #     return v
