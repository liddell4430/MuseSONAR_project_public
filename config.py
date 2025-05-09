# config.py

"""
MuseSonar 프로젝트의 설정을 관리하는 파일로 public버전은 대부분 비공개 되어있습니다.
임계값, API 관련 상수, LLM 프롬프트 등을 정의.
""" 

from typing import Tuple # Tuple 타입 힌트를 위해 추가

# --- 유사도 및 평가 관련 임계값 ---
RELEVANCE_THRESHOLD: float =
HIGH_SIMILARITY_THRESHOLD: float =
JHGAN_THRESHOLD_IGNORE_LLM: float =
LLM_YES_THRESHOLD_LOW: float =

# --- API 요청 관련 설정 ---
KIPRIS_ADVANCED_SEARCH_ROWS: int = 
KIPRIS_WORD_SEARCH_ROWS: int =
MAX_KIPRIS_KEYWORDS: int =

# --- LLM 검증 관련 설정 ---
MAX_LLM_VERIFICATION_TARGETS: int =

# 프롬프트 버전 
PROMPT_VERSION: str =

# LLM 입력 스니펫 길이 관련 설정
MAX_EXCERPT: int = 
# 구체성 힌트 키워드 (튜플) - build_excerpt 함수에서 사용
KW_HINTS: Tuple[str, ...] =

# LLM 검증용 프롬프트 템플릿 
LLM_VERIFICATION_PROMPT_TEMPLATE: str = """
[Context]
You are an AI assistant judging whether a given search result ('Hit-excerpt') provides concrete evidence that a user's idea ('User-idea') has already been implemented or exists in a tangible form. Focus *primarily*... 
""" # 일부 공개