# MuseSONAR_public.py

import config
from pydantic_models import AnalysisResultModel, MetricModel, SimilarResultModel, LlmVerificationModel
import requests
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import google.generativeai as genai
import time
from dotenv import load_dotenv
import xml.etree.ElementTree as ET 
import re
import diskcache
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import traceback
# --- 타입 힌트용 임포트 ---
from typing import List, Dict, Tuple, Optional, Any, Literal, Union # 필요한 타입 임포트
from google.generativeai.generative_models import GenerativeModel # Gemini 모델 타입
from konlpy.tag import Okt as OktType # KoNLPy Okt 타입 (Optional 처리 위해 별도 임포트)

# --- 로깅 설정 ---
log_format = '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    handlers=[
                        logging.FileHandler("musessonar_analysis.log", encoding='utf-8'),
                        logging.StreamHandler()
                    ])
logging.info("==================== MuseSonar 스크립트 시작 ====================")
logging.info("로깅 설정 완료.")


# --- 캐시 설정 ---
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_dir")
cache: Optional[diskcache.Cache] = None
try:
    cache = diskcache.Cache(CACHE_DIR)
    logging.info(f"DiskCache 초기화 완료. 캐시 디렉토리: {CACHE_DIR}")

    # --- 임시 코드(디버깅용): 캐시 클리어 ---
    # if cache:
        # logging.warning("!!! 개발용 임시 코드: DiskCache 전체 초기화 수행 !!!")
        # cache.clear()
        # logging.warning("!!! 캐시 초기화 완료 !!!")
    
except Exception as e:
    logging.critical(f"DiskCache 초기화 실패! 캐시 기능 비활성화. 오류: {e}")
    cache = None

# --- KoNLPy 임포트 ---
okt: Optional[OktType] = None # Okt 객체 타입 힌트 (Optional)
try:
    from konlpy.tag import Okt
    okt = Okt()
    logging.info("KoNLPy Okt 형태소 분석기 로딩 완료.")
except ImportError:
    logging.warning("konlpy 라이브러리를 찾을 수 없습니다. 'pip install konlpy JPype1'으로 설치해주세요. 키워드 추출 기능이 비활성화됩니다.")
    okt = None
except Exception as e:
    logging.warning(f"KoNLPy Okt 로딩 중 오류 발생: {e}. 키워드 추출 기능이 비활성화됩니다.", exc_info=True)
    okt = None


# --- 환경 변수 로드 ---
load_dotenv()

# --- 필수 환경 변수 중앙 검증 ---
logging.info("필수 환경 변수 검증 시작...")
required_env_vars: Dict[str, str] = { # 딕셔너리 타입 힌트
    'GOOGLE_SEARCH_API_KEY': 'Google Custom Search API 키',
    'SEARCH_ENGINE_ID': 'Google Custom Search Engine ID',
    'GOOGLE_API_KEY_GEMINI': 'Google Gemini API 키'
}
missing_vars: List[str] = [] # 리스트 타입 힌트
env_vars: Dict[str, str] = {} # 딕셔너리 타입 힌트

for var_name, description in required_env_vars.items():
    value: Optional[str] = os.getenv(var_name) # Optional[str] 타입 힌트
    if not value:
        missing_vars.append(f"'{var_name}' ({description})")
    else:
        env_vars[var_name] = value

if missing_vars:
    error_message: str = "치명적 오류: 필수 환경 변수가 설정되지 않았습니다. 다음 변수를 .env 파일에 설정해주세요:\n - " + "\n - ".join(missing_vars)
    logging.critical(error_message)
    sys.exit(1)
else:
    GOOGLE_SEARCH_API_KEY: str = env_vars['GOOGLE_SEARCH_API_KEY'] # str 타입 힌트
    SEARCH_ENGINE_ID: str = env_vars['SEARCH_ENGINE_ID'] # str 타입 힌트
    GOOGLE_API_KEY_GEMINI: str = env_vars['GOOGLE_API_KEY_GEMINI'] # str 타입 힌트
    logging.info("모든 필수 환경 변수 검증 통과.")

KIPRIS_API_KEY: Optional[str] = os.getenv('KIPRIS_API_KEY') # Optional[str] 타입 힌트
if not KIPRIS_API_KEY:
    logging.warning("KIPRIS API 키(KIPRIS_API_KEY)가 환경 변수에 설정되지 않았습니다. 특허 검색 기능이 비활성화됩니다.")
else:
    logging.info("KIPRIS API 키 확인 완료.")


# 검색 결과 및 딕셔너리 타입 정의 (딕셔너리 리스트)
GoogleResultType = List[Dict[str, str]]
KiprisResultType = List[Dict[str, str]]
LlmVerificationResultType = Tuple[str, str] # LLM 검증 결과 타입 (상태 문자열, 이유 문자열)
ResultDictType = Dict[str, Any] # 필요시 더 상세하게 정의 가능
SortedResultItemType = Dict[str, Any] # 'text', 'score', 'link', 'source' 등 포함
LlmVerificationResultsMapType = Dict[str, LlmVerificationResultType] # text -> (status, reason)
llm_verification_results: LlmVerificationResultsMapType = {}


# --- 함수 정의 ---

# 키워드 추출 함수
def extract_keywords(text: str, max_kw: int = config.MAX_KIPRIS_KEYWORDS) -> str:
    """KoNLPy Okt와 정규식을 사용하여 입력 텍스트에서 명사 및 영문/숫자 키워드를 추출합니다."""
    logging.info(f"키워드 추출 시작 (max_kw={max_kw}): '{text[:50]}...'")

    if okt is None:
        logging.warning("KoNLPy Okt 객체가 없어 키워드 추출 불가. 원문 텍스트를 사용합니다.")
        return text.strip()

    keywords: List[str] = []
    try:
        pos_tagged: List[Tuple[str, str]] = okt.pos(text, norm=True, stem=True)
        nouns: List[str] = [n for n, t in pos_tagged if t == 'Noun']
        logging.debug(f"형태소 분석 결과 (명사): {nouns}")

        alphas_digits: List[str] = re.findall(r'\b[A-Za-z]{1,5}\d*\b|\b\d+[A-Za-z]+\b', text)
        logging.debug(f"정규식 추출 결과 (영문/숫자): {alphas_digits}")

        filtered_keywords: List[str] = [w for w in nouns if len(w) > 1] + alphas_digits
        logging.debug(f"필터링된 키워드: {filtered_keywords}")

        seen: set[str] = set()
        unique_keywords: List[str] = []
        for k in filtered_keywords:
            if k not in seen:
                unique_keywords.append(k)
                seen.add(k)
        logging.debug(f"중복 제거된 키워드: {unique_keywords}")

        final_keywords: str = ' '.join(unique_keywords[:max_kw]) if unique_keywords else text.strip()

        logging.info(f"추출된 최종 KIPRIS 키워드: '{final_keywords}'")
        return final_keywords

    except Exception as e:
        logging.error(f"키워드 추출 중 오류 발생. 원문 텍스트를 사용합니다.", exc_info=True)
        return text.strip()


# =============== KIPRIS 특허 검색 함수  ===============
# requests + tenacity + fallback + XML 파싱 조합 사용

# KIPRIS 특허 검색 함수
@cache.memoize(expire=864000) if cache else lambda f: f # 캐시 비활성화 시 데코레이터 적용 안 함
def search_kipris_patents(query: str) -> KiprisResultType:
    """
    KIPRIS 특허 검색 결과를 가져옵니다 (캐싱 적용).
    API 키는 내부적으로 환경 변수에서 읽어옵니다.
    """
    logging.info(f"KIPRIS 검색 시도 (캐시 확인): query='{query}'")

    api_key: Optional[str] = os.getenv('KIPRIS_API_KEY')
    if not api_key:
        logging.warning("KIPRIS API 키가 없어 특허 검색을 건너뜁니다 (캐시 래퍼 함수).")
        return []

    try:
        logging.debug(f"캐시 미스 또는 만료. KIPRIS 내부 검색 함수 호출: query='{query}'")
        results: KiprisResultType = _search_kipris_patents_internal(query, api_key)
        logging.info(f"KIPRIS 검색 완료 (캐시 저장됨): query='{query}', 결과 {len(results)}개")
    except Exception as e:
        logging.error(f"KIPRIS 검색 중 예외 발생 (캐시 래퍼): query='{query}'", exc_info=True)
        results = []

    return results

# KIPRIS API 응답 타입 정의 (XML 요소 리스트, None, 또는 False)
KiprisApiResponseType = List[ET.Element] | None | Literal[False] # Python 3.10+ Union Syntax

# 재시도 전 로그를 남기기 위한 설정 (logging 모듈과 연동)
# tenacity가 재시도하기 전에 INFO 레벨로 로그를 남김
tenacity_logger = logging.getLogger(__name__) # 현재 모듈의 로거 사용
before_sleep = before_sleep_log(tenacity_logger, logging.INFO)

# 재시도 데코레이터 설정
@retry(
    stop=stop_after_attempt(3),  # 최대 3번 시도 (최초 1번 + 재시도 2번)
    wait=wait_exponential(multiplier=1, min=2, max=10), # 2초, 4초 간격으로 재시도 (최대 10초)
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)), # Timeout 또는 ConnectionError 발생 시 재시도
    before_sleep=before_sleep # 재시도 전에 위에서 설정한 로그 남기기
)

# KIPRIS 특허 검색 요청 함수(재시도 로직 추가됨)
def request_kipris(url: str, params: Dict[str, Any], search_type: str) -> KiprisApiResponseType:
    """KIPRIS API 요청 및 기본 처리 (tenacity 재시도 적용)"""
    logging.debug(f"KIPRIS API 요청 시도: Type='{search_type}', URL='{url}'")
    try:
        logging.debug(f"  요청 Params (일부): word='{params.get('word', '')}', rows='{params.get('numOfRows')}', query(title/astrt)='{params.get('inventionTitle', 'N/A')}'")
        response = requests.get(url, params=params, headers={'User-Agent': 'MuseSonar-prototype/1.0'}, timeout=30) # timeout은 그대로 유지
        response.raise_for_status() # HTTP 5xx 같은 오류 시 여기서 예외 발생 (기본적으로 tenacity 재시도 안 함)

        # --- 성공적인 응답 수신 후 처리 ---
        logging.debug(f"KIPRIS 응답 수신 완료 (Status: {response.status_code}). XML 파싱 시작...")
        xml_root: ET.Element = ET.fromstring(response.content)
        result_code: Optional[str] = xml_root.findtext('.//resultCode')
        result_msg: Optional[str] = xml_root.findtext('.//resultMsg')

        if result_code == '00':
            items: List[ET.Element] = xml_root.findall('.//item')
            if items:
                logging.info(f"KIPRIS {search_type} 검색 성공: {len(items)}개 결과 확보.")
                return items # 성공 시 결과 반환
            else:
                logging.warning(f"KIPRIS {search_type} 검색 성공했으나 결과 0건.")
                return None # 성공했으나 결과 없음
        else:
            # KIPRIS API 자체 오류 (e.g., 잘못된 요청, 키 오류 등)는 재시도 대상 아님
            logging.error(f"KIPRIS {search_type} API 자체 오류 (재시도 대상 아님): {result_msg} (코드: {result_code})")
            return False # API 오류 시 False 반환

    # --- 예외 처리: tenacity가 재시도할 예외 ---
    except requests.exceptions.Timeout as e:
        logging.warning(f"KIPRIS {search_type} API 요청 시간 초과 (timeout=30s). Tenacity 재시도 예정...")
        raise e # 예외를 다시 발생시켜 tenacity가 잡고 재시도하도록 함
    except requests.exceptions.ConnectionError as e:
        logging.warning(f"KIPRIS {search_type} API 연결 오류 발생. Tenacity 재시도 예정...")
        raise e # 예외를 다시 발생시켜 tenacity가 잡고 재시도하도록 함

    # --- 예외 처리: tenacity가 재시도하지 않을 예외 ---
    except requests.exceptions.RequestException as e: # Timeout, ConnectionError 외의 requests 예외
        logging.error(f"KIPRIS {search_type} API 요청 중 기타 오류 발생 (재시도 대상 아님): {e}", exc_info=True)
        return False # 재시도 안 할 오류 시 False 반환
    except ET.ParseError as e: # XML 파싱 오류
        logging.error(f"KIPRIS {search_type} API 응답 XML 파싱 오류 (재시도 대상 아님): {e}", exc_info=True)
        try:
            # 파싱 오류 시 응답 내용 일부 로깅 (디버깅 도움)
            logging.debug(f"XML 파싱 오류 시 응답 내용 (일부): {response.text[:500]}...")
        except NameError: pass # response 객체가 없을 수도 있음
        return False # 재시도 안 할 오류 시 False 반환
    except Exception as e: # 그 외 모든 예상치 못한 예외
        logging.error(f"KIPRIS {search_type} 검색 중 알 수 없는 오류 발생 (재시도 대상 아님)", exc_info=True)
        return False # 재시도 안 할 오류 시 False 반환

# KIPRIS 특허 검색 내부 함수 
def _search_kipris_patents_internal(query: str, api_key: str) -> KiprisResultType:
    """
    KIPRIS Open API를 호출하여 특허 검색 결과를 반환합니다. (내부 함수)
    3단계 Fallback 적용.
    """
    logging.info(f"KIPRIS 내부 검색 시작: query='{query}'")
    if not api_key:
        logging.error("KIPRIS API 키가 없음 (내부 함수). 검색 불가.")
        return []

    patent_data: KiprisResultType = []
    search_keywords: str = extract_keywords(query)

    # --- 1단계: 키워드 기반 Advanced Search ---
    logging.info(f"KIPRIS 1단계 시도: Advanced Search (키워드='{search_keywords}', max_rows={config.KIPRIS_ADVANCED_SEARCH_ROWS})")
    url_advanced: str = "https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"
    params_advanced: Dict[str, Any] = {
        'word': '', 'inventionTitle': search_keywords, 'astrtCont': search_keywords,
        'patent': 'true', 'utility': 'true', 'numOfRows': config.KIPRIS_ADVANCED_SEARCH_ROWS,
        'pageNo': 1, 'sortSpec': 'OPD', 'descSort': 'true', 'ServiceKey': api_key
    }
    items: KiprisApiResponseType = request_kipris(url_advanced, params_advanced, "Advanced(Keyword)")

    # --- 2단계: 원문 기반 Word Search ---
    fallback_reason: str = ""
    if items is None or items is False:
        fallback_reason = "API 오류" if items is False else "결과 없음"
        logging.info(f"KIPRIS 1단계 결과({fallback_reason}). 2단계 시도: Word Search (원문='{query[:50]}...', max_rows={config.KIPRIS_WORD_SEARCH_ROWS})")
        url_word_orig: str = "https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getWordSearch"
        params_word_orig: Dict[str, Any] = { 'word': query, 'year': 0, 'patent': 'true', 'utility': 'true', 'numOfRows': config.KIPRIS_WORD_SEARCH_ROWS, 'pageNo': 1, 'ServiceKey': api_key }
        items = request_kipris(url_word_orig, params_word_orig, "Word(Original)")

    # --- 3단계: 키워드 기반 Word Search ---
    if items is None or items is False:
        fallback_reason = "API 오류" if items is False else "결과 없음"
        if search_keywords != query.strip():
            logging.info(f"KIPRIS 2단계 결과({fallback_reason}). 3단계 시도: Word Search (키워드='{search_keywords}', max_rows={config.KIPRIS_WORD_SEARCH_ROWS})")
            url_word_kw: str = "https://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getWordSearch"
            params_word_kw: Dict[str, Any] = { 'word': search_keywords, 'year': 0, 'patent': 'true', 'utility': 'true', 'numOfRows': config.KIPRIS_WORD_SEARCH_ROWS, 'pageNo': 1, 'ServiceKey': api_key }
            items = request_kipris(url_word_kw, params_word_kw, "Word(Keyword)")
        else:
            logging.info("KIPRIS 3단계 검색 건너뜀 (추출된 키워드가 원문과 동일).")

    # --- 최종 결과 처리 ---
    if isinstance(items, list) and items: # items가 list이고 비어있지 않은 경우
        logging.info(f"KIPRIS 최종 검색 성공. 파싱 시작 (items: {len(items)}개)")
        patent_data = parse_kipris_items(items)
    else:
        logging.warning(f"KIPRIS 최종 특허 검색 결과 없음: query='{query}'")
        patent_data = []

    logging.info(f"KIPRIS 내부 검색 완료: 최종 결과 {len(patent_data)}개")
    return patent_data

# KIPRIS 특허 XML 파싱
def parse_kipris_items(items: List[ET.Element]) -> KiprisResultType:
    """KIPRIS API 응답의 <item> 리스트를 파싱하여 딕셔너리 리스트로 반환합니다."""
    logging.debug(f"KIPRIS XML 아이템 파싱 시작 (입력 item 수: {len(items)})")
    parsed_data: KiprisResultType = []
    skipped_count: int = 0
    for i, item in enumerate(items):
        try:
            title: str = item.findtext('inventionTitle', '').strip()
            abstract: str = item.findtext('astrtCont', '').strip()
            app_num: str = item.findtext('applicationNumber', '').strip()
            link: str = f"https://kpat.kipris.or.kr/kpat/searchLogina.do?next=MainSearch&target=pat_reg&Method=biblioTM&INPUT_TYPE=applno&query={app_num}" if app_num else ""
            clean_abstract: str = abstract if abstract and abstract != "내용 없음." else ""

            if title or clean_abstract:
                content: str = f"{title}: {clean_abstract}" if title and clean_abstract else title
                parsed_data.append({
                        'text': content, 'title': title, 'abstract': clean_abstract,
                        'link': link, 'source': 'KIPRIS Patent'
                    })
            else:
                skipped_count += 1
                logging.debug(f"Item {i+1} 건너뜀: 제목과 유효한 초록 모두 없음 (app_num: {app_num})")

        except Exception as e:
            logging.error(f"KIPRIS 아이템 파싱 중 오류 발생 (Item index: {i}): {e}", exc_info=True)
            skipped_count += 1

    logging.debug(f"KIPRIS XML 아이템 파싱 완료 (결과: {len(parsed_data)}개, 건너뜀: {skipped_count}개)")
    return parsed_data

# ============= KIPRIS 특허 검색함수 종료 ============= 


# 구글 검색함수
@cache.memoize(expire=864000) if cache else lambda f: f
def google_search(query: str, num_results: int = 10) -> GoogleResultType:
    """
    Google 검색 결과를 가져옵니다 (캐싱 적용).
    API 키와 CX ID는 내부적으로 환경 변수에서 읽어옵니다.
    """
    logging.info(f"Google 검색 시도 (캐시 확인): query='{query}', num={num_results}")

    api_key: Optional[str] = os.getenv('GOOGLE_SEARCH_API_KEY')
    cx: Optional[str] = os.getenv('SEARCH_ENGINE_ID')
    if not api_key or not cx:
        logging.error("Google Search API 키/ID 환경 변수 누락 (캐시 래퍼). 검색 불가.")
        return []

    try:
        logging.debug(f"캐시 미스 또는 만료. Google 내부 검색 함수 호출: query='{query}', num={num_results}")
        results: GoogleResultType = _google_search_internal(query, api_key, cx, num_results)
        logging.info(f"Google 검색 완료 (캐시 저장됨): query='{query}', 결과 {len(results)}개")
    except Exception as e:
        logging.error(f"Google 검색 중 예외 발생 (캐시 래퍼): query='{query}'", exc_info=True)
        results = []

    return results

# 구글 검색 내부 함수
def _google_search_internal(query: str, api_key: str, cx: str, num_results: int = 10) -> GoogleResultType:
    """Google Custom Search API를 호출하여 검색 결과를 반환"""
    logging.info(f"Google 내부 검색 시작: query='{query}', num={num_results}")
    search_url: str = "https://www.googleapis.com/customsearch/v1"
    headers: Dict[str, str] = {'User-Agent': 'MuseSONAR - prototype/1.0'}
    params: Dict[str, Any] = {'key': api_key, 'cx': cx, 'q': query, 'num': num_results}

    try:
        logging.debug(f"Google API 요청 시작: URL='{search_url}'")
        logging.debug(f"  요청 Params (일부): q='{query}', num='{num_results}'")

        response = requests.get(search_url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        logging.debug(f"Google API 응답 수신 완료 (Status: {response.status_code}). JSON 파싱 시작...")
        search_results_json: Dict[str, Any] = response.json()
        result_data: GoogleResultType = []

        if 'items' in search_results_json:
            logging.debug(f"Google API 응답에서 '{len(search_results_json['items'])}'개의 아이템 발견.")
            for item in search_results_json['items']:
                title: str = item.get('title', '')
                snippet: str = item.get('snippet', '')
                link: str = item.get('link', '')
                if title or snippet:
                    content: str = f"{title}: {snippet}" if title and snippet else title if title else snippet
                    result_data.append({'text': content, 'link': link, 'source': 'Google Search'})
            logging.info(f"Google 검색 파싱 완료: {len(result_data)}개의 유효 결과 확보.")
            return result_data
        else:
            logging.warning("Google 검색 결과가 없습니다 ('items' 키 없음).")
            return []
    except requests.exceptions.Timeout:
        logging.error("Google Search API 요청 시간 초과 (timeout=20s).")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Google Search API 요청 중 오류 발생: {e}", exc_info=True)
        return []
    except Exception as e:
        logging.error(f"Google 검색 처리 중 알 수 없는 오류 발생", exc_info=True)
        return []

# 구글 검색 발췌 함수
def build_excerpt(full_text: str) -> str:
    """입력 텍스트에서 LLM 검증에 사용할 스니펫을 생성합니다. 특정 키워드 포함 시 길이를 늘립니다."""
    # 키워드가 텍스트 앞부분(예: 1200자)에 있는지 확인
    # full_text가 1200자보다 짧을 수 있으므로 슬라이싱 주의
    check_range = min(len(full_text), 1200)
    if any(k in full_text[:check_range] for k in config.KW_HINTS):
        # 힌트 키워드가 있으면 더 길게 자름 (예: 1000자)
        # full_text가 1000자보다 짧을 수 있으므로 슬라이싱 주의
        target_length = min(len(full_text), 1000)
        logging.debug(f"힌트 키워드 발견! 스니펫 길이를 {target_length}자로 확장.")
        return full_text[:target_length]
    else:
        # 힌트 키워드 없으면 기본 길이로 자름
        target_length = min(len(full_text), config.MAX_EXCERPT)
        # logging.debug(f"힌트 키워드 없음. 스니펫 기본 길이 {target_length}자 적용.") # 로그가 너무 많을 수 있어 주석 처리
        return full_text[:target_length]


# LLM 2차 검증 함수
@cache.memoize(expire=86400) if cache else lambda f: f # 1일 캐싱
def verify_similarity_with_llm_cached(user_idea: str,
                                    search_text_excerpt: str,
                                    source_type_mapped: str,
                                    model_llm: Optional[GenerativeModel],
                                    prompt_ver: str = config.PROMPT_VERSION) -> LlmVerificationResultType:
    """[캐시 래퍼] LLM 검증 결과를 캐시에서 찾거나 내부 함수를 호출합니다. (프롬프트 버전 캐시 키 포함)"""
    # 해당llm 2차검증 로직은 비공개 처리 영역입니다

# LLM 2차 검증 내부 함수
def _verify_similarity_with_llm_internal(user_idea: str, search_text_excerpt: str, source_type_mapped: str, model_llm: GenerativeModel) -> LlmVerificationResultType:
    """[내부 함수] LLM API를 호출하여 유사성을 검증합니다."""
    logging.debug(f"LLM 내부 검증 시작: source='{source_type_mapped}'")
    # model_llm은 호출 전에 None 체크됨 (여기서는 GenerativeModel 타입으로 가정)

    # 해당llm 2차검증 로직은 비공개 처리 영역입니다

# LLM 2차 검증 내부 함수
def verify_similarity_with_llm(user_idea: str, hit: Dict[str, Any], model_llm: Optional[GenerativeModel]) -> LlmVerificationResultType:
    """
    LLM 검증을 수행합니다 (캐싱 적용).
    입력 데이터를 준비하고 캐시된 함수를 호출합니다.
    """
    # 해당llm 2차검증 로직은 비공개 처리 영역입니다

# 점수 계산 함수
def calculate_MuseSONAR_score(final_rating: str, jhgan_avg_score: float, llm_yes_ratio: float, num_to_verify: int) -> int:
    """최종 등급과 세부 지표를 바탕으로 고유성 점수(0-100) 계산"""
    logging.debug(f"MuseSonar 점수 계산 시작: rating='{final_rating}', avg_score={jhgan_avg_score:.2f}, yes_ratio={llm_yes_ratio:.2f}, num_verify={num_to_verify}")
    base_score: int = 0
    adjustment: int = 0 # 반올림 후 정수가 되므로 int

    # 해당 점수 계산 로직은 비공개 처리 영역입니다.

# 등급 결정 함수
def determine_originality(average_score: float, verified_similar_count: int, llm_yes_ratio: float, has_search_results_flag: bool, max_similarity_score: float) -> Tuple[str, str, str]:
    """
    평균 유사도, LLM 검증 결과, 최고 유사도 등을 종합하여
    세분화된 고유성 등급 및 해석 메시지를 결정합니다.
    """
    logging.debug(f"고유성 등급 결정 시작: avg_score={average_score:.2f}, yes_cnt={verified_similar_count}, yes_ratio={llm_yes_ratio:.2f}, has_results={has_search_results_flag}, max_sim={max_similarity_score:.2f}")

    final_rating: str = "평가 불가"
    interpretation_message: str = "분석 결과를 바탕으로 고유성 등급을 평가하기 어렵습니다."
    conditional_warning: str = ""

    # --- 해당 등급 결정 로직은 비공개 처리 영역입니다 ---
    


# 최종 평가 함수
def display_results_dashboard(results: AnalysisResultModel) -> None: # 인자 타입을 AnalysisResultModel로 변경
    """구조화된 AnalysisResultModel 객체를 대시보드 형식으로 출력"""
    logging.info("\n" + "="*30 + " 아이디어 고유성 최종 평가 " + "="*30)

    # 객체 속성으로 접근 (e.g., results.get('rating') -> results.rating)
    rating: Optional[str] = results.rating # Optional일 수 있음
    score: Optional[int] = results.score
    interpretation: Optional[str] = results.interpretation # Optional일 수 있음

    logging.info(f"\n📊 아이디어 고유성 평가: {rating if rating else 'N/A'}") 
    if score is not None:
        logging.info(f"⭐ 고유성 점수: {score} / 100")
    else:
        logging.info(f"⭐ 고유성 점수: 계산 불가 또는 정보 부족")
    if rating: 
      logging.info("-" * (len(f"📊 아이디어 고유성 평가: {rating}") + 2))
    if interpretation: # interpretation이 있을 때만 출력
        for line in interpretation.splitlines():
            logging.info(line)

    warning: Optional[str] = results.warning
    if warning:
        for line in warning.splitlines():
            logging.warning(line)

    logging.info("\n🔍 평가 근거 상세:")
    metrics: Optional[MetricModel] = results.metrics # Optional일 수 있음
    if not metrics or not metrics.combined_results_found: # metrics 객체 존재 및 플래그 확인
        logging.info("  - 웹/특허 검색 결과: 아이디어 관련 정보를 찾을 수 없거나 분석 불가.")
    else:
        density: Optional[float] = metrics.concept_density_percentage
        relevant_count: int = metrics.relevant_search_results_count
        if density is not None:
            logging.info(f"  - 웹/특허 개념 밀도 (평균 유사도): {density:.2f}% (관련성 높은 {relevant_count}개 결과 기준)")
        else:
            logging.info("  - 웹/특허 개념 밀도: 관련성 높은 정보 없음")

        evidence_rate: Optional[float] = metrics.evidence_discovery_rate_percentage
        attempts: int = metrics.verification_attempts
        evidence_count: int = metrics.evidence_count
        verify_threshold: float = metrics.verification_threshold_percentage
        if attempts > 0 and evidence_rate is not None: # evidence_rate None 체크 추가
            logging.info(f"  - 구체적 구현 증거 (발견율): {evidence_rate:.1f}% ({evidence_count}개 / {attempts}개 검증)")
        elif relevant_count > 0:
            logging.info(f"  - 구체적 구현 증거: 검증 대상 결과 없음 (유사도 < {verify_threshold:.0f}%)")
        else:
            logging.info(f"  - 구체적 구현 증거: 해당 사항 없음")

    logging.info("\n" + "="*20 + " 참고: 가장 유사한 웹/특허 정보 (상위 5개) " + "="*20)
    top_results: List[SimilarResultModel] = results.top_similar_results # 리스트 직접 접근
    if top_results:
        for item in top_results: # item은 이제 SimilarResultModel 객체
            rank: int = item.rank
            sim_pct: float = item.similarity_percentage
            source: str = item.source
            content: str = item.content_preview
            logging.info(f"\n#{rank}. 유사도: {sim_pct:.2f}% (출처: {source})")
            logging.info(f"   - 내용: {content}")
            # link: Optional[str] = item.link # 필요시 사용
            # if link: logging.info(f"   - 링크: {link}")

            llm_verification: Optional[LlmVerificationModel] = item.llm_verification
            if llm_verification: # LlmVerificationModel 객체 존재 확인
                status: Optional[str] = llm_verification.status
                reason: Optional[str] = llm_verification.reason
                status_icon: str = {"Yes": "✅", "No": "❌", "Unclear": "❓", "Error": "⚠️", "Skipped": "⏭️"}.get(status, "") if status else "" # status None 체크
                logging.info(f"   - LLM 검증 (구체적 구현 증거): {status_icon} {status if status else 'Unknown'}")
                if reason and status not in ["Skipped"]:
                    for line in reason.splitlines():
                        logging.info(f"     ㄴ 이유/오류: {line}")
            else:
                logging.info(f"   - LLM 검증: 수행되지 않음")
    else:
        logging.info("표시할 유사 결과가 없습니다.")

logging.debug(f"create_results_dictionary 호출됨. llm_verification_results 내용 (일부): {list(llm_verification_results.items())[:2]}") 

#결과 딕셔너리 생성 함수
def create_results_dictionary(
    final_rating: str,
    MuseSONAR_score: Optional[int],
    interpretation_message: str,
    conditional_warning: str,
    average_score: float,
    num_filtered_results: int,
    llm_yes_ratio: float,
    verified_similar_count: int,
    num_to_verify: int,
    sorted_results: List[SortedResultItemType], # SortedResultItemType은 Dict[str, Any] 형태였음
    llm_verification_results: LlmVerificationResultsMapType,
    has_combined_results_flag: bool,
    high_similarity_threshold: float
) -> AnalysisResultModel: # 반환 타입을 AnalysisResultModel로 변경
    """
    모든 평가 결과를 분석하고 구조화된 Pydantic 모델(AnalysisResultModel) 객체로 생성
    """
    logging.debug("결과 모델(AnalysisResultModel) 생성 시작.")

    # --- 상위 5개 결과 추출 및 재정렬 (기존 로직 유지) ---
    top_5_raw: List[SortedResultItemType] = sorted_results[:5]
    logging.debug(f"정렬 전 상위 결과 추출 (최대 5개): 원본 {len(top_5_raw)}개.")
# 결과 정렬 함수
    def sort_key_for_top5(item: SortedResultItemType) -> Tuple[int, float]:
        llm_status: Optional[str] = None
        text_key: Optional[str] = item.get('text') # Optional[str] 로 명시하는 것이 더 정확
        logging.debug(f"sort_key: item text (first 30char): '{text_key[:30] if text_key else 'N/A'}'")

        # if 블록 시작
        if text_key and llm_verification_results and text_key in llm_verification_results:
            llm_status = llm_verification_results[text_key][0]
            # if 조건이 참일 때의 로그 (LLM 상태가 있을 때)
            logging.debug(f"sort_key: LLM status FOUND for '{text_key[:30] if text_key else 'N/A'}': {llm_status}")
        # else 블록 시작 (if 조건이 거짓일 때)
        else:
            # if 조건이 거짓일 때의 로그 (LLM 상태가 없을 때)
            logging.debug(f"sort_key: No LLM status for '{text_key[:30] if text_key else 'N/A'}'. Reason - text_key: {bool(text_key)}, llm_results_exist: {bool(llm_verification_results)}, text_key_in_map: {text_key in llm_verification_results if text_key and llm_verification_results else 'N/A'}")
        # if-else 블록 끝

        similarity_score: float = item.get('score', 0.0)
        llm_priority: int = 0 if llm_status == "Yes" else 1
        similarity_priority: float = -similarity_score # 점수가 높을수록 작은 값이 되어 오름차순 정렬 시 앞에 옴
        logging.debug(f"sort_key: Final priorities for '{text_key[:30] if text_key else 'N/A'}': llm_p={llm_priority}, sim_p={similarity_priority:.4f} (calculated_status: {llm_status})")
        return (llm_priority, similarity_priority)

    logging.debug("상위 결과 재정렬 시작 (LLM 'Yes' 우선, 다음 유사도 내림차순)...")
    top_5_sorted_for_display: List[SortedResultItemType] = sorted(top_5_raw, key=sort_key_for_top5)
    logging.debug(f"상위 결과 재정렬 완료. 최종 표시될 결과 수: {len(top_5_sorted_for_display)}개. 정렬된 결과 (일부): {[{'text': r.get('text', '')[:20], 'score': r.get('score'), 'llm_status': llm_verification_results.get(r.get('text', ''), (None,))[0]} for r in top_5_sorted_for_display]}") # 정렬 결과 확인 로그 추가

    # --- Pydantic 모델 객체 생성 ---
    logging.debug("Pydantic 모델 데이터 구성 시작...")

    # 1. MetricModel 생성
    metrics_data = MetricModel(
        concept_density_percentage=average_score if num_filtered_results > 0 else None,
        evidence_discovery_rate_percentage=llm_yes_ratio * 100 if num_to_verify > 0 else None,
        evidence_count=verified_similar_count,
        verification_attempts=num_to_verify,
        relevant_search_results_count=num_filtered_results,
        combined_results_found=has_combined_results_flag,
        verification_threshold_percentage=high_similarity_threshold * 100
    )

    # 2. top_similar_results 리스트 생성 (SimilarResultModel 객체 포함)
    top_results_models: List[SimilarResultModel] = []
    for i, r in enumerate(top_5_sorted_for_display):
        llm_ver_data: Optional[LlmVerificationModel] = None
        text_key = r.get('text')
        if text_key and llm_verification_results and text_key in llm_verification_results: 
            status, reason = llm_verification_results[text_key]
            llm_ver_data = LlmVerificationModel(status=status, reason=reason)

        similar_result = SimilarResultModel(
            rank=i + 1,
            similarity_percentage=r.get('score', 0.0) * 100,
            content_preview=r.get('text', '')[:150] + "..." if len(r.get('text', '')) > 150 else r.get('text', ''),
            link=r.get('link'), # .get() 사용으로 None 가능성 처리
            source=r.get('source', 'Unknown'),
            llm_verification=llm_ver_data
        )
        top_results_models.append(similar_result)

    # 3. 최종 AnalysisResultModel 생성
    analysis_result_model = AnalysisResultModel(
        rating=final_rating,
        score=MuseSONAR_score, # Optional[int]
        interpretation=interpretation_message,
        warning=conditional_warning if conditional_warning else None, # 빈 문자열이면 None
        metrics=metrics_data, # 위에서 생성한 MetricModel 객체
        top_similar_results=top_results_models # 위에서 생성한 SimilarResultModel 리스트
    )

    logging.info("결과 모델(AnalysisResultModel) 생성 완료.")
    return analysis_result_model # 딕셔너리 대신 Pydantic 모델 객체 반환

# 아이디어 분석 함수
def analyze_idea(user_text_original: str,
                sbert_model: SentenceTransformer,
                gemini_model: Optional[GenerativeModel]) -> AnalysisResultModel:
    """
    입력된 아이디어 텍스트의 고유성을 분석하고 결과를 AnalysisResultModel 객체로 반환합니다.
    오류 발생 시 AnalysisResultModel의 'error' 필드에 메시지를 담아 반환합니다.
    """
    logging.info(f"===== MuseSonar 분석 시작 =====")
    logging.info(f"입력 아이디어: '{user_text_original}'")
    # --- 1. 기본 프롬프트 주입 방어: 입력 텍스트 필터링 (일부 공개) ---
    filtered_user_text = user_text_original
    # 매우 기본적인 위험 패턴 예시 (정규식 사용, 대소문자 무시)
    # 예: "ignore ... instructions", "disregard ... prompt" 등
    # 실제로는 더 정교한 패턴 필요할 수 있음
    potential_injection_patterns = [
        r'ignore\s+(all\s+)?(previous|prior|above|following)\s+instructions?',
        r'disregard\s+(all\s+)?(previous|prior|above|following)\s+instructions?', ...
        # 추가적인 위험 패턴들 추가 예정
    ]

    for pattern in potential_injection_patterns:
        # re.IGNORECASE 플래그로 대소문자 무시
        # 패턴 발견 시, 해당 부분을 제거하고 경고 로그 남김
        filtered_user_text, num_subs = re.subn(pattern, '', filtered_user_text, flags=re.IGNORECASE)
        if num_subs > 0:
            logging.warning(f"잠재적 프롬프트 주입 패턴 감지 및 제거: '{pattern}'")
            ...(일부 공개)

    # --- 2. 필수 설정 확인 (조기 종료 조건) ---
    if not GOOGLE_SEARCH_API_KEY or not SEARCH_ENGINE_ID: 
        # 이 조건은 스크립트 시작 시 검증을 통과했다면 이론적으로 발생하지 않아야 함
        # 하지만 안전 장치로 남겨둠
        error_msg = "환경 변수 오류: Google Search API 키 또는 Engine ID가 유효하지 않습니다. (.env 파일 확인 필요)"
        logging.critical(error_msg)
        return AnalysisResultModel(error=error_msg)
    # SBERT 모델 확인
    if not sbert_model:
        error_msg = "모델 오류: SBERT 모델이 로드되지 않았습니다."
        logging.critical(error_msg)
        return AnalysisResultModel(error=error_msg)
    # Gemini 모델 로깅
    if not gemini_model:
        logging.warning("LLM 검증 건너뜀: Gemini 모델이 로드되지 않았거나 설정되지 않았습니다.")

    # --- 3. 변수 초기화 ---
    search_results_data_raw: GoogleResultType = []
    patent_data: KiprisResultType = []
    combined_data_raw: List[Dict[str, str]] = []
    combined_data: List[Dict[str, str]] = []

    # --- 4. 동시 검색 실행 ---
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            logging.info("--- Google 및 KIPRIS 검색 동시 요청 시작 ---")

            # Google 검색 작업 제출 
            future_google = executor.submit(google_search, user_text_to_analyze)
            logging.info("Google 검색 작업 제출됨.")

            # KIPRIS 검색 작업 제출 
            future_kipris = None
            if KIPRIS_API_KEY: 
                future_kipris = executor.submit(search_kipris_patents, user_text_to_analyze)
                logging.info("KIPRIS 검색 작업 제출됨.")
            else:
                logging.warning("KIPRIS API 키가 없어 특허 검색 작업을 건너뜁니다.")

            # Google 결과 가져오기 (
            logging.debug("Google 검색 결과 기다리는 중...")
            search_results_data_raw = future_google.result() # 여기서 예외 발생 시 아래 except 블록으로 이동
            logging.info(f"Google 검색 결과 수신 완료 ({len(search_results_data_raw)}개).")

            # KIPRIS 결과 가져오기 
            if future_kipris:
                logging.debug("KIPRIS 검색 결과 기다리는 중...")
                patent_data = future_kipris.result() # 여기서 예외 발생 시 아래 except 블록으로 이동
                logging.info(f"KIPRIS 검색 결과 수신 완료 ({len(patent_data)}개).")

        logging.info("--- 모든 검색 요청 처리 완료 ---")

    except Exception as e:
        # 동시 검색 중 어떤 이유로든 예외 발생 시
        error_msg = f"외부 데이터 검색 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        return AnalysisResultModel(error=error_msg) # 오류 모델 반환

    # --- 4. 결과 통합 및 중복 제거 ---
    logging.debug("웹/특허 검색 결과 통합 시작...")
    # (타입 체크 및 통합 로직은 이전과 동일하게 유지)
    if isinstance(search_results_data_raw, list):
        combined_data_raw.extend(search_results_data_raw)
    # ... (patent_data 통합 로직) ...
    logging.debug(f"결과 통합 완료 (통합 전: {len(combined_data_raw)}개).")

    seen_texts: set[str] = set()
    combined_data = []
    skipped_non_dict: int = 0
    for item in combined_data_raw:
        if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
            text_content: str = item['text']
            if text_content not in seen_texts:
                combined_data.append(item)
                seen_texts.add(text_content)
        else:
            logging.warning(f"통합 데이터 항목 타입 오류 또는 'text' 키/값 없음: {item}")
            skipped_non_dict += 1
    logging.info(f"중복 제거 후 분석 대상 데이터: {len(combined_data)}개 (원래 {len(combined_data_raw)}개, 형식 오류 {skipped_non_dict}개 제외)")

    # --- 5. 분석 가능 데이터 없음 ("정보 부족") 처리 ---
    if not combined_data:
        # 검색은 시도했으나 유효한 분석 대상 데이터가 없는 경우
        logging.warning("분석 결과: 관련성 있는 웹/특허 정보를 찾을 수 없거나 유효한 정보가 없습니다.")
        final_rating = "정보 부족"
        interpretation_message = "아이디어와 관련된 웹 또는 특허 정보를 찾을 수 없거나 유효한 정보가 없습니다. 아이디어를 다시 입력하거나 키워드를 확인해 보세요."
        # MetricModel 기본값 생성 (필수 필드 위주)
        metrics = MetricModel(
            evidence_count=0, verification_attempts=0, relevant_search_results_count=0,
            combined_results_found=bool(combined_data_raw), # 검색 시도 여부
            verification_threshold_percentage=config.HIGH_SIMILARITY_THRESHOLD * 100
        )
        # 정보 부족 상태 모델 반환 (오류는 아님)
        result_model = AnalysisResultModel(
            rating=final_rating,
            interpretation=interpretation_message,
            metrics=metrics
            # score, warning, top_similar_results는 기본값(None, None, []) 사용
        )
        logging.warning("===== MuseSonar 분석 완료 (결과 정보 부족) =====")
        return result_model

    # --- 6. 유사도 분석 및 LLM 검증 ---
    average_score: float = 0.0
    num_filtered_results: int = 0
    num_to_verify: int = 0
    verified_similar_count: int = 0
    llm_verification_results: LlmVerificationResultsMapType = {}
    all_results_with_scores: List[SortedResultItemType] = []
    max_similarity_score: float = 0.0
    sorted_results: List[SortedResultItemType] = []

    try:
        combined_texts: List[str] = [item['text'] for item in combined_data if 'text' in item]

        if not combined_texts: # 이 경우는 위 combined_data 체크에서 걸러졌어야 하지만 방어 코드
            raise ValueError("분석할 텍스트 데이터가 없습니다 (combined_texts 비어있음).")

        logging.info("--- SBERT 유사도 계산 및 분석 시작 ---")
        # 임베딩 계산
        logging.debug("SBERT 모델 임베딩 계산 시작...")
        embeddings = sbert_model.encode([user_text_to_analyze] + combined_texts, convert_to_tensor=True)
        user_vec = embeddings[0]
        result_vecs = embeddings[1:]
        logging.debug(f"임베딩 계산 완료. 코사인 유사도 계산 시작...")
        cos_scores: List[float] = util.cos_sim(user_vec, result_vecs)[0].cpu().tolist()
        logging.debug(f"코사인 유사도 계산 완료.")

        # 모든 결과에 점수 등 매핑
        all_results_with_scores = []
        for i, score in enumerate(cos_scores):
            if i < len(combined_data) and isinstance(combined_data[i], dict):
                all_results_with_scores.append({
                    'text': combined_data[i].get('text', ''), 'score': score,
                    'link': combined_data[i].get('link', ''), 'source': combined_data[i].get('source', 'Unknown')
                })
            else:
                logging.error(f"결과 매핑 오류: 인덱스 {i} 또는 combined_data[{i}] 형식 오류")

        # 관련성 필터링
        filtered_results_list = [r for r in all_results_with_scores if r.get('score', 0.0) >= config.RELEVANCE_THRESHOLD]
        num_filtered_results = len(filtered_results_list)
        logging.info(f"유사도 필터링 완료: {num_filtered_results}개 결과 >= {config.RELEVANCE_THRESHOLD*100:.0f}%")

        # --- 7. 필터링 결과 0개 ("관련성 높은 정보 부족") 처리 ---
        if num_filtered_results == 0:
            logging.warning(f"분석 결과: 관련성 높은(유사도 >= {config.RELEVANCE_THRESHOLD*100:.0f}%) 웹/특허 정보를 찾을 수 없음.")
            final_rating = "정보 부족"
            interpretation_message = f"아이디어와 관련성이 높은(유사도 {config.RELEVANCE_THRESHOLD*100:.0f}% 이상) 웹 또는 특허 정보를 찾을 수 없습니다. 아이디어를 더 구체화하거나 다른 키워드로 시도해 보세요."
            # 유사도 높은 결과는 없지만, 전체 결과는 정렬해서 참고용으로 제공 가능
            sorted_results = sorted(all_results_with_scores, key=lambda x: x.get('score', 0.0), reverse=True)
            # MetricModel 생성
            metrics = MetricModel(
                evidence_count=0, verification_attempts=0,
                relevant_search_results_count=num_filtered_results, # 0
                combined_results_found=True, # 검색 결과 자체는 있었음
                verification_threshold_percentage=config.HIGH_SIMILARITY_THRESHOLD * 100
            )
            # 정보 부족 상태 모델 반환
            result_model = AnalysisResultModel(
                rating=final_rating,
                interpretation=interpretation_message,
                metrics=metrics,
                top_similar_results=[ # 상위 5개 정보는 모델에 맞게 변환하여 전달
                    SimilarResultModel(
                        rank=i+1, similarity_percentage=r.get('score', 0.0)*100,
                        content_preview=r.get('text', '')[:150] + "...", link=r.get('link'), source=r.get('source', 'Unknown')
                    ) for i, r in enumerate(sorted_results[:5]) # LLM 검증은 없으므로 None
                ]
            )
            logging.warning("===== MuseSonar 분석 완료 (관련성 높은 정보 부족) =====")
            return result_model

        # --- 8. 관련성 높은 결과 기반 통계 계산 및 LLM 검증(비공개) ---
        ...

    except Exception as e:
        # SBERT 분석, LLM 검증 등 이 블록 내에서 발생하는 모든 예외 처리
        error_msg = f"아이디어 분석 처리 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        return AnalysisResultModel(error=error_msg) # 오류 모델 반환

    # --- 9. 최종 평가 및 결과 생성(비공개) ---
    

    try:
        #--- 10. 등급 및 해석 결정 ---
        final_rating, interpretation_message, conditional_warning = determine_originality(
            average_score, verified_similar_count, llm_yes_ratio, has_combined_results_flag, max_similarity_score
        )
        # 점수 계산
        if final_rating not in ["평가 불가 (오류)", "정보 부족"]: # 오류/정보부족 아닐 때만 점수 계산
            score_int = calculate_MuseSONAR_score(final_rating, average_score, llm_yes_ratio, num_to_verify)
            MuseSONAR_score = score_int if score_int != -1 else None
        else:
            MuseSONAR_score = None # 점수 계산 불가

    except Exception as e:
        # 최종 평가/점수 계산 중 예외 발생 시
        error_msg = f"최종 평가 또는 점수 계산 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        # 오류 모델을 반환하거나, rating/interpretation을 오류 상태로 두고 아래 모델 생성으로 넘어갈 수 있음
        # 여기서는 오류 모델을 반환하는 것이 더 명확할 수 있음
        return AnalysisResultModel(error=error_msg)

    # --- 11. 최종 결과 모델 생성 (create_results_dictionary 호출) ---
    try:
        results_model = create_results_dictionary(
            final_rating, MuseSONAR_score, interpretation_message, conditional_warning,
            average_score, num_filtered_results, llm_yes_ratio, verified_similar_count, num_to_verify,
            sorted_results, llm_verification_results, has_combined_results_flag, config.HIGH_SIMILARITY_THRESHOLD
        )
        logging.info("===== MuseSonar 분석 완료 =====")
        return results_model # 최종 성공 시 결과 모델 반환

    except Exception as e:
        # 결과 모델 생성 중 예외 발생 시 (드문 경우)
        error_msg = f"최종 결과 모델 생성 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        return AnalysisResultModel(error=error_msg) # 오류 모델 반환

# --- 메인 실행 로직 (테스트용) ---
if __name__ == "__main__":
    logging.info("--- __main__ 블록 실행: 테스트용 모델 로딩 시작 ---")
    sbert_model_test: Optional[SentenceTransformer] = None # 타입 힌트 추가
    try:
        sbert_model_test = SentenceTransformer('jhgan/ko-sroberta-multitask')
        logging.info("테스트용 Sentence Transformer 모델 로딩 완료 (jhgan/ko-sroberta-multitask).")
    except Exception as e:
        logging.critical(f"테스트용 SBERT 모델 로딩 실패! 분석 불가.", exc_info=True)

    gemini_model_test: Optional[GenerativeModel] = None # 타입 힌트 추가
    if GOOGLE_API_KEY_GEMINI:
        try:
            genai.configure(api_key=GOOGLE_API_KEY_GEMINI)
            default_model = 'models/gemini-2.0-flash' 
            model_name: str = os.getenv('GEMINI_MODEL_NAME', default_model)
            logging.info(f"사용할 Gemini 모델: {model_name} (환경 변수 'GEMINI_MODEL_NAME' 또는 기본값)")
            gemini_model_test = genai.GenerativeModel(model_name) # 수정된 model_name 사용
            logging.info(f"테스트용 Gemini API 설정 완료 ({model_name} 모델 사용).")
        except Exception as e:
            logging.warning(f"테스트용 Gemini API 설정 중 오류 발생. LLM 검증 기능 비활성화됨.", exc_info=True)
            gemini_model_test = None
    else:
        logging.warning("테스트용 Gemini API 키가 없어 LLM 검증이 비활성화됩니다.")

    logging.info("--- 테스트용 모델 로딩 완료 ---")

    test_idea: str = "고양이 사료 영양분석 어플." #테스트용 아이디어
    logging.info(f"--- MuseSonar.py 직접 실행 테스트 시작 (아이디어: '{test_idea}') ---")

    analysis_result: AnalysisResultModel # 반환 타입은 AnalysisResultModel

    if sbert_model_test:
        analysis_result = analyze_idea(test_idea, sbert_model_test, gemini_model_test)
    else:
        # SBERT 모델 로딩 실패 시 오류 모델 생성
        analysis_result = AnalysisResultModel(error="테스트용 SBERT 모델 로딩 실패로 분석 불가")

    # --- 결과 처리: analysis_result.error 필드 확인 ---
    if analysis_result.error:
        # 오류가 있는 경우
        logging.error(f"테스트 실행 중 오류 발생: {analysis_result.error}")
    elif analysis_result.rating is None: # 정상 종료되었으나 분석 정보 부족 등 rating이 없는 경우
        logging.warning("분석은 완료되었으나 유의미한 결과를 얻지 못했습니다. (예: 정보 부족)")
        # 이 경우에도 dashboard는 호출하여 제한된 정보라도 표시 가능
        display_results_dashboard(analysis_result)
    else:
        # 정상적으로 분석 결과가 나온 경우
        display_results_dashboard(analysis_result) # AnalysisResultModel 객체를 그대로 전달

    logging.info("--- MuseSonar.py 직접 실행 테스트 완료 ---")
    logging.info("==================== MuseSonar 스크립트 종료 ====================")