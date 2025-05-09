# app.py (수정된 부분)

import markdown
from flask import Flask, render_template, request, redirect, url_for
import os
import logging

# --- MuseSonar 관련 모듈 임포트 ---
try:
    from MuseSONAR_public import analyze_idea
    from pydantic_models import AnalysisResultModel
    muse_sonar_imported = True
except ImportError as e:
    logging.error(f"MuseSonar 모듈 임포트 실패: {e}. 분석 기능을 사용할 수 없습니다.")
    muse_sonar_imported = False
    class AnalysisResultModel:
        def __init__(self, error=None, rating=None, score=None, interpretation=None, warning=None, metrics=None, top_similar_results=None):
            self.error = error or "MuseSonar 모듈 로딩 실패"
            self.rating = rating
            self.score = score
            self.interpretation = interpretation
            self.warning = warning
            self.metrics = metrics
            self.top_similar_results = top_similar_results or []
    def analyze_idea(*args, **kwargs):
        return AnalysisResultModel()

# --- 모델 로딩 ---
sbert_model = None
gemini_model = None

if muse_sonar_imported:
    logging.info("--- 앱 시작: 모델 로딩 시도 ---")
    try:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        logging.info("SBERT 모델(jhgan/ko-sroberta-multitask) 로딩 완료.")
    except Exception as e:
        logging.critical(f"SBERT 모델 로딩 실패! 분석 기능 비활성화.", exc_info=True)
        sbert_model = None

    try:
        from dotenv import load_dotenv
        load_dotenv()
        google_api_key_gemini = os.getenv('GOOGLE_API_KEY_GEMINI')
        if google_api_key_gemini:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key_gemini)
            default_model = 'models/gemini-2.0-flash'
            model_name = os.getenv('GEMINI_MODEL_NAME', default_model)
            logging.info(f"사용할 Gemini 모델: {model_name}")
            gemini_model = genai.GenerativeModel(model_name)
            logging.info(f"Gemini 모델({model_name}) 로딩 및 설정 완료.")
        else:
            logging.warning("환경 변수 'GOOGLE_API_KEY_GEMINI' 없음. LLM 검증 비활성화.")
            gemini_model = None
    except Exception as e:
        logging.warning(f"Gemini 모델 설정 중 오류 발생. LLM 검증 비활성화.", exc_info=True)
        gemini_model = None
    logging.info("--- 모델 로딩 완료 ---")
else:
    logging.error("MuseSonar 모듈 로딩 실패로 모델 로딩 건너뜀.")


app = Flask(__name__)

log_format = '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.FileHandler("musessonar_flask_app.log", encoding='utf-8'),
    logging.StreamHandler()
])
app.logger.info("Flask 애플리케이션 시작 및 로깅 설정 완료.")



@app.route('/')
def index():
    app.logger.info("메인 페이지 요청됨.")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    idea_text = request.form.get('idea_text', '').strip()
    app.logger.info(f"분석 요청 수신: '{idea_text[:50]}...'")
    # 1. 변수 초기화
    interpretation_html = None
    warning_html = None

    if not idea_text:
        app.logger.warning("입력된 아이디어가 없습니다.")
        return redirect(url_for('index'))

    if not muse_sonar_imported or not sbert_model:
        app.logger.error("분석 수행 불가: MuseSonar 모듈 또는 SBERT 모델 로드 실패.")
        error_result = AnalysisResultModel(
            error="핵심 분석 모듈 또는 SBERT 모델 로딩에 실패하여 분석을 수행할 수 없습니다. 서버 로그를 확인해주세요."
        )
        # 오류 발생 시에도 초기화된 HTML 변수들을 전달
        return render_template('results.html',
                            result=error_result,
                            user_idea=idea_text,
                            interpretation_html=interpretation_html, # None 전달
                            warning_html=warning_html) # None 전달

    try:
        app.logger.info("MuseSonar.analyze_idea 함수 호출 시작...")
        analysis_result = analyze_idea(idea_text, sbert_model, gemini_model)
        app.logger.info("MuseSonar.analyze_idea 함수 호출 완료.")

        if hasattr(analysis_result, 'interpretation') and analysis_result.interpretation:
            interpretation_html = markdown.markdown(analysis_result.interpretation, extensions=['nl2br'])

        if hasattr(analysis_result, 'warning') and analysis_result.warning:
            warning_html = markdown.markdown(analysis_result.warning, extensions=['nl2br'])

        app.logger.info(f"분석 결과: Rating={analysis_result.rating}, Score={analysis_result.score}, Error='{analysis_result.error}'")

        return render_template('results.html',
                            result=analysis_result,
                            user_idea=idea_text,
                            interpretation_html=interpretation_html,
                            warning_html=warning_html)

    except Exception as e:
        app.logger.critical(f"Flask /analyze 라우트 처리 중 심각한 오류 발생.", exc_info=True)
        critical_error_result = AnalysisResultModel(
            error=f"분석 요청 처리 중 예상치 못한 오류가 발생했습니다: {e}"
        )
        # 2. except 블록에서도 명시적으로 None 전달 (또는 빈 문자열 등)
        return render_template('results.html',
                            result=critical_error_result,
                            user_idea=idea_text,
                            interpretation_html=None, # 명시적으로 None 전달
                            warning_html=None)      # 명시적으로 None 전달

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)