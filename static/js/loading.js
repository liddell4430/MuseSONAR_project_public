// static/js/loading.js

/* ────────────────────────────────────────────── */
/* 1) 전역 변수 하나 추가해 인터벌 핸들 저장     */
let dotTimer = null;
/* ────────────────────────────────────────────── */

/* 2) 점 애니메이션을 시작하는 함수 */
function startDotAnimation(baseText) {
    baseText = baseText.replace(/\.*$/, '');
    let dots = 0;
    loadingText.textContent = baseText;
    dotTimer = setInterval(() => {
        dots = (dots % 3) + 1;
        loadingText.textContent = baseText + '.'.repeat(dots);
    }, 500);
}

/* 3) 점 애니메이션을 멈추는 함수 */
function stopDotAnimation() {
    if (dotTimer) {
        clearInterval(dotTimer);
        dotTimer = null;
    }
}

const form = document.getElementById('ideaForm');
const button = document.getElementById('analyzeButton');
const loadingContainer = document.getElementById('loadingContainer');
const loadingRobotImage = document.getElementById('loadingRobotImage');
const loadingBulbImage = document.getElementById('loadingBulbImage');
const loadingText = document.getElementById('loadingText');
const textarea = form.querySelector('textarea[name="idea_text"]');

const loadingSteps = [
    { duration: 4000, image: 'thinking.png', text: '사용자님의 아이디어를 웹과 특허에서 찾고 있어요...' },
    { duration: 8000, image: 'thinking.png', text: 'AI가 개념의 유사성을 분석하고 있어요...' },
    { duration: 3000, image: 'thinking.png', text: '분석 결과를 바탕으로 점수를 계산 중입니다...' },
    { duration: 0, image: 'solve.png', text: '분석 완료! 결과를 곧 보여드릴게요! 😊' }
];

let currentStep = 0;
let timerId = null;
let previousImageSrc = ''; // ✨ 변수명 변경 및 초기값 (src 전체 경로 저장)

function showLoadingStep(stepIndex) {
    console.log(`--- Step ${stepIndex} 시작 ---`);

    if (stepIndex >= loadingSteps.length) {
        if (timerId) clearTimeout(timerId);
        console.log('모든 단계 완료 또는 인덱스 오류');
        return;
    }

    const step = loadingSteps[stepIndex];
    const currentImageSrc = "/static/" + step.image; // ✨ 현재 이미지 src 경로
    const isLastStep = stepIndex === loadingSteps.length - 1;

    // ✨ 로봇 이미지 애니메이션 조건 수정
    const shouldAnimateRobotImage = (currentImageSrc.includes('solve.png') || (previousImageSrc.includes('solve.png') && currentImageSrc !== previousImageSrc));
    const imageFileChanged = (currentImageSrc !== previousImageSrc); // ✨ 단순 파일명 변경 여부 (전구 제어 등에 사용)

    const showBulb = (currentImageSrc.includes('solve.png'));
    console.log(`현재 상태: currentImageSrc=${currentImageSrc}, shouldAnimateRobotImage=${shouldAnimateRobotImage}, imageFileChanged=${imageFileChanged}, showBulb=${showBulb}, previousImageSrc=${previousImageSrc}`);

    // --- Fade-out 처리 ---
    // ✨ 로봇 이미지는 shouldAnimateRobotImage 조건에 따라 애니메이션 적용
    if (shouldAnimateRobotImage) {
        loadingRobotImage.classList.add('hidden');
        console.log('[FadeOut] 로봇 이미지 숨김 (.hidden 추가) - 애니메이션 조건 충족');
    }
    // ✨ 전구 이미지는 파일이 변경되었고, 이전 이미지가 solve.png 였을 때만 숨김 처리 (이미지 변경 시 항상 숨기는 것은 아님)
    if (imageFileChanged && previousImageSrc.includes('solve.png')) {
        loadingBulbImage.style.display = 'none';
        loadingBulbImage.classList.add('hidden');
        console.log('[FadeOut] 전구 이미지 숨김 (display:none, .hidden 추가) - solve.png에서 다른 이미지로 변경 시');
    }
    loadingText.classList.add('hidden'); // 텍스트는 항상 페이드 아웃
    console.log('[FadeOut] 텍스트 숨김 (.hidden 추가)');

    // --- 내용 변경 및 Fade-in 처리 ---
    setTimeout(() => {
        console.log(`[Timeout ${stepIndex}] 콜백 실행: 내용 변경 및 fade-in 시작`);

        // ✨ 이미지 파일이 실제로 변경되었을 때만 src 업데이트
        if (imageFileChanged) {
            loadingRobotImage.src = currentImageSrc;
            console.log(`[Timeout ${stepIndex}] 로봇 이미지 src 변경: ${currentImageSrc}`);
        }

        stopDotAnimation();
        startDotAnimation(step.text);

        requestAnimationFrame(() => {
            console.log(`[RAF ${stepIndex}] 콜백 실행: fade-in 클래스 제거`);

            // ✨ 로봇 이미지는 shouldAnimateRobotImage 조건에 따라 애니메이션 적용
            if (shouldAnimateRobotImage) {
                loadingRobotImage.classList.remove('hidden');
                console.log(`[RAF ${stepIndex}] 로봇 이미지 보임 (.hidden 제거) - 애니메이션 조건 충족`);
            } else if (imageFileChanged) {
                // ✨ 애니메이션 조건은 아니지만, 이미지 파일 자체가 바뀌었다면 (예: thinking.png 유지)
                // src는 이미 위에서 변경되었으므로, hidden 클래스가 없어야 이미지가 보임.
                // (만약 이전 스텝에서 hidden이 추가된 상태였다면 제거 필요. 지금 로직에서는 텍스트만 애니메이션되므로 로봇은 계속 보여야 함)
                // 이 부분은 로봇 이미지가 애니메이션 없이 계속 보여야 하는 경우를 위한 안전장치.
                // 다만, 현재 로봇 이미지는 solve.png일때만 애니메이션 하므로, 그 외에는 hidden 상태가 되면 안됨.
                // form submit 시 초기 상태에서 hidden이 없음을 보장해야 함.
                // 지금 로직에서는 shouldAnimateRobotImage가 false이면 hidden을 추가하지 않았으므로, 특별히 여기서 remove할 필요는 없어보임.
                // 하지만 명시적으로 계속 보이게 하려면:
                // loadingRobotImage.classList.remove('hidden');
                 console.log(`[RAF ${stepIndex}] 로봇 이미지 애니메이션 없음, src는 변경됨 (${currentImageSrc}), hidden 상태 유지 또는 제거 확인 필요`);
            }


            // ✨ 전구 표시 로직은 이미지 파일이 변경되었을 때, 그리고 그 이미지가 solve.png일 때만 실행
            if (imageFileChanged && showBulb) {
                console.log(`[RAF ${stepIndex}] >>> 전구 표시 로직 진입 (imageFileChanged and showBulb is true)`);
                loadingBulbImage.style.display = 'block';
                console.log(`[RAF ${stepIndex}] 전구 display: block 설정됨`);
                setTimeout(() => {
                    console.log(`[Delay 750ms] 전구 fade-in 시작`);
                    loadingBulbImage.style.opacity = '1'; // hidden 클래스 제어 대신 opacity 직접 제어
                }, 750); // 전구 페이드인 시간은 그대로 0.75초 유지
            } else if (imageFileChanged && !showBulb && previousImageSrc.includes('solve.png')) {
                // ✨ solve.png 였다가 다른 이미지로 바뀌는 경우, 위에서 display:none 처리했으므로 여기선 특별히 할 것 없음.
                console.log(`[RAF ${stepIndex}] 전구 표시 안 함 (solve.png에서 다른 이미지로 변경됨)`);
            }


            loadingText.classList.remove('hidden'); // 텍스트는 항상 페이드 인
            console.log(`[RAF ${stepIndex}] 텍스트 보임 (.hidden 제거)`);

            // ✨ previousImageSrc 업데이트 위치는 그대로 유지 (실제 src 값으로)
            previousImageSrc = currentImageSrc;
            console.log(`[RAF ${stepIndex}] previousImageSrc 업데이트됨: ${previousImageSrc}`);

            if (!isLastStep && step.duration > 0) {
                if (timerId) clearTimeout(timerId);
                console.log(`[RAF ${stepIndex}] 다음 단계 타이머 설정: ${step.duration}ms 후`);
                timerId = setTimeout(() => {
                    currentStep++;
                    showLoadingStep(currentStep);
                }, step.duration);
            } else if (isLastStep) {
                stopDotAnimation();
                loadingText.textContent = step.text; // 마지막 텍스트는 점 애니메이션 없이 고정
                console.log(`[RAF ${stepIndex}] 마지막 단계, 타이머 설정 안 함`);
            } else { // duration 0 인 경우
                console.log(`[RAF ${stepIndex}] duration 0, 바로 다음 단계 호출`);
                currentStep++;
                showLoadingStep(currentStep);
            }
        });
    }, 300); // 내용 변경과 fade-in 사이의 딜레이 (기존 300ms 유지)
}

form.addEventListener('submit', function(event) {
    console.log('--- Form submitted! ---');
    // event.preventDefault(); // ✨ 실제 제출을 막으려면 주석 해제 (테스트용)
    if (textarea.value.trim() === '') {
        alert("아이디어를 입력해주세요!");
        console.log('아이디어 없음, 분석 중단');
        event.preventDefault(); // ✨ 아이디어 없으면 제출 막기
        return;
    }

    console.log('로딩 UI 표시 및 초기화 시작');
    loadingContainer.style.display = 'block';
    button.disabled = true;
    button.textContent = '분석 중...';

    if (timerId) clearTimeout(timerId);
    currentStep = 0;
    console.log('타이머 초기화, currentStep=0 설정');

    const firstStep = loadingSteps[0];
    console.log('첫 번째 단계 즉시 설정');
    loadingRobotImage.src = "/static/" + firstStep.image; // ✨ 초기 이미지 src 설정
    // loadingText.textContent = firstStep.text; // ✨ 첫 텍스트는 startDotAnimation으로 설정
    stopDotAnimation(); // 기존 애니메이션 중지 (혹시 모를 경우 대비)
    startDotAnimation(firstStep.text); // ✨ 첫 텍스트 점 애니메이션 시작

    loadingRobotImage.classList.remove('hidden'); // ✨ 처음에는 로봇 이미지 보이게
    loadingText.classList.remove('hidden'); // ✨ 처음에는 텍스트 보이게
    loadingBulbImage.style.display = 'none';
    loadingBulbImage.style.opacity = '0'; // ✨ 초기 opacity 0으로 명시 (CSS에서 .hidden 제거 시 transition 위함)
    // loadingBulbImage.classList.add('hidden'); // hidden 클래스보다는 opacity로 제어하는 것이 나을 수 있음

    previousImageSrc = "/static/" + firstStep.image; // ✨ 첫 이미지 src 저장
    console.log(`첫 단계 설정 완료: previousImageSrc=${previousImageSrc}`);

    if (firstStep.duration > 0) {
        console.log(`첫 단계 후 다음 단계 타이머 설정: ${firstStep.duration}ms`);
        timerId = setTimeout(() => {
            currentStep++;
            showLoadingStep(currentStep);
        }, firstStep.duration);
    } else {
        console.log('첫 단계 duration 0, 즉시 다음 단계 호출');
        currentStep++;
        showLoadingStep(currentStep);
    }
});

window.addEventListener('pagehide', function() {
    console.log('Page hide 이벤트 발생, 타이머 정리');
    if (timerId) clearTimeout(timerId);
    stopDotAnimation(); // 페이지 벗어날 때 점 애니메이션도 중지
});