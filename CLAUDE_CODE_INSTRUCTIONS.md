# 상토 AI 프로젝트 — Claude Code 자율 수행 지시서
> 이 파일을 Claude Code 터미널에서 아래 명령으로 실행하세요:
> `$ claude "이 파일의 지시사항을 처음부터 끝까지 순서대로 자율 수행해줘: $(cat CLAUDE_CODE_INSTRUCTIONS.md)"`

---

## ⚙️ 전제 조건 (Claude Code가 자동 확인할 것)

- 작업 디렉토리: `C:\ai_workspace\sh-ai-model`
- 기존 완성 데이터: `C:\ai_workspace\sh-ai-model\data\raw\weather`에 있음
- Python 3.10+ 설치 확인
- 인터넷 연결 확인 (크롤링 필요)

---

## PHASE 1 — 프로젝트 초기화 및 환경 구성

### 1-1. 디렉토리 구조 생성

```
다음 디렉토리를 자동 생성하라. 이미 존재하면 건너뛴다.

C:\ai_workspace\sh-ai-model
```


### 1-2. 패키지 전체 설치

```
아래 패키지를 순서대로 설치하라.
설치 실패 시 에러를 logs/install_errors.log에 기록하고 다음 패키지로 넘어간다.
각 설치 완료 후 import 테스트까지 수행한다.
```

```bash
# 기본 데이터 처리
pip install pandas numpy openpyxl xlrd

# 크롤링
pip install requests beautifulsoup4 selenium webdriver-manager
pip install newspaper3k feedparser

# 텍스트 마이닝
pip install konlpy
pip install kiwipiepy        # KoNLPy 대안 (설치 실패시 대체)
pip install wordcloud matplotlib Pillow
pip install networkx

# 최신 NLP (BERTopic)
pip install sentence-transformers
pip install bertopic
pip install umap-learn hdbscan

# 토픽 모델링 (LDA 백업)
pip install gensim pyLDAvis

# 시각화
pip install plotly seaborn

# DB
pip install sqlalchemy

# 스케줄러
pip install apscheduler

# 보고서
pip install jinja2 fpdf2

# 검증
python -c "import pandas, requests, bs4, wordcloud, networkx, sqlalchemy; print('✅ 핵심 패키지 정상')"
```

---

## PHASE 2 — 네이버 뉴스 크롤러 구축

### 2-1. 크롤러 스크립트 생성

**파일 경로:** `C:\ai_workspace\sh-ai-model\crawlers\naver_news_crawler.py`

```
아래 사양으로 크롤러를 작성하라. 주석은 한국어로 작성한다.

[크롤링 대상 키워드 목록 - 총 3그룹]

그룹A - 상토 관련:
  "수도용상토", "벼모상토", "육묘상토", "상토부족", "상토불량",
  "상토민원", "모상토", "상토출고", "상토수요"

그룹B - 재배/농업 관련:
  "모내기시기", "이앙시기", "모내기일정", "벼이앙", "수도작재배",
  "못자리", "벼모기르기", "파종시기", "육묘방법"

그룹C - 농업 행정/정책:
  "지자체보조상토", "상토지원사업", "농협상토", "보조상토",
  "영농자재지원", "농협검수", "못자리상토지원사업", "벼육묘용", "상토보조"

[수집 기간]
  - 시작: 2005년 1월 1일
  - 종료: 2026년 4월 (현재)

[수집 데이터 컬럼]
  keyword, date, title, press, content_summary, url, crawled_at

[저장 형식]
  - SQLite: data/master_db/news_corpus.db (테이블명: news_articles)
  - CSV 백업: data/raw/news_raw_{keyword}_{YYYYMM}.csv

[중복 처리]
  - url 기준 중복 제거
  - title 기준 코사인 유사도 0.8 이상이면 최신 것만 보관

[에러 처리]
  - 요청 실패 시 3초 대기 후 3회 재시도
  - 재시도 모두 실패 시 logs/crawl_errors.log에 기록 후 다음으로
  - 요청 간격: 1~2초 랜덤 (서버 부하 방지)

[진행 상황 출력]
  - 키워드별 수집 건수 실시간 출력
  - 완료 시 총 수집 건수 요약 출력
```

---

### 2-2. 농촌진흥청 / 비정형 데이터 크롤러

**파일 경로:** `C:\ai_workspace\sh-ai-model\crawlers\agri_data_crawler.py`

```
아래 4개 소스에서 데이터를 수집하는 크롤러를 작성하라.

[소스 1] 농촌진흥청 농사로 (nongsaro.go.kr)
  - 수집 내용: 지역별 벼 모내기 적정시기 표, 지역별 육묘일수
  - 저장: data/processed/agri_calendar.csv

[소스 2] 기상청 농업기상 정보
  - URL: weather.go.kr/w/agri
  - 수집 내용: 지역별 농업기상 특보, 파종/이앙 적기 예보
  - 저장: data/processed/agri_weather.csv

[소스 3] 공공데이터포털 지방보조금 (data.go.kr)
  - API 엔드포인트: https://www.data.go.kr/dataset/15013191/openapi.do
  - API 키: 환경변수 DATA_GO_KR_API_KEY 에서 읽기
  - API 키 없으면: "API_KEY_NEEDED"를 logs/api_keys_needed.txt에 기록하고 크롤링 방식으로 대체
  - 수집 내용: 농업용 자재 보조사업 공고, 계약 종료일
  - 필터: 상토, 육묘, 농자재 관련 사업
  - 저장: data/processed/subsidy_schedule.csv

[소스 4] 학술논문 (RISS, KISS)
  - RISS: https://www.riss.kr/search/Search.do?searchGubun=simple&query=수도용상토
  - KISS: https://kiss.kstudy.com
  - 수집 내용: 논문제목, 저자, 연도, 초록, 키워드
  - 저장: data/processed/academic_papers.csv
  - 주의: Selenium 필요 (JavaScript 렌더링), RISS API 승인이 늦어지는 경우, 초기 단계에서는 **Selenium**을 활용한 웹 스크래핑으로 제목과 초록 데이터를 먼저 수집하여 분석 로직을 검증
```

---

## PHASE 3 — 텍스트 마이닝 파이프라인

### 3-1. 전처리 모듈

**파일 경로:** `C:\ai_workspace\sh-ai-model\text_mining\preprocessor.py`

```
다음 기능을 포함한 전처리 클래스 TextPreprocessor를 작성하라.

[기능 1] 텍스트 정규화
  - HTML 태그 제거
  - 특수문자 제거 (단, 한글/영문/숫자/공백 유지)
  - 연속 공백 단일화
  - 종결어미 통일

[기능 2] 형태소 분석 (이중 엔진)
  - 1순위: KiwiPiepy (속도 빠름)
  - 2순위: KoNLPy Okt (KiwiPiepy 실패 시 자동 전환)
  - 추출 품사: 명사(NNG, NNP), 동사어근, 형용사어근
  
[기능 3] 불용어 처리
  - 불용어 사전 파일 자동 생성: data/processed/stopwords.txt
  - 기본 불용어 포함: 것, 수, 있다, 하다, 되다, 등, 및, 또, 의, 에서...
  - 농업 도메인 불용어 추가: 기자, 뉴스, 보도, 취재, 기사...
  - 사용자가 stopwords.txt 수정으로 커스터마이징 가능하게

[기능 4] 지역명 추출
  - 17개 시도 + 228개 시군구 전체 목록을 코드에 하드코딩
  - 텍스트에서 지역명 자동 태깅
  - 결과: [(지역명, 등장횟수, 문서ID)] 리스트 반환

[기능 5] 날짜 추출
  - 패턴: YYYY년 MM월, YYYY.MM.DD, YYYY/MM/DD 등
  - 모내기 관련 날짜 우선 태깅

[배치 처리]
  - data/master_db/news_corpus.db 전체 자동 처리
  - 처리 결과: data/processed/news_tokenized.parquet
  - 진행률 tqdm으로 표시
```

---

### 3-2. 핵심어 분석 모듈

**파일 경로:** `C:\ai_workspace\sh-ai-model\text_mining\keyword_analyzer.py`

```
다음 5가지 분석 기능을 구현하라.

[분석 1] TF-IDF 키워드 추출
  - 연도별, 월별, 지역별 상위 30개 키워드
  - 결과 저장: data/processed/tfidf_keywords.csv

[분석 2] 동시출현 네트워크 (NetworkX)
  - 윈도우 크기: 동일 문장 내 함께 등장하는 단어 쌍
  - 최소 동시출현 빈도: 5회 이상만 포함
  - 중심성 지표: betweenness, degree, eigenvector 계산
  - 결과 저장: data/processed/cooccurrence_network.graphml

[분석 3] 시계열 키워드 트렌드
  - 월별 키워드 빈도 변화
  - "상토", "모내기", "이앙", "부족", "민원" 키워드 집중 추적
  - 기후 데이터와 날짜 기준 JOIN 가능한 형태로 저장
  - 결과 저장: data/processed/keyword_trends.csv

[분석 4] 감성 분석 (긍/부정/중립)
  - 부정 키워드 사전: 부족, 품귀, 불량, 민원, 피해, 지연, 문제
  - 긍정 키워드 사전: 풍작, 공급확대, 품질향상, 적기, 원활
  - 기사별 감성 점수 계산 (-1 ~ +1)
  - 결과 저장: data/processed/sentiment_scores.csv

[분석 5] 지역별 이슈 분류
  - 각 지역에서 가장 많이 등장하는 이슈 TOP5 추출
  - 출력 형식: {지역명: [(이슈키워드, 빈도, 대표기사제목), ...]}
  - 결과 저장: data/processed/regional_issues.json
```

---

### 3-3. BERTopic 토픽 모델링

**파일 경로:** `C:\ai_workspace\sh-ai-model\text_mining\topic_modeler.py`

```
BERTopic으로 뉴스 기사의 주제를 자동 분류하라.

[모델 설정]
  - 임베딩 모델: "jhgan/ko-sroberta-multitask" (한국어 특화)
  - 임베딩 모델 로드 실패 시: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 사용
  - 최소 토픽 크기: 10개 문서
  - 예상 토픽 수: 자동 감지 (nr_topics="auto")

[기대 토픽 카테고리 - 모델이 자동 발견하도록 유도]
  예시 토픽들 (실제는 모델이 발견):
  - "모내기 시기 / 이앙 일정"
  - "상토 품질 불량 / 민원"
  - "상토 수급 부족 / 품귀"
  - "기상 이변 / 냉해 / 가뭄"
  - "지자체 보조사업 / 농협 공급"
  - "상토 가격 인상"
  - "신품종 / 육묘 기술"

[저장]
  - 토픽 모델: data/processed/bertopic_model/ (폴더)
  - 토픽-문서 매핑: data/processed/topic_assignments.csv
  - 토픽 키워드 요약: data/processed/topic_summary.json
  - 연도별 토픽 비중 변화: data/processed/topic_trends_yearly.csv

[시각화]
  - 토픽 분포 HTML: visualization/topic_distribution.html
  - 토픽 트렌드 HTML: visualization/topic_trends.html
  - BERTopic 내장 시각화 함수 활용
```

---

## PHASE 4 — 워드클라우드 및 시각화

### 4-1. 워드클라우드 생성

**파일 경로:** `C:\ai_workspace\sh-ai-model\visualization\wordcloud_generator.py`

```
다음 7종의 워드클라우드를 자동 생성하라.
폰트: 나눔고딕 또는 시스템에 있는 한글 폰트 자동 탐색
폰트 없으면: wget으로 NanumGothic.ttf 다운로드 시도
배경색: 흰색, 최대 단어수: 100개

[생성 목록]
1. 전체 기사 통합 워드클라우드
   → visualization/wc_total.png

2. 연도별 워드클라우드 (2015~2026, 연도별 1개)
   → visualization/wc_year_{YYYY}.png

3. 월별 워드클라우드 (1월~12월)
   → visualization/wc_month_{MM}.png

4. 지역별 워드클라우드 (전남, 전북, 충남, 경남, 경북 각 1개)
   → visualization/wc_region_{지역명}.png

5. 부정 감성 기사만 추출한 워드클라우드 (민원/불량/부족 이슈)
   → visualization/wc_negative_issues.png

6. 모내기 시기 전후 2주 기사 워드클라우드
   → visualization/wc_transplanting_season.png

7. 상위 출고월(4월, 5월) 기사 워드클라우드
   → visualization/wc_peak_months.png

모든 생성 완료 후 visualization/wc_index.html 인덱스 파일 자동 생성
```

---

### 4-2. 출고 × 기후 × 뉴스 통합 시각화

**파일 경로:** `C:\ai_workspace\sh-ai-model\visualization\integrated_dashboard.py`

```
Plotly로 다음 4개 인터랙티브 차트를 생성하고
visualization/dashboard.html 하나의 파일로 통합하라.

[차트 1] 3축 시계열 (메인 대시보드)
  X축: 날짜 (일별)
  Y축1 (좌): 일별 출고량 (막대그래프, 파란색)
  Y축2 (우): 일평균기온 (선그래프, 빨간색)
  Y축3 (우2): 뉴스 기사 수 (점선, 녹색)
  범례: 토글 가능
  제목: "출고량 × 기온 × 뉴스 빈도 통합 분석 (2019-2026)"

[차트 2] 임계기온 돌파 이벤트 분석
  임계기온 후보: 5°C, 8°C, 10°C, 13°C, 15°C
  각 임계기온 돌파 후 7일, 14일, 21일 출고량 변화율 계산
  히트맵으로 표시
  제목: "임계기온 돌파 후 출고 반응 분석"

[차트 3] 지역별 모내기 시기 × 출고 상관관계
  X축: 이앙 적정 시기 (지역별 다름)
  Y축: 해당 지역 출고량
  지역별 색상 구분
  제목: "지역별 모내기 시기와 출고량 상관관계"

[차트 4] 연도별 이슈 키워드 트렌드 히트맵
  X축: 연도 (2005~2026)
  Y축: 주요 키워드 20개
  색상: 등장 빈도 (밝을수록 많음)
  제목: "연도별 주요 키워드 빈도 변화"
```

---

## PHASE 5 — 전략 인사이트 자동 생성

### 5-1. 분석 결과 통합 및 인사이트 추출

**파일 경로:** `C:\ai_workspace\sh-ai-model\analysis\insight_extractor.py`

```
지금까지 생성된 모든 분석 결과를 읽어서
다음 항목을 자동으로 계산하고 reports/insights.json에 저장하라.

[계산 항목 1] 임계기온 확정
  - climate_data.csv에서 일평균기온 컬럼 읽기
  - shipment_data.csv에서 일별 출고량 읽기
  - 5, 8, 10, 13, 15°C 각각에 대해
    "해당 기온 돌파일 기준 이후 14일 출고량 평균 / 돌파 전 14일 출고량 평균" 비율 계산
  - 비율이 가장 높은 온도 = 핵심 임계기온으로 선정
  - 결과: {"critical_temp": 13, "ratio": 2.8, "confidence": "high"}

[계산 항목 2] 지역별 출고 피크 시기
  - 지역별, 연도별 출고 피크 월 계산
  - 전남, 전북, 충남, 경남, 경북 상위 5개 지역 집중 분석
  - 결과: {"전남": {"peak_month": 5, "peak_week": 2, "std_days": 7}}

[계산 항목 3] 뉴스-출고 선행 지표 분석
  - "모내기", "이앙", "상토" 키워드 급증 후 출고 급증까지의 평균 일수
  - Cross-correlation 계산 (lag 0~30일)
  - 결과: {"keyword": "이앙", "lead_days": 14, "correlation": 0.72}

[계산 항목 4] 민원/불량 리스크 계절 패턴
  - 부정 감성 기사 집중 시기 계산
  - 출고 급증 시기와의 시간적 관계 분석
  - 결과: {"risk_peak_month": [4, 5], "risk_keywords": ["부족", "불량", "지연"]}

[계산 항목 5] 경쟁사 동향 (뉴스 기반)
  - 뉴스에서 등장하는 상토 브랜드/회사명 빈도 계산
  - 남해화학, 풍농, 팜한농 등 상위 경쟁사 언급 빈도 추적
  - 결과: {"competitor_mentions": {"남해화학": 234, "풍농": 156}}
```

---

### 5-2. 전략 보고서 자동 생성

**파일 경로:** `C:\ai_workspace\sh-ai-model\reports\report_generator.py`

```
다음 내용을 포함하는 HTML 보고서를 자동 생성하라.
보고서 경로: reports/sangto_strategy_report_{YYYYMMDD}.html
보고서 스타일: 프로페셔널 비즈니스 리포트 스타일 (CSS 인라인)

[보고서 목차]
1. 요약 (Executive Summary)
   - 핵심 임계기온: {critical_temp}°C
   - 지역별 피크 시기 요약 테이블
   - 뉴스 선행 지표 요약

2. 출고 패턴 분석
   - 월별 출고 분포 차트 (Plotly 임베드)
   - 임계기온 분석 결과

3. 텍스트 마이닝 결과
   - 전체 워드클라우드 이미지 임베드
   - 상위 토픽 20개 테이블
   - 연도별 트렌드 히트맵

4. 지역별 전략 제안
   - 전남/전북/충남/경남/경북 각 지역 분석
   - 각 지역 모내기 시기, 출고 피크, 주요 이슈

5. 선제적 배차 전략 제안
   - 임계기온 돌파 X일 전 배차 권고 기준
   - 트럭 대수 계산 로직 (11톤=1800포, 5톤=700포)
   - 알림 메시지 템플릿 예시

6. 사업 방향성 인사이트
   - 뉴스 분석 기반 시장 기회
   - 경쟁사 동향 요약
   - 리스크 키워드 모니터링 제안

보고서 하단에 생성 일시, 분석 기간, 수집 기사 건수 자동 표시
```

---

## PHASE 6 — 자동화 스케줄러

### 6-1. 일일 자동 수집 스케줄러

**파일 경로:** `C:\ai_workspace\sh-ai-model\crawlers\scheduler.py`

```
APScheduler로 다음 자동화 작업을 설정하라.
스케줄러 실행: python crawlers/scheduler.py
백그라운드 실행: nohup python crawlers/scheduler.py > logs/scheduler.log 2>&1 &

[작업 1] 매일 오전 7시 - 뉴스 수집
  - 어제 날짜 기준 모든 키워드 그룹 크롤링
  - 신규 기사만 DB에 추가 (url 중복 체크)
  - 완료 후 logs/daily_crawl_{YYYYMMDD}.log에 기록

[작업 2] 매주 월요일 오전 9시 - 텍스트 마이닝 갱신
  - 지난 1주일 신규 기사 전처리
  - TF-IDF 및 키워드 트렌드 업데이트
  - 워드클라우드 재생성

[작업 3] 매월 1일 오전 6시 - 월간 보고서 생성
  - 전월 데이터 통합 분석
  - 월간 전략 보고서 HTML 자동 생성
  - reports/monthly/ 폴더에 저장

[작업 4] 4~5월 집중 수집 모드 (자동 감지)
  - 현재 월이 3, 4, 5월이면 수집 주기를 1일 1회 → 1일 3회로 자동 변경
  - 오전 6시, 낮 12시, 오후 9시 3회 수집
```

---

## PHASE 7 — 전체 파이프라인 실행 및 검증

### 7-1. 마스터 실행 스크립트

**파일 경로:** `C:\ai_workspace\sh-ai-model\run_all.py`

```
전체 파이프라인을 순서대로 실행하는 마스터 스크립트를 작성하라.
각 단계 실행 전 의존 파일 존재 여부 체크
실패 시 에러 내용을 logs/pipeline_errors.log에 기록하고 계속 진행

실행 순서:
STEP 1: 패키지 임포트 검증
STEP 2: 디렉토리 구조 검증
STEP 3: 네이버 뉴스 크롤링 (crawlers/naver_news_crawler.py)
STEP 4: 농업 데이터 수집 (crawlers/agri_data_crawler.py)
STEP 5: 텍스트 전처리 (text_mining/preprocessor.py)
STEP 6: 키워드 분석 (text_mining/keyword_analyzer.py)
STEP 7: BERTopic 모델링 (text_mining/topic_modeler.py)
STEP 8: 워드클라우드 생성 (visualization/wordcloud_generator.py)
STEP 9: 통합 대시보드 생성 (visualization/integrated_dashboard.py)
STEP 10: 인사이트 추출 (analysis/insight_extractor.py)
STEP 11: 보고서 생성 (reports/report_generator.py)
STEP 12: 스케줄러 시작 (crawlers/scheduler.py 백그라운드)

각 STEP 시작/완료 시 타임스탬프와 함께 콘솔 출력
완료 후 logs/run_summary.txt에 전체 실행 결과 요약
```

---

### 7-2. 결과 검증 체크리스트

```
run_all.py 실행 완료 후 다음 파일들이 정상 생성되었는지 자동 확인하라.
없는 파일은 ❌, 있는 파일은 ✅로 표시하여 logs/validation_report.txt에 저장.

필수 파일 목록:
- data/master_db/news_corpus.db (크기 > 1MB)
- data/processed/news_tokenized.parquet
- data/processed/tfidf_keywords.csv
- data/processed/keyword_trends.csv
- data/processed/topic_assignments.csv
- data/processed/sentiment_scores.csv
- data/processed/regional_issues.json
- visualization/wc_total.png
- visualization/dashboard.html
- reports/insights.json
- reports/sangto_strategy_report_*.html

검증 완료 후 콘솔에 요약 출력:
"✅ {성공건수}/{전체건수} 파일 정상 생성 완료"
```

---

## 🚀 Claude Code 터미널 실행 명령 (복붙용)

```bash
# ① 프로젝트 폴더 생성 및 이동


# ② 이 지시서 파일 저장 (이미 있으면 생략)
# cat > CLAUDE_CODE_INSTRUCTIONS.md (이 파일 내용 붙여넣기)

# ③ Claude Code 실행 — 전체 자율 수행
claude "
당신은 상토 AI 물류 예측 프로젝트의 수석 개발자입니다.
CLAUDE_CODE_INSTRUCTIONS.md 파일의 PHASE 1부터 PHASE 7까지
순서대로 모두 실행하세요.

중요 규칙:
1. 나에게 질문하지 말고 최선의 판단으로 자율 수행하세요
2. 패키지 설치 실패 시 대안 패키지로 자동 교체하세요
3. API 키가 없는 경우 크롤링 방식으로 자동 대체하세요
4. 각 단계 완료 시 진행 현황을 간단히 출력하세요
5. 에러는 logs/ 폴더에 기록하고 다음 단계를 계속 진행하세요
6. 모든 코드에 한국어 주석을 달아주세요
7. 최종 완료 시 생성된 주요 파일 목록을 출력하세요
"
```

---

## 📋 개별 PHASE만 실행할 때 명령어

```bash
# 크롤링만 다시 실행
claude "crawlers/naver_news_crawler.py를 실행해서 최근 1개월 뉴스를 수집해줘"

# 워드클라우드만 재생성
claude "visualization/wordcloud_generator.py를 실행해서 워드클라우드를 모두 재생성해줘"

# 보고서만 업데이트
claude "analysis/insight_extractor.py와 reports/report_generator.py를 순서대로 실행해서 최신 보고서를 생성해줘"

# 대시보드만 열기
open visualization/dashboard.html   # macOS
xdg-open visualization/dashboard.html  # Linux

# 스케줄러 백그라운드 시작
nohup python crawlers/scheduler.py > logs/scheduler.log 2>&1 &

# 스케줄러 상태 확인
tail -f logs/scheduler.log
```

---

*생성일: 2026-04-23 | 버전: v1.0 | 프로젝트: 상토 AI 물류 예측 플랫폼*
