"""
토픽 모델링 모듈
BERTopic을 활용하여 한국어 농업 뉴스 데이터의 토픽 분석 수행
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 경고 메시지 억제
warnings.filterwarnings("ignore")

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(r"C:\ai_workspace\sh-ai-model")

# 디렉토리 경로 상수 정의
DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
DIR_VISUALIZATION = PROJECT_ROOT / "visualization"

# 출력 파일 경로 상수 정의
PATH_INPUT_PARQUET = DIR_PROCESSED / "news_tokenized.parquet"
PATH_BERTOPIC_MODEL = DIR_PROCESSED / "bertopic_model"
PATH_TOPIC_ASSIGNMENTS = DIR_PROCESSED / "topic_assignments.csv"
PATH_TOPIC_SUMMARY = DIR_PROCESSED / "topic_summary.json"
PATH_TOPIC_TRENDS_YEARLY = DIR_PROCESSED / "topic_trends_yearly.csv"
PATH_VIZ_DISTRIBUTION = DIR_VISUALIZATION / "topic_distribution.html"
PATH_VIZ_TRENDS = DIR_VISUALIZATION / "topic_trends.html"

# 모델 설정 상수
EMBEDDING_MODEL_PRIMARY = "jhgan/ko-sroberta-multitask"
EMBEDDING_MODEL_FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_TOPIC_SIZE = 10
NR_TOPICS = "auto"
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# 문서 수 최소 임계값 (이하이면 모델링 생략)
MIN_DOCS_FOR_MODELING = 50

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 샘플 한국어 농업 텍스트 (입력 파일이 없을 때 동작 확인용)
SAMPLE_AGRICULTURE_TEXTS = [
    "올해 쌀 생산량이 예년보다 감소할 것으로 전망되며 벼 재배 농가의 수익이 줄어들 가능성이 높다",
    "폭염과 가뭄으로 인해 채소 가격이 급등하면서 배추와 무 등 김치 재료 수급에 어려움이 생겼다",
    "정부는 농업 보조금 정책을 개편하여 친환경 농법을 실천하는 농가에 추가 지원을 제공하기로 했다",
    "스마트팜 기술 도입으로 딸기와 토마토 재배 효율이 크게 향상되어 수출 경쟁력이 강화되고 있다",
    "기후 변화로 인한 병해충 발생이 증가하면서 농약 사용량이 늘어나고 있어 안전성 문제가 대두되고 있다",
    "농촌 고령화가 심화되면서 청년 농업인 육성 프로그램에 대한 관심과 지원이 확대되고 있다",
    "한우 가격 하락으로 축산 농가들이 경영난을 겪고 있으며 정부의 긴급 지원책 마련이 시급하다",
    "수입 농산물 증가로 국내 과일 농가들이 어려움을 호소하고 있으며 사과와 배 가격이 크게 하락했다",
    "농업 용수 부족 문제를 해결하기 위해 저수지 확충과 절수 관개 기술 보급이 추진되고 있다",
    "친환경 농업 인증을 받은 농산물에 대한 소비자 수요가 증가하면서 유기농 시장이 빠르게 성장하고 있다",
]


class TopicModeler:
    """
    BERTopic 기반 한국어 농업 뉴스 토픽 모델링 클래스

    주요 기능:
    - 한국어 특화 임베딩 모델로 문서 벡터화
    - UMAP 차원 축소 + HDBSCAN 클러스터링으로 토픽 추출
    - 토픽 모델 및 분석 결과 저장
    - 토픽 분포 및 연도별 트렌드 시각화
    """

    def __init__(self):
        """초기화: 디렉토리 생성 및 상태 플래그 설정"""
        # 필요한 출력 디렉토리 생성
        DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
        DIR_VISUALIZATION.mkdir(parents=True, exist_ok=True)
        PATH_BERTOPIC_MODEL.mkdir(parents=True, exist_ok=True)

        # 모델 객체
        self.topic_model = None
        self.embedding_model = None
        self.embedding_model_name = None

        # 데이터
        self.documents = []
        self.metadata_df = None

        # 토픽 모델링 결과
        self.topics = []
        self.probs = []
        self.topic_info = None

        # 문서 수 부족 시 설정되는 플래그
        self.SKIP_MODELING = False

        logger.info("TopicModeler 초기화 완료")

    # ------------------------------------------------------------------
    # 데이터 로드
    # ------------------------------------------------------------------

    def load_data(self) -> bool:
        """
        토큰화된 뉴스 parquet 파일 로드.
        파일이 없으면 샘플 텍스트로 대체하고, 문서 수가 부족하면 SKIP_MODELING 플래그 설정.

        Returns:
            bool: 로드 성공 여부
        """
        if PATH_INPUT_PARQUET.exists():
            logger.info(f"입력 파일 로드: {PATH_INPUT_PARQUET}")
            df = pd.read_parquet(PATH_INPUT_PARQUET)

            # 텍스트 컬럼 탐색 (우선순위 순)
            text_col = self._find_text_column(df)
            if text_col is None:
                logger.warning("텍스트 컬럼을 찾을 수 없습니다. 샘플 데이터로 대체합니다.")
                self._use_sample_data()
                return True

            # 결측값 제거 및 문자열 변환
            self.documents = (
                df[text_col].dropna().astype(str).str.strip()
            )
            self.documents = self.documents[self.documents != ""].tolist()
            self.metadata_df = df.loc[df[text_col].notna()].copy()

            logger.info(f"로드된 문서 수: {len(self.documents)}")
        else:
            logger.warning(
                f"입력 파일 없음: {PATH_INPUT_PARQUET}\n"
                "샘플 한국어 농업 텍스트로 동작 확인을 진행합니다."
            )
            self._use_sample_data()

        # 문서 수 임계값 확인
        if len(self.documents) < MIN_DOCS_FOR_MODELING:
            logger.warning(
                f"문서 수({len(self.documents)})가 최소 임계값({MIN_DOCS_FOR_MODELING})보다 "
                "적습니다. SKIP_MODELING 플래그를 설정합니다."
            )
            self.SKIP_MODELING = True

        return True

    def _find_text_column(self, df: pd.DataFrame) -> str | None:
        """데이터프레임에서 텍스트 컬럼명 탐색"""
        # 가능한 컬럼명 후보 (우선순위 순)
        candidates = [
            "tokenized_text", "tokens", "text", "content",
            "article", "body", "cleaned_text", "processed_text",
        ]
        for col in candidates:
            if col in df.columns:
                return col
        # 문자열 타입 컬럼 중 첫 번째 선택
        str_cols = df.select_dtypes(include="object").columns.tolist()
        return str_cols[0] if str_cols else None

    def _use_sample_data(self):
        """샘플 한국어 농업 텍스트를 문서 목록으로 설정"""
        self.documents = SAMPLE_AGRICULTURE_TEXTS.copy()
        # 샘플 메타데이터 생성 (연도 포함)
        self.metadata_df = pd.DataFrame({
            "text": self.documents,
            "year": [2022, 2022, 2023, 2023, 2024, 2024, 2025, 2025, 2026, 2026],
        })
        logger.info(f"샘플 데이터 {len(self.documents)}건 설정 완료")

    # ------------------------------------------------------------------
    # 임베딩 모델 로드
    # ------------------------------------------------------------------

    def _load_embedding_model(self):
        """
        한국어 임베딩 모델 로드.
        1순위: jhgan/ko-sroberta-multitask
        2순위: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        """
        from sentence_transformers import SentenceTransformer

        # 1순위 모델 시도
        try:
            logger.info(f"임베딩 모델 로드 시도 (1순위): {EMBEDDING_MODEL_PRIMARY}")
            model = SentenceTransformer(EMBEDDING_MODEL_PRIMARY)
            self.embedding_model = model
            self.embedding_model_name = EMBEDDING_MODEL_PRIMARY
            logger.info("1순위 임베딩 모델 로드 성공")
            return
        except Exception as e:
            logger.warning(f"1순위 모델 로드 실패: {e}")

        # 2순위 폴백 모델 시도
        try:
            logger.info(f"임베딩 모델 로드 시도 (2순위): {EMBEDDING_MODEL_FALLBACK}")
            model = SentenceTransformer(EMBEDDING_MODEL_FALLBACK)
            self.embedding_model = model
            self.embedding_model_name = EMBEDDING_MODEL_FALLBACK
            logger.info("2순위 임베딩 모델 로드 성공 (폴백)")
        except Exception as e:
            logger.error(f"2순위 모델도 로드 실패: {e}")
            raise RuntimeError(
                f"임베딩 모델 로드에 모두 실패했습니다.\n"
                f"  1순위: {EMBEDDING_MODEL_PRIMARY}\n"
                f"  2순위: {EMBEDDING_MODEL_FALLBACK}"
            )

    # ------------------------------------------------------------------
    # BERTopic 모델 빌드 및 학습
    # ------------------------------------------------------------------

    def build_model(self):
        """
        UMAP + HDBSCAN + BERTopic 모델 구성.
        SKIP_MODELING 플래그가 True이면 모델 빌드를 건너뜀.
        """
        if self.SKIP_MODELING:
            logger.info("SKIP_MODELING 플래그 설정됨 — 모델 빌드를 건너뜁니다.")
            return

        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP

        # 임베딩 모델 로드
        self._load_embedding_model()

        # UMAP 차원 축소 모델 설정
        umap_model = UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            n_components=UMAP_N_COMPONENTS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=42,
        )
        logger.info(
            f"UMAP 설정: n_neighbors={UMAP_N_NEIGHBORS}, n_components={UMAP_N_COMPONENTS}, "
            f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}"
        )

        # HDBSCAN 클러스터링 모델 설정
        hdbscan_model = HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            metric=HDBSCAN_METRIC,
            cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
            prediction_data=True,
        )
        logger.info(
            f"HDBSCAN 설정: min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, "
            f"metric={HDBSCAN_METRIC}, cluster_selection_method={HDBSCAN_CLUSTER_SELECTION_METHOD}"
        )

        # BERTopic 모델 구성
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            min_topic_size=MIN_TOPIC_SIZE,
            nr_topics=NR_TOPICS,
            language="multilingual",
            calculate_probabilities=True,
            verbose=True,
        )
        logger.info(
            f"BERTopic 모델 구성 완료: min_topic_size={MIN_TOPIC_SIZE}, nr_topics={NR_TOPICS}"
        )

    def fit_transform(self):
        """
        문서에 대해 BERTopic 학습 및 토픽 할당 수행.
        SKIP_MODELING이 True이면 더미 결과를 설정하고 종료.
        """
        if self.SKIP_MODELING:
            logger.info("SKIP_MODELING — 더미 결과를 설정합니다.")
            self.topics = [-1] * len(self.documents)
            self.probs = [0.0] * len(self.documents)
            return

        logger.info(f"BERTopic 학습 시작: 문서 수 = {len(self.documents)}")
        self.topics, self.probs = self.topic_model.fit_transform(self.documents)

        # 토픽 정보 저장
        self.topic_info = self.topic_model.get_topic_info()
        num_topics = len(self.topic_info[self.topic_info["Topic"] != -1])
        logger.info(f"토픽 추출 완료: {num_topics}개 토픽 발견 (노이즈 제외)")

    # ------------------------------------------------------------------
    # 결과 저장
    # ------------------------------------------------------------------

    def save_model(self):
        """BERTopic 모델을 지정된 폴더에 저장"""
        if self.SKIP_MODELING or self.topic_model is None:
            logger.info("SKIP_MODELING — 모델 저장을 건너뜁니다.")
            # 빈 모델 디렉토리에 메타 파일만 저장
            meta = {"status": "SKIP_MODELING", "reason": f"문서 수 < {MIN_DOCS_FOR_MODELING}"}
            with open(PATH_BERTOPIC_MODEL / "model_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            return

        logger.info(f"BERTopic 모델 저장: {PATH_BERTOPIC_MODEL}")
        self.topic_model.save(str(PATH_BERTOPIC_MODEL), serialization="safetensors", save_ctfidf=True)
        logger.info("모델 저장 완료")

    def save_topic_assignments(self):
        """토픽-문서 매핑 결과를 CSV로 저장"""
        logger.info(f"토픽-문서 매핑 저장: {PATH_TOPIC_ASSIGNMENTS}")

        # 기본 데이터프레임 구성
        result_df = pd.DataFrame({
            "document": self.documents,
            "topic_id": self.topics,
            "probability": self.probs if isinstance(self.probs[0], float) else [
                float(p) if not hasattr(p, "__len__") else float(max(p)) for p in self.probs
            ],
        })

        # 메타데이터 컬럼 병합 (연도 등)
        if self.metadata_df is not None and "year" in self.metadata_df.columns:
            # 인덱스 리셋 후 병합
            meta_reset = self.metadata_df.reset_index(drop=True)
            if len(meta_reset) == len(result_df):
                result_df["year"] = meta_reset["year"].values

        # 토픽 라벨 추가 (SKIP_MODELING이 아닌 경우)
        if not self.SKIP_MODELING and self.topic_info is not None:
            topic_label_map = dict(
                zip(self.topic_info["Topic"], self.topic_info.get("Name", self.topic_info["Topic"]))
            )
            result_df["topic_label"] = result_df["topic_id"].map(topic_label_map).fillna("Noise")

        result_df.to_csv(PATH_TOPIC_ASSIGNMENTS, index=False, encoding="utf-8-sig")
        logger.info(f"토픽-문서 매핑 저장 완료: {len(result_df)}건")

    def save_topic_summary(self):
        """토픽별 키워드 요약을 JSON으로 저장"""
        logger.info(f"토픽 키워드 요약 저장: {PATH_TOPIC_SUMMARY}")

        if self.SKIP_MODELING or self.topic_model is None:
            # SKIP_MODELING 시 빈 요약 저장
            summary = {
                "status": "SKIP_MODELING",
                "reason": f"문서 수({len(self.documents)})가 최소 임계값({MIN_DOCS_FOR_MODELING})보다 적음",
                "topics": [],
            }
        else:
            # 각 토픽의 키워드 및 문서 수 정리
            topics_summary = []
            for _, row in self.topic_info.iterrows():
                topic_id = int(row["Topic"])
                if topic_id == -1:
                    # 노이즈 토픽 제외
                    continue
                topic_words = self.topic_model.get_topic(topic_id)
                keywords = [word for word, _ in topic_words[:10]] if topic_words else []
                keyword_scores = {word: round(float(score), 4) for word, score in topic_words[:10]} if topic_words else {}

                topics_summary.append({
                    "topic_id": topic_id,
                    "topic_name": str(row.get("Name", f"Topic_{topic_id}")),
                    "document_count": int(row.get("Count", 0)),
                    "keywords": keywords,
                    "keyword_scores": keyword_scores,
                })

            summary = {
                "status": "SUCCESS",
                "total_topics": len(topics_summary),
                "embedding_model": self.embedding_model_name,
                "total_documents": len(self.documents),
                "topics": topics_summary,
            }

        with open(PATH_TOPIC_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("토픽 키워드 요약 저장 완료")

    def save_topic_trends_yearly(self):
        """연도별 토픽 비중 변화를 CSV로 저장"""
        logger.info(f"연도별 토픽 트렌드 저장: {PATH_TOPIC_TRENDS_YEARLY}")

        # 토픽-문서 매핑 파일 로드 (이미 저장된 것 재사용)
        if not PATH_TOPIC_ASSIGNMENTS.exists():
            logger.warning("토픽-문서 매핑 파일이 없어 트렌드 계산을 건너뜁니다.")
            pd.DataFrame().to_csv(PATH_TOPIC_TRENDS_YEARLY, index=False, encoding="utf-8-sig")
            return

        assign_df = pd.read_csv(PATH_TOPIC_ASSIGNMENTS, encoding="utf-8-sig")

        if "year" not in assign_df.columns or self.SKIP_MODELING:
            # 연도 정보 없거나 SKIP_MODELING이면 빈 파일 저장
            logger.info("연도 정보 없음 또는 SKIP_MODELING — 빈 트렌드 파일 저장")
            pd.DataFrame(columns=["year", "topic_id", "document_count", "proportion"]).to_csv(
                PATH_TOPIC_TRENDS_YEARLY, index=False, encoding="utf-8-sig"
            )
            return

        # 노이즈 토픽(-1) 제외 후 연도별 토픽 비중 계산
        valid_df = assign_df[assign_df["topic_id"] != -1].copy()
        valid_df["year"] = valid_df["year"].astype(str)

        # 연도별 토픽 문서 수 집계
        year_topic_counts = (
            valid_df.groupby(["year", "topic_id"]).size().reset_index(name="document_count")
        )

        # 연도별 전체 문서 수 대비 비중 계산
        year_totals = valid_df.groupby("year").size().reset_index(name="year_total")
        year_topic_counts = year_topic_counts.merge(year_totals, on="year")
        year_topic_counts["proportion"] = (
            year_topic_counts["document_count"] / year_topic_counts["year_total"]
        ).round(4)
        year_topic_counts = year_topic_counts.drop(columns=["year_total"])

        # 토픽 라벨 병합 (있는 경우)
        if "topic_label" in assign_df.columns:
            label_map = (
                assign_df[assign_df["topic_id"] != -1]
                .drop_duplicates("topic_id")[["topic_id", "topic_label"]]
            )
            year_topic_counts = year_topic_counts.merge(label_map, on="topic_id", how="left")

        year_topic_counts = year_topic_counts.sort_values(["year", "topic_id"])
        year_topic_counts.to_csv(PATH_TOPIC_TRENDS_YEARLY, index=False, encoding="utf-8-sig")
        logger.info(f"연도별 토픽 트렌드 저장 완료: {len(year_topic_counts)}행")

    # ------------------------------------------------------------------
    # 시각화
    # ------------------------------------------------------------------

    def visualize_topic_distribution(self):
        """
        토픽 분포 시각화 HTML 생성.
        SKIP_MODELING이면 안내 메시지가 담긴 더미 HTML 생성.
        """
        logger.info(f"토픽 분포 시각화 생성: {PATH_VIZ_DISTRIBUTION}")

        if self.SKIP_MODELING or self.topic_model is None:
            html = _make_placeholder_html(
                title="토픽 분포",
                message=f"문서 수({len(self.documents)})가 최소 임계값({MIN_DOCS_FOR_MODELING})보다 "
                        "적어 토픽 모델링이 수행되지 않았습니다.",
            )
            PATH_VIZ_DISTRIBUTION.write_text(html, encoding="utf-8")
            return

        try:
            fig = self.topic_model.visualize_topics()
            fig.write_html(str(PATH_VIZ_DISTRIBUTION))
            logger.info("토픽 분포 시각화 HTML 저장 완료")
        except Exception as e:
            logger.warning(f"토픽 분포 시각화 실패: {e} — 더미 HTML 저장")
            html = _make_placeholder_html(
                title="토픽 분포",
                message=f"시각화 생성 중 오류 발생: {e}",
            )
            PATH_VIZ_DISTRIBUTION.write_text(html, encoding="utf-8")

    def visualize_topic_trends(self):
        """
        연도별 토픽 트렌드 시각화 HTML 생성.
        SKIP_MODELING이면 안내 메시지가 담긴 더미 HTML 생성.
        """
        logger.info(f"토픽 트렌드 시각화 생성: {PATH_VIZ_TRENDS}")

        if self.SKIP_MODELING or self.topic_model is None:
            html = _make_placeholder_html(
                title="토픽 트렌드",
                message=f"문서 수({len(self.documents)})가 최소 임계값({MIN_DOCS_FOR_MODELING})보다 "
                        "적어 토픽 모델링이 수행되지 않았습니다.",
            )
            PATH_VIZ_TRENDS.write_text(html, encoding="utf-8")
            return

        # 연도 정보 추출
        timestamps = None
        if self.metadata_df is not None and "year" in self.metadata_df.columns:
            meta_reset = self.metadata_df.reset_index(drop=True)
            if len(meta_reset) == len(self.documents):
                timestamps = meta_reset["year"].astype(str).tolist()

        try:
            if timestamps is not None:
                # 연도별 토픽 빈도 계산
                topics_over_time = self.topic_model.topics_over_time(
                    self.documents,
                    timestamps,
                    global_tuning=True,
                    evolution_tuning=True,
                )
                fig = self.topic_model.visualize_topics_over_time(topics_over_time)
            else:
                # 연도 정보 없으면 토픽 계층 구조로 대체 시각화
                logger.info("연도 정보 없음 — 토픽 계층 구조 시각화로 대체")
                fig = self.topic_model.visualize_hierarchy()

            fig.write_html(str(PATH_VIZ_TRENDS))
            logger.info("토픽 트렌드 시각화 HTML 저장 완료")
        except Exception as e:
            logger.warning(f"토픽 트렌드 시각화 실패: {e} — 더미 HTML 저장")
            html = _make_placeholder_html(
                title="토픽 트렌드",
                message=f"시각화 생성 중 오류 발생: {e}",
            )
            PATH_VIZ_TRENDS.write_text(html, encoding="utf-8")

    # ------------------------------------------------------------------
    # 전체 파이프라인 실행
    # ------------------------------------------------------------------

    def run(self):
        """
        전체 토픽 모델링 파이프라인 실행.

        실행 순서:
        1. 데이터 로드
        2. 모델 빌드
        3. 학습 및 토픽 할당
        4. 모델 저장
        5. 토픽-문서 매핑 저장
        6. 토픽 키워드 요약 저장
        7. 연도별 트렌드 저장
        8. 토픽 분포 시각화
        9. 토픽 트렌드 시각화
        """
        logger.info("=" * 60)
        logger.info("토픽 모델링 파이프라인 시작")
        logger.info("=" * 60)

        # 1. 데이터 로드
        self.load_data()

        # 2. 모델 빌드 (SKIP_MODELING이면 내부에서 skip)
        self.build_model()

        # 3. 학습 및 토픽 할당
        self.fit_transform()

        # 4. 모델 저장
        self.save_model()

        # 5. 토픽-문서 매핑 저장
        self.save_topic_assignments()

        # 6. 토픽 키워드 요약 저장
        self.save_topic_summary()

        # 7. 연도별 트렌드 저장
        self.save_topic_trends_yearly()

        # 8. 토픽 분포 시각화
        self.visualize_topic_distribution()

        # 9. 토픽 트렌드 시각화
        self.visualize_topic_trends()

        logger.info("=" * 60)
        if self.SKIP_MODELING:
            logger.info("토픽 모델링 파이프라인 완료 (SKIP_MODELING — 더미 결과 저장)")
        else:
            logger.info("토픽 모델링 파이프라인 완료")
        logger.info("=" * 60)

        return {
            "skip_modeling": self.SKIP_MODELING,
            "num_documents": len(self.documents),
            "embedding_model": self.embedding_model_name,
            "output_files": {
                "bertopic_model": str(PATH_BERTOPIC_MODEL),
                "topic_assignments": str(PATH_TOPIC_ASSIGNMENTS),
                "topic_summary": str(PATH_TOPIC_SUMMARY),
                "topic_trends_yearly": str(PATH_TOPIC_TRENDS_YEARLY),
                "viz_distribution": str(PATH_VIZ_DISTRIBUTION),
                "viz_trends": str(PATH_VIZ_TRENDS),
            },
        }


# ------------------------------------------------------------------
# 유틸리티 함수
# ------------------------------------------------------------------

def _make_placeholder_html(title: str, message: str) -> str:
    """
    토픽 모델링 미수행 시 안내 메시지를 담은 플레이스홀더 HTML 생성

    Args:
        title: HTML 페이지 제목
        message: 사용자에게 표시할 안내 메시지

    Returns:
        str: HTML 문자열
    """
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
      background: #f8f9fa;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      padding: 40px 60px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.1);
      text-align: center;
      max-width: 600px;
    }}
    h2 {{ color: #495057; margin-bottom: 16px; }}
    p  {{ color: #868e96; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>{title} — 데이터 없음</h2>
    <p>{message}</p>
  </div>
</body>
</html>
"""


# ------------------------------------------------------------------
# 모듈 직접 실행 시 파이프라인 구동
# ------------------------------------------------------------------

if __name__ == "__main__":
    modeler = TopicModeler()
    result = modeler.run()
    print("\n[실행 결과 요약]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
