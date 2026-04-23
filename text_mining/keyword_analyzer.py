"""
keyword_analyzer.py
뉴스 텍스트 데이터에 대한 5가지 키워드 분석 기능을 제공하는 모듈.
입력: data/processed/news_tokenized.parquet
      (파일이 없으면 샘플 데이터를 자동 생성하여 동작)
"""

import json
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
from itertools import combinations

warnings.filterwarnings("ignore")

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(r"C:\ai_workspace\sh-ai-model")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ──────────────────────────────────────────────
# 샘플 데이터 생성 함수 (입력 파케이 파일이 없을 경우 사용)
# ──────────────────────────────────────────────
def _create_sample_data() -> pd.DataFrame:
    """
    news_tokenized.parquet 파일이 없을 경우
    테스트용 샘플 뉴스 토큰화 데이터를 생성한다.
    컬럼: date, region, title, tokens (list of str)
    """
    import random
    random.seed(42)
    np.random.seed(42)

    # 농업 관련 샘플 단어 풀
    token_pool_negative = [
        "상토", "부족", "품귀", "불량", "민원", "피해", "지연", "문제", "품절", "대란",
        "공급", "차질", "수급", "어려움", "가격상승", "재고부족", "농가", "이앙",
    ]
    token_pool_positive = [
        "풍작", "공급확대", "품질향상", "적기", "원활", "충분", "호조", "모내기",
        "성공", "증가", "개선", "안정", "확보", "생산",
    ]
    token_pool_general = [
        "상토", "모내기", "이앙", "벼", "농협", "비료", "종자", "농가", "수확",
        "기상", "강수", "가뭄", "홍수", "온도", "경기", "충남", "전남", "경북",
        "강원", "경기", "전북", "충북", "제주", "인천", "세종", "울산", "대전",
        "시장", "유통", "물량", "가격", "협력", "지원", "정책", "농림부",
    ]

    regions = ["경기", "충남", "전남", "경북", "강원", "전북", "충북", "경남", "제주", "인천"]
    titles_templates = [
        "{region} 지역 {keyword} 현황 보도",
        "{region} 농가 {keyword} 관련 뉴스",
        "{region} {keyword} 문제 대응 방안",
        "{region} {keyword} 공급 현황",
        "{region} 지역 농업 {keyword} 이슈",
    ]

    rows = []
    # 2022-01 ~ 2025-12 기간의 샘플 데이터 생성
    date_range = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    for _ in range(2000):
        date = random.choice(date_range)
        region = random.choice(regions)

        # 랜덤 토큰 생성 (긍/부정 혼합)
        num_neg = random.randint(0, 5)
        num_pos = random.randint(0, 5)
        num_gen = random.randint(5, 15)
        tokens = (
            random.choices(token_pool_negative, k=num_neg)
            + random.choices(token_pool_positive, k=num_pos)
            + random.choices(token_pool_general, k=num_gen)
        )
        random.shuffle(tokens)

        keyword = random.choice(tokens) if tokens else "농업"
        title_template = random.choice(titles_templates)
        title = title_template.format(region=region, keyword=keyword)

        rows.append({
            "date": date,
            "region": region,
            "title": title,
            "tokens": tokens,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


# ──────────────────────────────────────────────
# 데이터 로드 함수
# ──────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    news_tokenized.parquet 파일을 읽어 DataFrame 으로 반환한다.
    파일이 없으면 샘플 데이터를 생성하여 반환한다.
    필수 컬럼: date, region, title, tokens
    """
    parquet_path = PROCESSED_DIR / "news_tokenized.parquet"

    if parquet_path.exists():
        print(f"[데이터 로드] {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print("[데이터 로드] news_tokenized.parquet 없음 → 샘플 데이터 생성")
        df = _create_sample_data()

    # date 컬럼 datetime 변환
    df["date"] = pd.to_datetime(df["date"])
    # year, month 컬럼 없으면 생성
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    # tokens 컬럼이 문자열인 경우 리스트로 변환
    if df["tokens"].dtype == object and isinstance(df["tokens"].iloc[0], str):
        df["tokens"] = df["tokens"].apply(lambda x: x.split() if isinstance(x, str) else x)

    print(f"[데이터 로드 완료] 총 {len(df):,}건, 기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ──────────────────────────────────────────────────────────────────
# KeywordAnalyzer 클래스
# ──────────────────────────────────────────────────────────────────
class KeywordAnalyzer:
    """
    뉴스 텍스트 데이터에 대한 5가지 키워드 분석 기능을 제공하는 클래스.

    분석1: TF-IDF 키워드 추출
    분석2: 동시출현 네트워크 (NetworkX)
    분석3: 시계열 키워드 트렌드
    분석4: 감성 분석
    분석5: 지역별 이슈 분류
    """

    # 감성 분석 사전
    NEGATIVE_WORDS = {"부족", "품귀", "불량", "민원", "피해", "지연", "문제", "품절", "대란"}
    POSITIVE_WORDS = {"풍작", "공급확대", "품질향상", "적기", "원활", "충분", "호조"}

    # 시계열 트렌드 집중 추적 키워드
    TRACKED_KEYWORDS = ["상토", "모내기", "이앙", "부족", "민원"]

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            필수 컬럼: date(datetime), region(str), title(str), tokens(list of str)
        """
        self.df = df.copy()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────
    # 분석1: TF-IDF 키워드 추출
    # ──────────────────────────────────────────────
    def analyze_tfidf(self, top_n: int = 30) -> pd.DataFrame:
        """
        연도별, 월별, 지역별로 TF-IDF 상위 키워드를 추출한다.

        Parameters
        ----------
        top_n : int
            각 그룹에서 추출할 상위 키워드 수 (기본값 30)

        Returns
        -------
        pd.DataFrame
            컬럼: group_type, group_key, rank, keyword, tfidf_score
        """
        print("[분석1] TF-IDF 키워드 추출 시작...")

        results = []

        def _extract_tfidf(group_df: pd.DataFrame, group_type: str, group_key) -> list:
            """특정 그룹 내에서 TF-IDF 상위 키워드를 추출하는 내부 함수."""
            # 토큰 리스트를 공백으로 연결하여 문서 형태로 변환
            docs = group_df["tokens"].apply(
                lambda tokens: " ".join(tokens) if isinstance(tokens, list) else str(tokens)
            ).tolist()

            if len(docs) < 2:
                # 문서가 1건 이하이면 단순 빈도 기반으로 대체
                word_counts = Counter(
                    w for doc in docs for w in doc.split()
                )
                return [
                    {
                        "group_type": group_type,
                        "group_key": str(group_key),
                        "rank": rank + 1,
                        "keyword": word,
                        "tfidf_score": round(cnt, 6),
                    }
                    for rank, (word, cnt) in enumerate(word_counts.most_common(top_n))
                ]

            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(max_features=5000, min_df=1)
            try:
                tfidf_matrix = vectorizer.fit_transform(docs)
            except ValueError:
                return []

            feature_names = vectorizer.get_feature_names_out()
            # 문서 전체 TF-IDF 평균 스코어 계산
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = mean_scores.argsort()[::-1][:top_n]

            return [
                {
                    "group_type": group_type,
                    "group_key": str(group_key),
                    "rank": rank + 1,
                    "keyword": feature_names[idx],
                    "tfidf_score": round(float(mean_scores[idx]), 6),
                }
                for rank, idx in enumerate(top_indices)
            ]

        # 연도별 TF-IDF
        print("  - 연도별 TF-IDF 계산 중...")
        for year, grp in self.df.groupby("year"):
            results.extend(_extract_tfidf(grp, "year", year))

        # 월별 TF-IDF (연도-월 조합)
        print("  - 월별 TF-IDF 계산 중...")
        self.df["year_month"] = self.df["date"].dt.to_period("M").astype(str)
        for ym, grp in self.df.groupby("year_month"):
            results.extend(_extract_tfidf(grp, "month", ym))

        # 지역별 TF-IDF
        print("  - 지역별 TF-IDF 계산 중...")
        for region, grp in self.df.groupby("region"):
            results.extend(_extract_tfidf(grp, "region", region))

        result_df = pd.DataFrame(results)
        output_path = PROCESSED_DIR / "tfidf_keywords.csv"
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  [저장] {output_path}  ({len(result_df):,}행)")
        return result_df

    # ──────────────────────────────────────────────
    # 분석2: 동시출현 네트워크
    # ──────────────────────────────────────────────
    def analyze_cooccurrence_network(self, min_cooccurrence: int = 5) -> nx.Graph:
        """
        동일 문서(기사) 내에서 함께 등장하는 단어 쌍의 동시출현 네트워크를 구성한다.

        Parameters
        ----------
        min_cooccurrence : int
            네트워크에 포함할 최소 동시출현 횟수 (기본값 5)

        Returns
        -------
        nx.Graph
            중심성(betweenness, degree, eigenvector) 속성이 포함된 네트워크 그래프
        """
        print("[분석2] 동시출현 네트워크 분석 시작...")

        # 단어 쌍별 동시출현 횟수 계산
        cooccurrence_counts: Counter = Counter()
        for tokens in self.df["tokens"]:
            if not isinstance(tokens, list) or len(tokens) < 2:
                continue
            # 중복 제거 후 정렬하여 (단어A, 단어B) 쌍 생성
            unique_tokens = sorted(set(tokens))
            for pair in combinations(unique_tokens, 2):
                cooccurrence_counts[pair] += 1

        print(f"  - 총 단어 쌍: {len(cooccurrence_counts):,}개")

        # 최소 동시출현 횟수 필터링
        filtered_pairs = {
            pair: cnt
            for pair, cnt in cooccurrence_counts.items()
            if cnt >= min_cooccurrence
        }
        print(f"  - 최소 {min_cooccurrence}회 이상 단어 쌍: {len(filtered_pairs):,}개")

        # NetworkX 그래프 생성
        G = nx.Graph()
        for (word_a, word_b), weight in filtered_pairs.items():
            G.add_edge(word_a, word_b, weight=weight)

        if G.number_of_nodes() == 0:
            print("  [경고] 필터 기준을 충족하는 노드가 없어 빈 그래프를 반환합니다.")
            output_path = PROCESSED_DIR / "cooccurrence_network.graphml"
            nx.write_graphml(G, str(output_path))
            print(f"  [저장] {output_path}")
            return G

        # 중심성 계산
        print("  - betweenness centrality 계산 중...")
        betweenness = nx.betweenness_centrality(G, weight="weight")

        print("  - degree centrality 계산 중...")
        degree = nx.degree_centrality(G)

        print("  - eigenvector centrality 계산 중...")
        try:
            eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            # 수렴 실패 시 0으로 초기화
            eigenvector = {node: 0.0 for node in G.nodes()}

        # 중심성 값을 노드 속성으로 저장
        for node in G.nodes():
            G.nodes[node]["betweenness_centrality"] = round(betweenness.get(node, 0.0), 6)
            G.nodes[node]["degree_centrality"] = round(degree.get(node, 0.0), 6)
            G.nodes[node]["eigenvector_centrality"] = round(eigenvector.get(node, 0.0), 6)

        print(f"  - 노드 수: {G.number_of_nodes():,}, 엣지 수: {G.number_of_edges():,}")

        # GraphML 형식으로 저장
        output_path = PROCESSED_DIR / "cooccurrence_network.graphml"
        nx.write_graphml(G, str(output_path))
        print(f"  [저장] {output_path}")
        return G

    # ──────────────────────────────────────────────
    # 분석3: 시계열 키워드 트렌드
    # ──────────────────────────────────────────────
    def analyze_keyword_trends(self, tracked_keywords: list = None) -> pd.DataFrame:
        """
        월별 키워드 빈도 변화를 분석하고, 집중 추적 키워드의 트렌드를 저장한다.
        기후 데이터와 날짜 기준 JOIN 가능하도록 date 컬럼을 포함한다.

        Parameters
        ----------
        tracked_keywords : list, optional
            집중 추적할 키워드 목록. None이면 클래스 기본값 사용.

        Returns
        -------
        pd.DataFrame
            컬럼: date(YYYY-MM-01 형식), keyword, count, total_articles, frequency_ratio
        """
        print("[분석3] 시계열 키워드 트렌드 분석 시작...")

        if tracked_keywords is None:
            tracked_keywords = self.TRACKED_KEYWORDS

        # 연도-월 컬럼 준비
        self.df["year_month"] = self.df["date"].dt.to_period("M")

        rows = []
        for ym, grp in self.df.groupby("year_month"):
            total_articles = len(grp)
            # 해당 월의 전체 토큰 리스트 펼치기
            all_tokens = [
                token
                for tokens in grp["tokens"]
                if isinstance(tokens, list)
                for token in tokens
            ]
            token_counts = Counter(all_tokens)

            for keyword in tracked_keywords:
                count = token_counts.get(keyword, 0)
                # 전체 기사 수 대비 빈도 비율 계산
                freq_ratio = round(count / total_articles, 6) if total_articles > 0 else 0.0
                rows.append({
                    # 기후 데이터와 JOIN 가능하도록 월의 첫날 날짜 형식 사용
                    "date": ym.to_timestamp().strftime("%Y-%m-%d"),
                    "year_month": str(ym),
                    "keyword": keyword,
                    "count": count,
                    "total_articles": total_articles,
                    "frequency_ratio": freq_ratio,
                })

        result_df = pd.DataFrame(rows)

        output_path = PROCESSED_DIR / "keyword_trends.csv"
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  [저장] {output_path}  ({len(result_df):,}행)")
        return result_df

    # ──────────────────────────────────────────────
    # 분석4: 감성 분석
    # ──────────────────────────────────────────────
    def analyze_sentiment(self) -> pd.DataFrame:
        """
        기사별 감성 점수를 계산한다.
        긍정/부정 단어 사전을 기반으로 -1 ~ +1 범위의 감성 점수를 산출한다.

        Returns
        -------
        pd.DataFrame
            원본 df에 sentiment_score, pos_count, neg_count 컬럼이 추가된 DataFrame
        """
        print("[분석4] 감성 분석 시작...")

        def _calc_sentiment(tokens: list) -> dict:
            """토큰 목록을 받아 감성 점수 관련 정보를 반환하는 내부 함수."""
            if not isinstance(tokens, list) or len(tokens) == 0:
                return {"sentiment_score": 0.0, "pos_count": 0, "neg_count": 0}

            token_set = set(tokens)
            pos_count = len(token_set & self.POSITIVE_WORDS)
            neg_count = len(token_set & self.NEGATIVE_WORDS)

            total = pos_count + neg_count
            if total == 0:
                score = 0.0
            else:
                # -1 (완전 부정) ~ +1 (완전 긍정) 정규화
                score = round((pos_count - neg_count) / total, 6)

            return {
                "sentiment_score": score,
                "pos_count": pos_count,
                "neg_count": neg_count,
            }

        sentiment_data = self.df["tokens"].apply(_calc_sentiment)
        result_df = self.df.copy()
        result_df["sentiment_score"] = sentiment_data.apply(lambda x: x["sentiment_score"])
        result_df["pos_count"] = sentiment_data.apply(lambda x: x["pos_count"])
        result_df["neg_count"] = sentiment_data.apply(lambda x: x["neg_count"])

        # 감성 레이블 부여
        result_df["sentiment_label"] = result_df["sentiment_score"].apply(
            lambda s: "긍정" if s > 0 else ("부정" if s < 0 else "중립")
        )

        # tokens 컬럼은 리스트형이므로 문자열로 변환 후 저장
        save_df = result_df.drop(columns=["tokens"], errors="ignore").copy()
        if "year_month" in save_df.columns:
            save_df = save_df.drop(columns=["year_month"])

        output_path = PROCESSED_DIR / "sentiment_scores.csv"
        save_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # 감성 분포 요약 출력
        label_counts = result_df["sentiment_label"].value_counts()
        print(f"  - 감성 분포: {dict(label_counts)}")
        print(f"  [저장] {output_path}  ({len(save_df):,}행)")
        return result_df

    # ──────────────────────────────────────────────
    # 분석5: 지역별 이슈 분류
    # ──────────────────────────────────────────────
    def analyze_regional_issues(self, top_n: int = 5) -> dict:
        """
        각 지역에서 가장 많이 등장하는 이슈 키워드 TOP N을 추출한다.
        각 이슈별로 대표 기사 제목도 함께 반환한다.

        Parameters
        ----------
        top_n : int
            지역별로 추출할 상위 이슈 키워드 수 (기본값 5)

        Returns
        -------
        dict
            {지역명: [(이슈키워드, 빈도, 대표기사제목), ...]}
        """
        print("[분석5] 지역별 이슈 분류 시작...")

        regional_issues: dict = {}

        for region, grp in self.df.groupby("region"):
            # 토큰 → 기사 매핑 구성 (각 토큰이 등장하는 기사 제목 추적)
            token_to_titles: defaultdict = defaultdict(list)
            token_counts: Counter = Counter()

            for _, row in grp.iterrows():
                tokens = row["tokens"]
                title = row["title"] if pd.notna(row.get("title", "")) else ""
                if not isinstance(tokens, list):
                    continue
                # 중복 토큰 제거하여 기사당 1회씩만 카운트
                for token in set(tokens):
                    token_counts[token] += 1
                    # 대표 기사 제목 수집 (최대 3개까지만 저장하여 메모리 절약)
                    if len(token_to_titles[token]) < 3:
                        token_to_titles[token].append(title)

            # TOP N 이슈 추출
            top_issues = token_counts.most_common(top_n)

            regional_issues[region] = [
                (
                    keyword,
                    count,
                    # 대표 기사 제목: 가장 첫 번째 기사 제목 사용
                    token_to_titles[keyword][0] if token_to_titles[keyword] else "",
                )
                for keyword, count in top_issues
            ]

        # JSON 저장 (튜플은 JSON 직렬화 불가이므로 리스트로 변환)
        json_output = {
            region: [
                {
                    "keyword": issue[0],
                    "count": issue[1],
                    "representative_title": issue[2],
                }
                for issue in issues
            ]
            for region, issues in regional_issues.items()
        }

        output_path = PROCESSED_DIR / "regional_issues.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)

        print(f"  - 분석 지역 수: {len(regional_issues)}개")
        print(f"  [저장] {output_path}")
        return regional_issues

    # ──────────────────────────────────────────────
    # 전체 분석 실행 메서드
    # ──────────────────────────────────────────────
    def run_all(self) -> dict:
        """
        5가지 분석을 모두 순서대로 실행하고 결과를 딕셔너리로 반환한다.

        Returns
        -------
        dict
            {
                "tfidf": pd.DataFrame,
                "network": nx.Graph,
                "trends": pd.DataFrame,
                "sentiment": pd.DataFrame,
                "regional_issues": dict,
            }
        """
        print("=" * 60)
        print("KeywordAnalyzer 전체 분석 시작")
        print("=" * 60)

        results = {}

        results["tfidf"] = self.analyze_tfidf()
        print()

        results["network"] = self.analyze_cooccurrence_network()
        print()

        results["trends"] = self.analyze_keyword_trends()
        print()

        results["sentiment"] = self.analyze_sentiment()
        print()

        results["regional_issues"] = self.analyze_regional_issues()
        print()

        print("=" * 60)
        print("전체 분석 완료")
        print(f"결과 저장 위치: {PROCESSED_DIR}")
        print("=" * 60)
        return results


# ──────────────────────────────────────────────
# 직접 실행 진입점
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 데이터 로드 (없으면 샘플 생성)
    df = load_data()

    # 분석 실행
    analyzer = KeywordAnalyzer(df)
    analyzer.run_all()
