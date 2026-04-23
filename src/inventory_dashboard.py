# -*- coding: utf-8 -*-
"""
진짜 가용 재고 상황판
실행: streamlit run src/inventory_dashboard.py
"""

import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.inventory")

DB_PATH = Path(os.getenv("DB_PATH", "data/inventory.db"))
KST     = ZoneInfo("Asia/Seoul")

st.set_page_config(
    page_title="성화 가용재고 상황판",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 12px; }
    .ai-report-box {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-left: 4px solid #667eea;
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 1.05rem;
    }
    .peak-badge {
        background: #ff4b4b; color: white;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: bold;
    }
    .normal-badge {
        background: #0068c9; color: white;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DB 조회 헬퍼
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def load_inventory_log(days: int = 7) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    since = (date.today() - timedelta(days=days)).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(
            "SELECT * FROM inventory_log WHERE ts >= ? ORDER BY ts",
            con, params=(since,),
        )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    return df


@st.cache_data(ttl=60)
def load_latest_stock() -> dict:
    if not DB_PATH.exists():
        return {}
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            """
            SELECT product, prev_stock, production_qty, shipment_qty, available_stock, ts
            FROM inventory_log
            WHERE id IN (
                SELECT MAX(id) FROM inventory_log GROUP BY product
            )
            """
        ).fetchall()
    return {r[0]: {"prev": r[1], "production": r[2], "shipment": r[3],
                   "available": r[4], "ts": r[5]} for r in rows}


@st.cache_data(ttl=300)
def load_ai_report(report_date: str) -> str:
    if not DB_PATH.exists():
        return ""
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT report_text FROM ai_reports WHERE report_date = ?",
            (report_date,),
        ).fetchone()
    return row[0] if row else ""


@st.cache_data(ttl=60)
def load_production_today() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    today = date.today().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(
            "SELECT * FROM production_reports WHERE reported_at >= ? ORDER BY reported_at DESC",
            con, params=(today,),
        )
    return df


def is_peak_season() -> bool:
    return datetime.now(KST).month in (3, 4)


# ══════════════════════════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════════════════════════

now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
season  = '<span class="peak-badge">🔥 피크시즌</span>' if is_peak_season() \
          else '<span class="normal-badge">🟢 평시</span>'

col_title, col_refresh = st.columns([6, 1])
with col_title:
    st.markdown(f"## 📦 성화 가용재고 상황판 &nbsp; {season}", unsafe_allow_html=True)
    st.caption(f"기준 시각: {now_str} KST")
with col_refresh:
    st.write("")
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()


# ══════════════════════════════════════════════════════════════
# 현재고 Metric 카드
# ══════════════════════════════════════════════════════════════

latest = load_latest_stock()

c1, c2, c3, c4 = st.columns(4)

def delta_str(val: int) -> str:
    return f"{val:+,}포" if val != 0 else "변동없음"

with c1:
    d = latest.get("수도용", {})
    st.metric(
        label="🌾 수도용 가용재고",
        value=f"{d.get('available', 0):,} 포",
        delta=delta_str(d.get('production', 0) - d.get('shipment', 0)),
        help="전일재고 + 생산보고 - 아마란스출고",
    )
with c2:
    d = latest.get("원예용", {})
    st.metric(
        label="🌿 원예용 가용재고",
        value=f"{d.get('available', 0):,} 포",
        delta=delta_str(d.get('production', 0) - d.get('shipment', 0)),
    )
with c3:
    d = latest.get("수도용", {})
    st.metric(label="📤 수도용 금일 출고", value=f"{d.get('shipment', 0):,} 포")
with c4:
    d = latest.get("원예용", {})
    st.metric(label="📤 원예용 금일 출고", value=f"{d.get('shipment', 0):,} 포")

if latest:
    ts_list = [v["ts"] for v in latest.values()]
    st.caption(f"⏱ 마지막 데이터 갱신: {max(ts_list)[:19].replace('T', ' ')}")

# ── 재고 오차 경고 알람 ───────────────────────────────────────
# 임계값 설정 (사이드바에서 조정 가능 — 아래 sidebar 섹션 참고)
WARN_MIN  = int(os.getenv("WARN_MIN_STOCK",  "5000"))   # 최소 가용재고 (포)
WARN_DIFF = int(os.getenv("WARN_DIFF_RATIO", "20"))     # 오차 허용 비율 (%)

alerts = []
for product, d in latest.items():
    avail = d.get("available", 0)
    prev  = d.get("prev", 0)
    prod  = d.get("production", 0)
    ship  = d.get("shipment", 0)
    system_calc = prev + prod - ship
    diff_pct = abs(avail - system_calc) / max(abs(system_calc), 1) * 100

    if avail < WARN_MIN:
        alerts.append(("error", f"🚨 [{product}] 가용재고 부족! {avail:,}포 (기준 {WARN_MIN:,}포 미만)"))
    elif avail < WARN_MIN * 2:
        alerts.append(("warning", f"⚠️ [{product}] 재고 주의 — {avail:,}포 (기준 {WARN_MIN*2:,}포 미만)"))

    if diff_pct > WARN_DIFF and abs(avail - system_calc) > 500:
        alerts.append(("warning",
            f"⚠️ [{product}] 전산·실사 오차 {diff_pct:.1f}% "
            f"(전산계산 {system_calc:,}포 vs 가용 {avail:,}포)"))

for level, msg in alerts:
    if level == "error":
        st.error(msg)
    else:
        st.warning(msg)

st.divider()


# ══════════════════════════════════════════════════════════════
# 시계열 그래프
# ══════════════════════════════════════════════════════════════

st.subheader("📊 가용재고 추이")

tab_days = st.radio("기간", ["오늘", "3일", "7일"], horizontal=True,
                    label_visibility="collapsed")
days_map = {"오늘": 1, "3일": 3, "7일": 7}
df_log   = load_inventory_log(days=days_map[tab_days])

if df_log.empty:
    st.info("데이터가 없습니다. `python src/inventory_server.py` 를 먼저 실행하세요.")
else:
    fig = go.Figure()
    for product, color in [("수도용", "#1f77b4"), ("원예용", "#2ca02c")]:
        sub = df_log[df_log["product"] == product]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["ts"], y=sub["available_stock"],
            name=f"{product} 가용재고",
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%m/%d %H:%M}<br>가용재고: %{y:,}포<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=sub["ts"], y=sub["shipment_qty"],
            name=f"{product} 출고량",
            marker_color=color, opacity=0.25,
            yaxis="y2",
            hovertemplate="%{x|%m/%d %H:%M}<br>출고: %{y:,}포<extra></extra>",
        ))

    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(title="가용재고 (포)", showgrid=True, gridcolor="#eee"),
        yaxis2=dict(title="출고량 (포)", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 금일 생산 보고 내역
# ══════════════════════════════════════════════════════════════

with st.expander("📝 금일 생산 보고 내역", expanded=False):
    df_prod = load_production_today()
    if df_prod.empty:
        st.write("오늘 파싱된 생산 보고가 없습니다.")
    else:
        st.dataframe(
            df_prod[["reported_at", "product", "quantity", "from_source", "raw_text"]]
            .rename(columns={"reported_at": "보고시각", "product": "품목",
                             "quantity": "수량(포)", "from_source": "출처",
                             "raw_text": "원문"}),
            use_container_width=True,
            hide_index=True,
        )

st.divider()


# ══════════════════════════════════════════════════════════════
# AI 분석 레포트
# ══════════════════════════════════════════════════════════════

st.subheader("🤖 AI 분석 레포트")

today_str = date.today().isoformat()
ai_report = load_ai_report(today_str)

col_rep, col_btn = st.columns([5, 1])
with col_rep:
    if ai_report:
        st.markdown(
            f'<div class="ai-report-box">💡 {ai_report}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"생성일: {today_str}  (17:05 자동 또는 수동 실행)")
    else:
        st.info("오늘의 AI 레포트가 아직 없습니다. (17:05 자동 생성 또는 수동 실행)")

with col_btn:
    st.write("")
    if st.button("▶ 지금 분석", use_container_width=True,
                 help="17시 전에도 수동으로 AI 분석을 실행합니다."):
        with st.spinner("AI 분석 중..."):
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            try:
                from inventory_server import generate_ai_report, init_db
                init_db()
                generate_ai_report()
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"오류: {e}")


# ══════════════════════════════════════════════════════════════
# 사이드바: 수동 재고 보정
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 수동 재고 보정")
    st.caption("실사 후 전산 차이 발생 시 직접 입력")

    product_sel = st.selectbox("품목", ["수도용", "원예용"])
    manual_qty  = st.number_input("실제 가용재고 (포)", min_value=0, step=100)
    memo        = st.text_input("메모 (사유)")

    if st.button("보정 저장", type="primary", use_container_width=True):
        if DB_PATH.exists():
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from inventory_server import save_inventory, get_latest_stock, get_today_production
            prev     = get_latest_stock(product_sel)
            prod_qty = get_today_production(product_sel)
            ship_qty = max(prev + prod_qty - manual_qty, 0)
            save_inventory(product_sel, prev, prod_qty, ship_qty,
                           manual_qty, source=f"manual:{memo}")
            st.success(f"✅ {product_sel} {manual_qty:,}포로 보정 완료")
            st.cache_data.clear()
        else:
            st.error("DB가 없습니다. 서버를 먼저 실행하세요.")

    st.divider()
    st.subheader("🔔 경고 임계값 설정")
    new_min  = st.number_input("최소 가용재고 경고 (포)", value=WARN_MIN,  step=1000)
    new_diff = st.number_input("오차 허용 비율 (%)",      value=WARN_DIFF, step=5)
    if st.button("임계값 적용", use_container_width=True):
        os.environ["WARN_MIN_STOCK"]  = str(new_min)
        os.environ["WARN_DIFF_RATIO"] = str(new_diff)
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("**운영 정보**")
    st.write("현재 모드:", "🔥 피크 (10분)" if is_peak_season() else "🟢 평시 (08/13/17시)")
    st.write(f"DB: `{DB_PATH}`")
