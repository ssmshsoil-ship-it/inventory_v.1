# -*- coding: utf-8 -*-
"""
진짜 가용 재고 관리 서버
실행: python src/inventory_server.py

전제: Tailscale VPN 환경에서 운영
스케줄: 피크시즌(3~4월) 10분 간격 / 평시 08:00, 13:00, 17:00
"""

import os
import re
import json
import sqlite3
import logging
import requests
from pathlib import Path
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# ─── 환경변수 로드 ────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env.inventory")

AMARANTH_API_URL  = os.getenv("AMARANTH_API_URL",  "http://localhost:8080/api")
AMARANTH_API_KEY  = os.getenv("AMARANTH_API_KEY",  "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
KAKAO_LOG_PATH    = Path(os.getenv("KAKAO_LOG_PATH", "data/kakao_log.txt"))
PROD_REPORT_PATH  = Path(os.getenv("PROD_REPORT_PATH", "data/production_report.txt"))
DB_PATH           = Path(os.getenv("DB_PATH", "data/inventory.db"))
KST               = ZoneInfo("Asia/Seoul")

# ─── 로깅 ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/inventory_server.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. DATABASE
# ══════════════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS inventory_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT NOT NULL,
                product          TEXT NOT NULL,   -- '수도용' | '원예용'
                prev_stock       INTEGER DEFAULT 0,
                production_qty   INTEGER DEFAULT 0,
                shipment_qty     INTEGER DEFAULT 0,
                available_stock  INTEGER DEFAULT 0,
                source           TEXT DEFAULT 'scheduled'
            );

            CREATE TABLE IF NOT EXISTS production_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                reported_at TEXT NOT NULL,
                product     TEXT NOT NULL,
                quantity    INTEGER NOT NULL,
                from_source TEXT NOT NULL,  -- 'kakao' | 'text_file'
                raw_text    TEXT
            );

            CREATE TABLE IF NOT EXISTS ai_reports (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL UNIQUE,
                report_text TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
        """)
    log.info("DB 초기화 완료: %s", DB_PATH)


def get_latest_stock(product: str) -> int:
    """특정 품목의 가장 최근 가용재고 반환"""
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT available_stock FROM inventory_log "
            "WHERE product = ? ORDER BY ts DESC LIMIT 1",
            (product,)
        ).fetchone()
    return row[0] if row else 0


def save_inventory(product: str, prev: int, production: int,
                   shipment: int, available: int, source: str = "scheduled"):
    ts = datetime.now(KST).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO inventory_log "
            "(ts, product, prev_stock, production_qty, shipment_qty, available_stock, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, product, prev, production, shipment, available, source),
        )
    log.info("[재고저장] %s | 전일=%d 생산=%d 출고=%d → 가용=%d",
             product, prev, production, shipment, available)


def save_production_report(product: str, qty: int, from_source: str, raw: str):
    ts = datetime.now(KST).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO production_reports (reported_at, product, quantity, from_source, raw_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, product, qty, from_source, raw),
        )
    log.info("[생산보고] %s %d포 (출처: %s)", product, qty, from_source)


def save_ai_report(report_text: str):
    today = date.today().isoformat()
    ts    = datetime.now(KST).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT OR REPLACE INTO ai_reports (report_date, report_text, created_at) "
            "VALUES (?, ?, ?)",
            (today, report_text, ts),
        )
    log.info("[AI레포트] 저장 완료")


def get_today_production(product: str) -> int:
    """오늘 누적 생산 보고량"""
    today = date.today().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT COALESCE(SUM(quantity), 0) FROM production_reports "
            "WHERE product = ? AND reported_at >= ?",
            (product, today),
        ).fetchone()
    return row[0] if row else 0


# ══════════════════════════════════════════════════════════════
# 2. 아마란스10 API 클라이언트
# ══════════════════════════════════════════════════════════════

def fetch_shipment_from_amaranth(product: str) -> int:
    """
    아마란스10 ERP에서 당일 출고 수량을 조회한다.

    실제 연동 시 아래 주석 해제 후 AMARANTH_API_URL / AMARANTH_API_KEY 설정:
        GET {AMARANTH_API_URL}/shipments?date={today}&product={product}
        Headers: { "Authorization": "Bearer {AMARANTH_API_KEY}" }
        Response: { "total_qty": 12345 }

    현재는 Mock 응답 반환.
    """
    if AMARANTH_API_KEY:
        # ── 실제 API 연동 (API 스펙 확정 후 활성화) ──────────────
        # today = date.today().isoformat()
        # try:
        #     resp = requests.get(
        #         f"{AMARANTH_API_URL}/shipments",
        #         params={"date": today, "product_type": product},
        #         headers={"Authorization": f"Bearer {AMARANTH_API_KEY}"},
        #         timeout=10,
        #     )
        #     resp.raise_for_status()
        #     return int(resp.json()["total_qty"])
        # except requests.RequestException as e:
        #     log.error("아마란스 API 오류: %s", e)
        #     return 0
        pass

    # ── Mock 데이터 ───────────────────────────────────────────
    import random
    mock = {"수도용": random.randint(10000, 50000), "원예용": random.randint(2000, 10000)}
    qty = mock.get(product, 0)
    log.info("[MOCK] 아마란스 출고량 %s = %d포", product, qty)
    return qty


# ══════════════════════════════════════════════════════════════
# 3. 생산 보고 파서 (카카오톡 / 텍스트 파일)
# ══════════════════════════════════════════════════════════════

# 파싱 패턴 예시:
#   "수도용 생산완료 15000포"
#   "원예용 12,500포 완료"
#   "[오전 09:30] 수도용상토 15000포 생산 완료"
_PROD_PATTERN = re.compile(
    r"(수도용|원예용)[^\d]*?([\d,]+)\s*포",
    re.IGNORECASE,
)


def _parse_production_line(line: str) -> list[tuple[str, int]]:
    """한 줄에서 (품목, 수량) 목록 추출"""
    results = []
    for m in _PROD_PATTERN.finditer(line):
        product = "수도용" if "수도" in m.group(1) else "원예용"
        qty     = int(m.group(2).replace(",", ""))
        results.append((product, qty))
    return results


def _last_read_position(path: Path) -> int:
    """파일별 마지막 읽기 위치 추적 (재시작 후 중복 파싱 방지)"""
    pos_file = path.with_suffix(".pos")
    return int(pos_file.read_text()) if pos_file.exists() else 0


def _save_read_position(path: Path, pos: int):
    path.with_suffix(".pos").write_text(str(pos))


def parse_production_reports(source_path: Path, from_source: str):
    """
    텍스트 파일(카카오 내보내기 또는 별도 보고 파일)을 읽어
    생산 완료 보고를 파싱하고 DB에 저장한다.

    카카오톡 내보내기 예시 경로:
        C:/Users/PCuser/AppData/Local/Kakao/KakaoTalk/log/생산팀.txt
    해당 경로를 .env.inventory의 KAKAO_LOG_PATH에 설정.
    """
    if not source_path.exists():
        return

    last_pos = _last_read_position(source_path)

    with open(source_path, encoding="utf-8", errors="ignore") as f:
        f.seek(last_pos)
        new_lines = f.readlines()
        current_pos = f.tell()

    found = 0
    for line in new_lines:
        for product, qty in _parse_production_line(line):
            save_production_report(product, qty, from_source, line.strip())
            found += 1

    if found:
        log.info("[파서] %s: %d건 신규 생산보고 파싱", source_path.name, found)
        _save_read_position(source_path, current_pos)
    elif new_lines:
        # 새 라인이 있어도 생산보고 없으면 위치만 갱신
        _save_read_position(source_path, current_pos)


# ══════════════════════════════════════════════════════════════
# 4. 재고 계산
# ══════════════════════════════════════════════════════════════

PRODUCTS = ["수도용", "원예용"]


def calculate_and_save_inventory():
    """
    진짜 가용 재고 = (전일 재고 + 당일 생산 보고량) - 아마란스 실시간 출고량
    """
    log.info("── 재고 계산 시작 ──")

    # 생산 보고 파싱 (카카오 + 별도 텍스트 파일)
    parse_production_reports(KAKAO_LOG_PATH,  "kakao")
    parse_production_reports(PROD_REPORT_PATH, "text_file")

    for product in PRODUCTS:
        prev_stock   = get_latest_stock(product)
        production   = get_today_production(product)
        shipment     = fetch_shipment_from_amaranth(product)
        available    = prev_stock + production - shipment

        save_inventory(product, prev_stock, production, shipment, available)

    log.info("── 재고 계산 완료 ──")


# ══════════════════════════════════════════════════════════════
# 5. AI 분석 레포트 (17시 실행)
# ══════════════════════════════════════════════════════════════

def generate_ai_report():
    """당일 재고 오차를 Claude API로 분석하여 한 줄 요약 저장"""
    if not ANTHROPIC_API_KEY:
        log.warning("[AI레포트] ANTHROPIC_API_KEY 미설정 — 건너뜀")
        return

    import anthropic

    today = date.today().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            """
            SELECT product,
                   SUM(production_qty) AS 생산합계,
                   SUM(shipment_qty)   AS 출고합계,
                   MIN(prev_stock)     AS 전일재고,
                   available_stock     AS 현재가용
            FROM inventory_log
            WHERE ts >= ? AND source = 'scheduled'
            GROUP BY product
            """,
            (today,),
        ).fetchall()

    if not rows:
        log.warning("[AI레포트] 오늘 데이터 없음 — 건너뜀")
        return

    summary_lines = []
    for row in rows:
        product, prod, ship, prev, avail = row
        system_stock = prev + prod - ship  # 전산 계산값
        diff = avail - system_stock
        summary_lines.append(
            f"- {product}: 전산계산={system_stock:,}포 / 현재가용={avail:,}포 / 오차={diff:+,}포"
        )

    prompt = (
        f"오늘({today}) 성화 상토 출고 및 재고 현황입니다:\n"
        + "\n".join(summary_lines)
        + "\n\n오차 원인과 조치 필요 사항을 실무 담당자 관점에서 "
          "50자 이내 한 줄로 요약해주세요."
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        report_text = msg.content[0].text.strip()
    except Exception as e:
        log.error("[AI레포트] API 오류: %s", e)
        report_text = f"AI 분석 실패: {e}"

    save_ai_report(report_text)
    log.info("[AI레포트] %s", report_text)


# ══════════════════════════════════════════════════════════════
# 6. 스케줄러
# ══════════════════════════════════════════════════════════════

def is_peak_season() -> bool:
    return datetime.now(KST).month in (3, 4)


def setup_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=KST)

    if is_peak_season():
        # 피크시즌: 10분마다 재고 계산
        scheduler.add_job(
            calculate_and_save_inventory,
            trigger=IntervalTrigger(minutes=10),
            id="inventory_peak",
            name="재고계산(피크)",
            max_instances=1,
            replace_existing=True,
        )
        log.info("스케줄 설정: 피크시즌 모드 (10분 간격)")
    else:
        # 평시: 08:00, 13:00, 17:00
        for hour in (8, 13, 17):
            scheduler.add_job(
                calculate_and_save_inventory,
                trigger=CronTrigger(hour=hour, minute=0),
                id=f"inventory_{hour:02d}",
                name=f"재고계산({hour:02d}시)",
                max_instances=1,
                replace_existing=True,
            )
        log.info("스케줄 설정: 평시 모드 (08, 13, 17시)")

    # 17시 AI 레포트 (시즌 무관)
    scheduler.add_job(
        generate_ai_report,
        trigger=CronTrigger(hour=17, minute=5),
        id="ai_report",
        name="AI 분석 레포트",
        max_instances=1,
        replace_existing=True,
    )

    # 자정: 스케줄 재평가 (시즌 전환 대응)
    scheduler.add_job(
        lambda: _reload_schedule(scheduler),
        trigger=CronTrigger(hour=0, minute=1),
        id="schedule_reload",
        name="스케줄 재평가",
    )

    return scheduler


def _reload_schedule(scheduler: BlockingScheduler):
    """자정에 피크/평시 스케줄을 재평가하여 교체"""
    peak = is_peak_season()
    log.info("스케줄 재평가: %s", "피크시즌" if peak else "평시")

    for job_id in ("inventory_peak", "inventory_08", "inventory_13", "inventory_17"):
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass

    if peak:
        scheduler.add_job(
            calculate_and_save_inventory,
            trigger=IntervalTrigger(minutes=10),
            id="inventory_peak",
            name="재고계산(피크)",
            max_instances=1,
        )
    else:
        for hour in (8, 13, 17):
            scheduler.add_job(
                calculate_and_save_inventory,
                trigger=CronTrigger(hour=hour, minute=0),
                id=f"inventory_{hour:02d}",
                name=f"재고계산({hour:02d}시)",
                max_instances=1,
            )


# ══════════════════════════════════════════════════════════════
# 7. 메인
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=== 재고 관리 서버 시작 (Tailscale 환경) ===")
    init_db()

    # 시작 시 즉시 1회 실행
    calculate_and_save_inventory()

    scheduler = setup_scheduler()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("서버 종료")
