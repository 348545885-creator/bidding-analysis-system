"""
投标决策辅助分析系统 - FastAPI 统一 API 服务

整合数据查询、价格分析、数据采集和 AI 对话功能。
作为系统的 HTTP 入口，提供 RESTful API 和静态文件服务。

启动方式:
    uvicorn app:app --host 0.0.0.0 --port 5001
    或 python app.py
"""

import os, sys, re, json
from datetime import datetime
from typing import Optional

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

from core.数据采集器 import DatabaseManager, selenium_scrape_realtime, Config as ScraperConfig
from core.config import DISPLAY_CITIES, PROJECT_TYPES, CITY_NAME_MAP, EVALUATION_METHODS

app = FastAPI(title="投标报价分析系统", version="2.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

db_manager = DatabaseManager()

class LLMClient:
    def __init__(self):
        self._load_env()
        self.api_key = os.getenv("ALIBABA_CLOUD_API_KEY", "")
        self.api_base = os.getenv("ALIBABA_CLOUD_API_BASE", "https://dashscope.aliyuncs.com/api/v1")
        self.model = os.getenv("ALIBABA_CLOUD_MODEL", "qwen-turbo")

    def _load_env(self):
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()

    def chat(self, message: str, history: list = None) -> str:
        if not self.api_key:
            return None
        messages = (history or []) + [{"role": "user", "content": message}]
        try:
            resp = requests.post(
                f"{self.api_base}/services/aigc/text-generation/generation",
                json={"model": self.model, "input": {"messages": messages},
                       "parameters": {"max_tokens": 2000, "temperature": 0.7}},
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=60
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") == "Success":
                return result.get("output", {}).get("text", "")
            return None
        except Exception:
            return None

llm_client = LLMClient()


# ─── Pydantic Models ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    control_price: str = ""
    bidder_count_min: Optional[int] = None
    bidder_count_max: Optional[int] = None
    project_type: str = ""
    evaluation_method: str = ""
    city: str = ""

class ScrapeRequest(BaseModel):
    max_pages: int = 20
    start_date: str = ""
    end_date: str = ""

class ChatRequest(BaseModel):
    message: str = ""
    history: list = []


# ─── Helper ─────────────────────────────────────────────────────

def _coerce_nonneg_int(value, label: str):
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{label}格式不正确"
    if isinstance(value, int):
        return (value, None) if value >= 0 else (None, f"{label}不能为负数")
    if isinstance(value, float) and value.is_integer():
        iv = int(value)
        return (iv, None) if iv >= 0 else (None, f"{label}不能为负数")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None, None
        try:
            iv = int(float(s))
            return (iv, None) if iv >= 0 else (None, f"{label}不能为负数")
        except ValueError:
            return None, f"{label}必须是有效数字"
    return None, f"{label}格式不正确"


def _generate_analysis_report(control_price, suggested_price, suggested_min, suggested_max,
                                confidence, avg_discount, avg_winning_price,
                                similar_projects, **kwargs):
    report = []
    report.append("=" * 60)
    report.append("📊 投标报价智能分析报告")
    report.append("=" * 60)
    report.append(f"\n📌 招标控制价：¥{control_price:,.2f}")
    report.append(f"📌 投标人数量：{kwargs.get('bidder_count_min', '?')} ~ {kwargs.get('bidder_count_max', '?')} 家")
    if kwargs.get('project_type'):
        report.append(f"📌 项目类型：{kwargs['project_type']}")
    if kwargs.get('city'):
        report.append(f"📌 项目城市：{kwargs['city']}")
    report.append("")
    report.append("-" * 60)
    report.append("🎯 推荐报价分析")
    report.append("-" * 60)
    report.append(f"\n💰 建议报价：¥{suggested_price:,.2f}")
    report.append(f"📈 建议区间：¥{suggested_min:,.2f} ~ ¥{suggested_max:,.2f}")
    report.append(f"⭐ 置信度：{confidence * 100:.1f}%")
    report.append(f"📉 平均折扣率：{avg_discount * 100:.2f}%")
    report.append(f"📊 历史中标均价：¥{avg_winning_price:,.2f}")
    report.append("")
    report.append("-" * 60)
    report.append("📋 参考项目")
    report.append("-" * 60)
    for p in similar_projects[:5]:
        name = p.get('project_name', '')[:30]
        wp = p.get('winning_price') or '无'
        cp = p.get('control_price') or '无'
        report.append(f"\n  · {name}")
        report.append(f"    控制价：¥{cp}  中标价：¥{wp}")
    report.append("")
    report.append("=" * 60)
    report.append("免责声明：本分析仅供参考，实际报价请结合项目具体情况。")
    report.append("=" * 60)
    return "\n".join(report)


# ─── Routes ──────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/api/test")
def api_test():
    return {"success": True, "message": "API服务正常运行"}


@app.get("/api/projects")
def get_projects(
    city: str = "", project_type: str = "", start_date: str = "",
    end_date: str = "", keyword: str = "", page: int = 1, page_size: int = 20
):
    try:
        offset = (page - 1) * page_size
        conditions = ["announcement_date IS NOT NULL"]
        params = []
        if city:
            conditions.append("city = ?"); params.append(city)
        if project_type:
            conditions.append("project_type = ?"); params.append(project_type)
        if start_date:
            conditions.append("announcement_date >= ?"); params.append(start_date)
        if end_date:
            conditions.append("announcement_date <= ?"); params.append(end_date)
        if keyword:
            conditions.append("(project_name LIKE ? OR winning_candidate LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        where = " AND ".join(conditions)

        with db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute(f"SELECT COUNT(*) FROM bidding_results WHERE {where}", params)
            total = c.fetchone()[0]
            c.execute(f"""
                SELECT id, city, project_type, project_name, bid_price, winning_price,
                       winning_candidate, comprehensive_score, opening_time, announcement_date
                FROM bidding_results WHERE {where}
                ORDER BY announcement_date DESC LIMIT ? OFFSET ?
            """, params + [page_size, offset])
            projects = []
            for row in c.fetchall():
                projects.append({
                    "id": row[0], "city": row[1], "project_type": row[2],
                    "project_name": row[3], "bid_price": row[4],
                    "winning_price": row[5], "winning_candidate": row[6],
                    "comprehensive_score": row[7], "opening_time": row[8],
                    "announcement_date": row[9]
                })
        return {"success": True, "total": total, "page": page, "page_size": page_size, "projects": projects}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/filter/options")
def get_filter_options():
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT DISTINCT city FROM bidding_results WHERE city IS NOT NULL ORDER BY city")
            db_cities = [r[0] for r in c.fetchall()]
            c.execute("SELECT DISTINCT project_type FROM bidding_results WHERE project_type IS NOT NULL ORDER BY project_type")
            db_types = [r[0] for r in c.fetchall()]
            c.execute("SELECT DISTINCT evaluation_method FROM bidding_results WHERE evaluation_method IS NOT NULL ORDER BY evaluation_method")
            methods = [r[0] for r in c.fetchall()]
        
        # 按照 DISPLAY_CITIES 的固定顺序返回所有城市（无论是否有数据）
        all_cities = DISPLAY_CITIES
        extra_cities = [city for city in db_cities if city not in all_cities]
        final_cities = all_cities + extra_cities
        
        return {
            "success": True,
            "options": {
                "cities": final_cities,  # 使用固定顺序 + 额外城市
                "project_types": list(dict.fromkeys(db_types + PROJECT_TYPES)),
                "evaluation_methods": methods if methods else EVALUATION_METHODS
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/statistics")
def get_statistics(city: str = "", project_type: str = ""):
    try:
        where_clauses = []
        params = []
        if city:
            where_clauses.append("city = ?"); params.append(city)
        if project_type:
            where_clauses.append("project_type = ?"); params.append(project_type)
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        with db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute(f"SELECT COUNT(*) FROM bidding_results {where_sql}", params)
            total = c.fetchone()[0]

            c.execute(f"SELECT city, COUNT(*) FROM bidding_results {where_sql} GROUP BY city ORDER BY COUNT(*) DESC", params)
            db_cities = {r[0]: r[1] for r in c.fetchall() if r[0] and r[0].strip()}
            name_map = CITY_NAME_MAP
            merged = {}
            for k, v in db_cities.items():
                target = name_map.get(k, k)
                merged[target] = merged.get(target, 0) + v
            city_stats = [{"city": c, "count": merged.get(c, 0)} for c in DISPLAY_CITIES]

            c.execute(f"SELECT project_type, COUNT(*) FROM bidding_results {where_sql} GROUP BY project_type ORDER BY COUNT(*) DESC", params)
            db_types_list = [{"type": r[0], "count": r[1]} for r in c.fetchall() if r[0] and r[0].strip()]
            all_types = PROJECT_TYPES
            db_types = {t["type"]: t["count"] for t in db_types_list}
            type_stats = [{"type": t, "count": db_types.get(t, 0)} for t in all_types]

        return {"success": True, "total": total, "city_distribution": city_stats, "type_distribution": type_stats}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/analyze")
def analyze_price(req: AnalyzeRequest):
    """
    报价智能分析算法

    算法流程:
      1. 按筛选条件（城市、类型、投标人数量）查询相似历史项目
      2. 提取每对(控制价, 中标价)，计算折扣率 = (控制价 - 中标价) / 控制价
      3. 过滤异常值：仅保留 10% ≤ 中标价/控制价 ≤ 100% 的记录
      4. 平均折扣率 → 得出推荐降价幅度
      5. 建议报价 = 输入控制价 × (1 - 平均折扣率)
      6. 标准差 → 报价区间宽度（±0.5σ）
      7. 置信度：min(0.95, 0.5 + 样本量/100 × 0.3)，样本越多置信度越高
    """
    try:
        bidder_count_min, err_min = _coerce_nonneg_int(req.bidder_count_min, "投标人数量最小值")
        bidder_count_max, err_max = _coerce_nonneg_int(req.bidder_count_max, "投标人数量最大值")
        errors = []
        if err_min: errors.append(err_min)
        elif bidder_count_min is None: errors.append("请输入投标人数量最小值")
        if err_max: errors.append(err_max)
        elif bidder_count_max is None: errors.append("请输入投标人数量最大值")
        if bidder_count_min is not None and bidder_count_max is not None and bidder_count_min > bidder_count_max:
            errors.append("投标人数量最小值不能大于最大值")

        ctrl_str = re.sub(r"[^\d.]", "", req.control_price)
        if not ctrl_str:
            errors.append("请输入招标控制价")
        else:
            try:
                control_price = float(ctrl_str)
                if control_price <= 0:
                    errors.append("招标控制价必须是正数")
            except ValueError:
                errors.append("招标控制价格式不正确")

        if errors:
            return JSONResponse(status_code=400, content={"success": False, "error": "; ".join(errors)})

        # ── 步骤1: 查询相似历史项目 ──
        sql = "SELECT * FROM bidding_results WHERE 1=1"
        params = []
        if req.project_type:
            sql += " AND project_type = ?"; params.append(req.project_type)
        if req.city:
            sql += " AND city = ?"; params.append(req.city)
        if req.evaluation_method:
            sql += " AND evaluation_method = ?"; params.append(req.evaluation_method)
        if bidder_count_min is not None:
            sql += " AND (bidder_count >= ? OR bidder_count IS NULL)"; params.append(bidder_count_min)
        if bidder_count_max is not None:
            sql += " AND (bidder_count <= ? OR bidder_count IS NULL)"; params.append(bidder_count_max)

        with db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute(sql, params)
            rows = c.fetchall()

        if not rows:
            return {"success": False, "analysis": None, "error": "未找到符合条件的项目数据，建议扩大筛选范围", "data_count": 0}

        similar_projects = []
        for row in rows:
            similar_projects.append({
                "project_name": row[4], "city": row[1], "project_type": row[2],
                "control_price": row[6], "winning_price": row[10],
                "bidder_count": row[7], "comprehensive_score": row[11],
                "announcement_date": row[13]
            })

        control_price = float(ctrl_str)
        winning_prices = []
        discounts = []

        # ── 步骤2: 计算每条历史项目的折扣率 ──
        # 折扣率 = (控制价 - 中标价) / 控制价，表示投标人相对控制价的降价比例
        for proj in similar_projects:
            try:
                if proj["winning_price"] and proj["control_price"]:
                    wp = float(re.sub(r"[^\d.]", "", proj["winning_price"]))
                    cp = float(re.sub(r"[^\d.]", "", proj["control_price"]))
                    # 步骤3: 异常值过滤 — 中标价应在控制价的 10%~100% 之间
                    # 低于 10% 可能是数据错误，高于 100% 是超控制价中标（罕见）
                    if cp > 0 and 0.1 * cp <= wp <= cp:
                        discount = (cp - wp) / cp
                        winning_prices.append(wp)
                        discounts.append(discount)
            except (ValueError, ZeroDivisionError):
                continue

        if not winning_prices:
            return {"success": False, "analysis": None,
                    "error": f"历史数据不足以进行有效分析（找到 {len(similar_projects)} 条相似项目，但均无有效价格数据）",
                    "data_count": len(similar_projects)}

        # ── 步骤4: 计算平均折扣率 ──
        # 平均折扣率 = 所有有效历史折扣率的算术平均值
        avg_discount = sum(discounts) / len(discounts)
        # 历史中标均价 = 所有有效中标价的算术平均值
        avg_winning_price = sum(winning_prices) / len(winning_prices)

        # ── 步骤5: 建议报价 ──
        # 对用户输入的控制价应用平均折扣率，得到建议报价
        suggested_price = control_price * (1 - avg_discount)

        # ── 步骤6: 计算报价区间（标准差法） ──
        # 用历史中标价的标准差衡量离散程度，取 ±0.5σ 作为推荐区间
        price_std = 0
        if len(winning_prices) > 1:
            mean = avg_winning_price
            variance = sum((p - mean) ** 2 for p in winning_prices) / len(winning_prices)
            price_std = variance ** 0.5
        suggested_min = suggested_price - price_std * 0.5
        suggested_max = suggested_price + price_std * 0.5

        # ── 步骤7: 置信度评估 ──
        # 基于相似项目数量：基线 50%，每 100 条增加 30%，上限 95%
        # 样本量越大，统计结果越可信
        confidence = min(0.95, 0.5 + (len(similar_projects) / 100) * 0.3)

        report_text = _generate_analysis_report(
            control_price=control_price, suggested_price=suggested_price,
            suggested_min=suggested_min, suggested_max=suggested_max,
            confidence=confidence, avg_discount=avg_discount,
            avg_winning_price=avg_winning_price,
            similar_projects=similar_projects,
            bidder_count_min=bidder_count_min, bidder_count_max=bidder_count_max,
            project_type=req.project_type, city=req.city,
            evaluation_method=req.evaluation_method
        )

        return {
            "success": True,
            "analysis": {
                "suggested_min": f"{suggested_min:,.2f}",
                "suggested_max": f"{suggested_max:,.2f}",
                "suggested_price": f"{suggested_price:,.2f}",
                "confidence": round(confidence, 2),
                "similar_projects": len(similar_projects),
                "avg_discount": round(avg_discount * 100, 2),
                "avg_winning_price": f"{avg_winning_price:,.2f}",
                "price_range": f"{suggested_max - suggested_min:,.2f}",
                "control_price": req.control_price,
                "report_text": report_text,
                "extra": {
                    "similar_projects_count": len(similar_projects),
                    "avg_discount_percent": round(avg_discount * 100, 2),
                    "price_std": f"{price_std:,.2f}",
                    "projects_used": similar_projects[:5]
                }
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"分析失败：{str(e)}"})


@app.post("/api/trigger-scrape")
def trigger_scrape(req: ScrapeRequest):
    try:
        print(f"\n{'='*60}")
        print(f"开始实时采集四川省公共资源交易网数据")
        print(f"采集页数: {req.max_pages}")
        if req.start_date and req.end_date:
            print(f"日期范围: {req.start_date} ~ {req.end_date}")
        print(f"{'='*60}\n")

        results = selenium_scrape_realtime(
            max_pages=req.max_pages,
            start_date=req.start_date or None,
            end_date=req.end_date or None
        )

        print(f"\n{'='*60}")
        print(f"采集完成！")
        print(f"获取数据: {len(results)} 条")
        print(f"{'='*60}\n")

        dates = sorted(set(r.get('announcement_date', '') for r in results if r.get('announcement_date')))
        date_info = f"日期: {dates[0]}~{dates[-1]}" if dates else ""
        msg = f"采集完成。新增 {len(results)} 条数据{f'（{date_info}）' if date_info else ''}"

        return {
            "success": True,
            "message": msg,
            "count": len(results),
            "source": "四川省公共资源交易信息网"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"采集失败: {str(e)}"})


@app.post("/api/ai/chat")
def ai_chat(req: ChatRequest):
    if not req.message.strip():
        return JSONResponse(status_code=400, content={"success": False, "error": "消息不能为空"})
    try:
        if not llm_client.api_key:
            return JSONResponse(status_code=503, content={"success": False, "error": "AI服务未配置，请检查 .env 文件中的阿里云百炼配置"})
        response = llm_client.chat(req.message, history=req.history)
        return {"success": True, "message": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"AI对话失败：{str(e)}"})


# ─── Startup ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("投标报价分析系统 - 统一API服务")
    print("=" * 60)
    print("\n启动服务...")
    print("访问地址: http://localhost:5001\n")
    print("主要接口：")
    print("  GET  /api/test              - 测试接口")
    print("  GET  /api/projects          - 获取项目列表")
    print("  GET  /api/filter/options    - 获取筛选选项")
    print("  GET  /api/statistics        - 获取统计数据")
    print("  POST /api/analyze           - 分析价格区间")
    print("  POST /api/trigger-scrape    - 触发数据采集")
    print("  POST /api/ai/chat           - AI对话")
    print("\n" + "=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=5001)
