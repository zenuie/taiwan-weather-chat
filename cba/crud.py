# --- START OF FILE crud.py ---

from sqlalchemy.orm import Session
from typing import List, Dict, Any
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from . import models


def save_weather_forecasts(db: Session, forecasts: List[Dict[str, Any]]):
    """
    將天氣預報資料儲存到資料庫。

    Args:
        db: SQLAlchemy Session 物件。
        forecasts: 一個包含預報資料的 dict 列表 (由 transformer 產生)。
    """
    if not forecasts:
        return 0

    # 建立一個 SQLAlchemy Core 的 insert statement
    stmt = sqlite_insert(models.WeatherForecast)

    # 加上 ON CONFLICT DO NOTHING 子句
    # 當發生唯一性約束衝突時（由 index_elements 指定的欄位組合），不執行任何操作。
    stmt = stmt.on_conflict_do_nothing(
        index_elements=['town', 'start_time']
    )

    # 執行這個 statement，將 `forecasts` 列表作為要插入的資料
    db.execute(stmt, forecasts)
    db.commit()

    # 回傳嘗試插入的筆數
    return len(forecasts)


def get_all_forecasts(db: Session) -> List[models.WeatherForecast]:
    """從資料庫中獲取所有天氣預報紀錄。"""
    return db.query(models.WeatherForecast).all()
