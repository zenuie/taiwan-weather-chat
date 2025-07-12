# --- START OF FILE models.py ---

from sqlalchemy import Column, Float, String, DateTime, Integer, UniqueConstraint
from .database import Base


class WeatherForecast(Base):
    __tablename__ = "weather_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    county = Column(String, index=True)
    town = Column(String, index=True)
    geocode = Column(String)

    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime)

    t = Column(Float, nullable=True)
    rh = Column(Float, nullable=True)
    pop6h = Column(Float, nullable=True)
    pop12h = Column(Float, nullable=True)
    wx = Column(String, nullable=True)
    weather_description = Column(String, nullable=True)
    wind_speed = Column(Float, nullable=True)
    wind_direction = Column(String, nullable=True)

    # 新增：定義唯一性約束
    # 對於同一個鄉鎮(town)，預報的開始時間(start_time)必須是唯一的。
    __table_args__ = (UniqueConstraint('town', 'start_time', name='_town_start_time_uc'),)
