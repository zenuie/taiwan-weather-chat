# --- START OF FILE cba/etl/transformer.py (FINAL CORRECTED VERSION) ---

from typing import List, Dict, Any
import pandas as pd
import logging

# 設定日誌，方便追蹤程式執行狀況
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def transform_forecast_data(api_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    將從 CWA API 獲取的原始天氣預報資料，包含多種時間結構和 ElementValue 結構，
    轉換為扁平化且適合存入資料庫的格式。
    """
    transformed_records = []
    for item in api_data:
        if not item or not item.get("records"):
            logging.warning("API 資料格式不正確，或找不到 'records'。")
            return []

        records = item["records"]

        if not records.get("Locations"):
            logging.warning("Records 中找不到 'Locations'。")
            return []

        location_data = records["Locations"][0]
        county_name = location_data["LocationsName"]



        for town_location in location_data.get("Location", []):
            town_name = town_location.get("LocationName")
            geocode = town_location.get("Geocode")

            if not town_name:
                continue

            logging.info(f"正在處理鄉鎮: {county_name} {town_name}")

            interval_dfs = []
            point_in_time_dfs = []

            for element in town_location.get("WeatherElement", []):
                if not element.get("Time"):
                    continue

                time_df = pd.DataFrame(element["Time"])
                if time_df.empty:
                    continue

                element_dicts = time_df['ElementValue'].apply(lambda x: x[0] if isinstance(x, list) and x else {})
                values_df = element_dicts.apply(pd.Series)
                df = time_df.drop(columns='ElementValue').join(values_df)

                if 'StartTime' in df.columns and 'EndTime' in df.columns:
                    df['StartTime'] = pd.to_datetime(df['StartTime'])
                    df['EndTime'] = pd.to_datetime(df['EndTime'])
                    interval_dfs.append(df.set_index(['StartTime', 'EndTime']))
                elif 'DataTime' in df.columns:
                    df = df.rename(columns={'DataTime': 'start_time_point'})  # 避免和區間的 start_time 混淆
                    df['start_time_point'] = pd.to_datetime(df['start_time_point'])
                    point_in_time_dfs.append(df.set_index('start_time_point'))

            if not interval_dfs:
                logging.warning(f"鄉鎮 '{town_name}' 中沒有找到任何有效的『時間區間』資料，無法建立預報基礎，已跳過。")
                continue

            base_df = pd.concat([d[~d.index.duplicated(keep='first')] for d in interval_dfs], axis=1)

            if point_in_time_dfs:
                point_df = pd.concat([d[~d.index.duplicated(keep='first')] for d in point_in_time_dfs], axis=1)
                point_df_resampled = point_df.reindex(base_df.index.get_level_values('StartTime'), method='ffill')
                base_df = base_df.join(point_df_resampled)

            full_df = base_df.reset_index()
            full_df['county'] = county_name
            full_df['town'] = town_name
            full_df['geocode'] = geocode

            full_df.columns = [str(col).lower() for col in full_df.columns]

            # ✨ 關鍵修正：在這裡加入 starttime -> start_time 的映射 ✨
            rename_map = {
                'starttime': 'start_time',
                'endtime': 'end_time',
                'temperature': 't',
                'relativehumidity': 'rh',
                'probabilityofprecipitation': 'pop12h',
                'weather': 'wx',
                'weatherdescription': 'weather_description',
                'windspeed': 'wind_speed',
                'winddirection': 'wind_direction',
            }

            full_df = full_df.rename(columns=rename_map)

            numeric_cols = ['t', 'rh', 'pop6h', 'pop12h', 'wind_speed']
            for col in numeric_cols:
                if col in full_df.columns:
                    full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

            final_cols = [
                'county', 'town', 'geocode', 'start_time', 'end_time',
                't', 'rh', 'pop6h', 'pop12h', 'wx', 'weather_description',
                'wind_speed', 'wind_direction'
            ]

            existing_cols = [col for col in final_cols if col in full_df.columns]
            full_df = full_df[existing_cols]

            transformed_records.extend(full_df.to_dict('records'))

        logging.info(f"資料轉換完成，共產生 {len(transformed_records)} 筆扁平化預報資料。")
    return transformed_records
