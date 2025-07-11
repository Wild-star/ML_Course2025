#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import sys

def process_daily_aggregation(
    input_path: str,
    output_path: str,
    sep: str = ',',
    datetime_col: str = 'DateTime',
    datetime_format: str = '%Y-%m-%d %H:%M:%S'
) -> pd.DataFrame:
    """
    读取原始 CSV，处理缺失值，按日对各列做聚合，并保存结果到 output_path。
    
    参数：
      - input_path: 输入 CSV 路径
      - output_path: 输出 CSV 路径
      - sep: 分隔符，默认 ','
      - datetime_col: 时间列名，默认 'DateTime'
      - datetime_format: 已知的时间字符串格式，默认 '%Y/%m/%d %H:%M:%S'（可按需修改）
    
    聚合规则：
      Global_active_power, Global_reactive_power, Sub_metering_1/2/3: sum
      Voltage, Global_intensity: mean
      RR, NBJRR1, NBJRR5, NBJRR10, NBJBROU: first（可改 last/min/max）
    
    缺失值处理示例：时间插值 + 前后填充，可按需修改。
    """
    # 1. 读取 CSV，先不过早转换时间
    try:
        df = pd.read_csv(
            input_path,
            sep=sep,
            na_values=['?', '', 'NA']
        )
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        raise

    # 2. 确认时间列存在
    if datetime_col not in df.columns:
        raise ValueError(f"Time column '{datetime_col}' not found. Available columns: {df.columns.tolist()}")

    # 3. 显式转换时间列到 datetime 类型
    try:
        # 若格式固定，显式指定 format 会更快、更准确
        df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format)
    except Exception as e:
        # 如果转换失败，可尝试不指定 format 由 pandas 自动推断
        print(f"Warning: to_datetime with format '{datetime_format}' failed: {e}", file=sys.stderr)
        print("Attempting automatic datetime parsing...", file=sys.stderr)
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    # 检查是否有解析失败的情况
    if df[datetime_col].isna().any():
        n_bad = df[datetime_col].isna().sum()
        print(f"Warning: {n_bad} rows in '{datetime_col}' could not be parsed and are NaT", file=sys.stderr)
        # 根据需求，可选择 dropna 或其他处理；这里先 drop：
        df = df.dropna(subset=[datetime_col])

    # 4. 设置索引并排序
    df.set_index(datetime_col, inplace=True)
    df.sort_index(inplace=True)

    # 5. 缺失值处理
    # 时间序列插值 + 前后填充
    try:
        df = df.interpolate(method='time')
    except Exception:
        # 若索引不连续或某些列无法插值，可跳过
        pass
    df = df.fillna(method='bfill').fillna(method='ffill')

    # 6. 确保数值列类型正确
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 7. 计算新列 sub_metering_remainder
    required_for_remainder = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    if all(col in df.columns for col in required_for_remainder):
        # 注意：Global_active_power 单位假设为 kW，乘1000得 W，/60 得每分钟 Wh（或类似含义），与 Sub_metering_* 单位对齐
        df['sub_metering_remainder'] = (
            df['Global_active_power'] * 1000.0 / 60.0
            - (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
        )
        # 若希望对 remainder 做缺失或异常值处理，也可在此加入
    else:
        # 若缺少列，提醒用户
        missing = [col for col in required_for_remainder if col not in df.columns]
        print(f"Warning: cannot compute sub_metering_remainder because missing columns: {missing}", file=sys.stderr)
    # 再次填充因类型转换导致的新 NaN
    try:
        df = df.interpolate(method='time')
    except Exception:
        pass
    df = df.fillna(method='bfill').fillna(method='ffill')

    df['date'] = df.index.date
    agg_dict = {}
    for col in ['Global_active_power', 'Global_reactive_power',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','sub_metering_remainder']:
        if col in df.columns:
            agg_dict[col] = 'sum'
    for col in ['Voltage', 'Global_intensity']:
        if col in df.columns:
            agg_dict[col] = 'mean'
    for col in ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']:
        if col in df.columns:
            agg_dict[col] = 'first'
    if not agg_dict:
        raise ValueError("未检测到可聚合的列，请检查列名是否正确。")

    # 9. 按日聚合
    df_daily = df.groupby('date').agg(agg_dict).reset_index()

    # 10. 可调整日期列格式或列名
    #    例如：df_daily['date'] = df_daily['date'].astype(str)
    #    df_daily.rename(columns={'date': 'Date'}, inplace=True)

    # 11. 保存结果
    try:
        df_daily.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}", file=sys.stderr)
        raise

    print(f"Daily aggregated data saved to: {output_path}")
    return df_daily

def parse_args():
    parser = argparse.ArgumentParser(description="按天聚合电力数据 CSV（Pandas 2.0.3 兼容）")
    parser.add_argument('--input',default="D:\\Desktop\\Work\\machinelearn\\file\\test.csv", help="输入 CSV 文件路径")
    parser.add_argument('--output',default="D:\\Desktop\\Work\\machinelearn\\file\\test_finall.csv" ,help="输出 CSV 文件路径")
    parser.add_argument('--sep', default=',', help="输入文件分隔符，默认 ','；可设为 '\\t'、';' 等")
    parser.add_argument('--datetime_col', default='DateTime', help="时间列名，默认 'DateTime'")
    parser.add_argument('--datetime_format', default='%Y-%m-%d %H:%M:%S', 
                        help="时间解析格式")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        df_daily = process_daily_aggregation(
            input_path=args.input,
            output_path=args.output,
            sep=args.sep,
            datetime_col=args.datetime_col,
            datetime_format=args.datetime_format
        )
        # 打印前几行以便检查
        print(df_daily.head())
    except Exception as e:
        print(f"Processing failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
