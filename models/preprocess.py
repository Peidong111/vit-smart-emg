import re
import pandas as pd
import numpy as np


def load_signal_from_csv(csv_path: str, channel_name: str):
    # 1) 先尽量把文件读成文本
    text = None
    used_encoding = None

    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            with open(csv_path, "r", encoding=enc) as f:
                text = f.read()
            used_encoding = enc
            break
        except Exception:
            continue

    if text is None:
        raise ValueError("Could not read file with utf-8 / gbk / latin1.")

    print(f"Loaded raw file with encoding: {used_encoding}")

    # 2) 按行处理，提取数字
    lines = text.splitlines()

    parsed_rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 跳过表头行
        if "timestamp" in line.lower() and "channel" in line.lower():
            continue

        # 提取这一行里的所有数字
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)

        # 我们只接受恰好有 7 个数值的行：
        # timestamp + 6 channels
        if len(nums) == 7:
            parsed_rows.append([float(x) for x in nums])

    if len(parsed_rows) == 0:
        raise ValueError("No valid data rows found. Please check the raw file format.")

    # 3) 构建 DataFrame
    df = pd.DataFrame(
        parsed_rows,
        columns=[
            "timestamp",
            "Channel_1",
            "Channel_2",
            "Channel_3",
            "Channel_4",
            "Channel_5",
            "Channel_6",
        ],
    )

    print("Parsed columns:", list(df.columns))
    print("First 5 rows:\n", df.head())

    # 4) 支持 Channel 4 / Channel_4 两种写法
    if channel_name not in df.columns:
        alt_name = channel_name.replace(" ", "_")
        if alt_name in df.columns:
            channel_name = alt_name
        else:
            raise ValueError(
                f"Cannot find column: {channel_name}\n"
                f"Current columns: {list(df.columns)}"
            )

    signal = df[channel_name].values.astype(float)
    return signal, df


def sliding_window(signal, sampling_rate, window_sec, step_sec):
    win_size = int(window_sec * sampling_rate)
    step = int(step_sec * sampling_rate)

    windows = []
    for start in range(0, len(signal) - win_size, step):
        seg = signal[start:start + win_size]
        windows.append(seg)

    return np.array(windows), win_size, step