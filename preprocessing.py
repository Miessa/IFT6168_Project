import os
import pandas as pd
import numpy as np
import ast

from data_aquisition import NHLDataDownloader

def parse_coordinates(val):
    try:
        if isinstance(val, str):
            val = ast.literal_eval(val)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return val
    except:
        return (None, None)
    return (None, None)

def convert_time_to_seconds(t):
    try:
        minutes, seconds = map(int, t.split(':'))
        return minutes * 60 + seconds
    except:
        return None

def preprocess_shot_data(start_season=2016, final_season=2023, data_dir=None):
    downloader = NHLDataDownloader(start_season, final_season, data_dir)

    if not os.path.exists(downloader.nhl_games_file_path):
        downloader.get_nhl_game_data()

    df = downloader.parse_nhl_game_data()
    downloader.save_player_names()

    if df.empty:
        print("No shot data available.")
        return None

    # Convert goal/no goal to binary
    df['result'] = df['result'].replace({'goal': 1, 'no goal': 0})

    # Parse coordinate strings into numerical x/y columns
    df[['xcoord', 'ycoord']] = df['coordinates'].apply(parse_coordinates).apply(pd.Series)

    # Feature engineering
    df['distance_to_net'] = np.sqrt((df['xcoord'] - 89)**2 + df['ycoord']**2)
    df['shot_angle'] = np.abs(np.arctan2(df['ycoord'], (89 - df['xcoord'])) * 180 / np.pi)

    # Remove raw coordinate columns
    df.drop(columns=['coordinates', 'xcoord', 'ycoord'], inplace=True)

    # Time conversion
    df['timeInPeriod'] = df['timeInPeriod'].apply(convert_time_to_seconds)

    # Fill missing goalies with team mode
    df['goalie'] = df.groupby('teamId')['goalie'].transform(
        lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
    )

    # Impute shot features per shooter
    df['distance_to_net'] = df.groupby('shooter')['distance_to_net'].transform(
        lambda x: x.fillna(x.median())
    )
    df['shot_angle'] = df.groupby('shooter')['shot_angle'].transform(
        lambda x: x.fillna(x.median())
    )
    df['shotType'] = df.groupby('shooter')['shotType'].transform(
        lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else np.nan
    )
    df['shotType'] = df['shotType'].fillna(df['shotType'].mode().iloc[0])

    # Save final processed data
    processed_path = os.path.join(downloader.data_dir, 'processed_shot_events.csv')
    df.to_csv(processed_path, index=False)
    print(f"✅ Preprocessed data saved to: {processed_path}")
    return df

if __name__ == "__main__":
    preprocess_shot_data()
