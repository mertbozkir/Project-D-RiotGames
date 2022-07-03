import pandas as pd
import sys
train_path = sys.argv[1]
test_path  = sys.argv[2]

def feature_engineering(df):
    df['blue-red_total_golds_diff'] = df['blueTotalGolds'] - df['redTotalGolds']
    df['blue-red_current_golds_diff'] = df['blueCurrentGolds'] - df['redCurrentGolds']
    df['blue-red_total_level_diff'] = df['blueTotalLevel'] - df['redTotalLevel']
    df['blue-red_avg_level_diff'] = df['blueAvgLevel'] - df['redAvgLevel']
    df['blue-red_total_minion_diff'] = df['blueTotalMinionKills'] - df['redTotalMinionKills']
    df['blue-red_jungle_minion_diff'] = df['blueTotalJungleMinionKills'] - df['redTotalJungleMinionKills']
    df['blue-red_kill_diff'] = df['blueKill'] - df['redKill']
    df['blue-red_death_diff'] = df['blueDeath'] - df['redDeath']
    df['blue-red_assist_diff'] = df['blueAssist'] - df['redAssist']
    df['blue_total_wards'] = df['blueWardPlaced'] - df['redWardKills']
    df['red_total_wards'] = df['redWardPlaced'] - df['blueWardKills']
    df['blue-red_tower_diff'] = df['blueTowerKills'] - df['redTowerKills']
    df['blue-red_dragon_diff'] = df['blueDragon'] -df['redDragon']
    df['red_monster_kills'] = df['redDragon'] + df['redRiftHeralds']
    df['blue_monster_kills'] = df['blueDragon'] + df['blueRiftHeralds']



train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

feature_engineering(train_df)
feature_engineering(test_df)


train_df.to_csv('features/train_features.csv')
test_df.to_csv('features/test_features.csv')

