import os
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT_DIR = Path(os.environ.get('ROOT_DIR'))
DATA_DIR = ROOT_DIR / 'data'


def make_raw(use_pickle=True):
    filepath = DATA_DIR / 'processed/raw.pkl'
    if use_pickle:
        try:
            df = pd.read_pickle(filepath)
        except FileNotFoundError:
            df = make_raw(use_pickle=False)
    else:
        df = (pd
              .read_csv(DATA_DIR / 'raw/results.csv',
                        converters={'date': pd.to_datetime}
                        )
              .apply(lambda col: col.str.replace('Korea Republic', 'South Korea') if col.name in ['home_team', 'away_team'] else col)
              .assign(total_goals=lambda df: df.home_score + df.away_score)
              .assign(abs_goal_diff=lambda df: np.abs(df.home_score-df.away_score))
              .assign(home_win=lambda df: df.home_score > df.away_score)
              .assign(away_win=lambda df: df.home_score < df.away_score)
              )
        df.to_pickle(filepath)
    return df


def make_team_idx(use_pickle=True):
    filepath = DATA_DIR / 'processed/games_idx.pkl'
    if use_pickle:
        try:
            df = pd.read_pickle(filepath)
        except FileNotFoundError:
            df = make_team_idx(use_pickle=False)
    else:
        teams = all_teams()
        df = (make_raw()
              .merge(right=teams[['team', 'team_idx']], left_on='home_team', right_on='team')
              .drop('team', axis=1)
              .rename(columns={'team_idx': 'home_team_idx'})
              .merge(right=teams[['team', 'team_idx']], left_on='away_team', right_on='team')
              .drop('team', axis=1)
              .rename(columns={'team_idx': 'away_team_idx'})
              .assign(year=lambda df: df.date.dt.year - df.date.dt.year.min())
             )
        df.to_pickle(filepath)
    return df


def make_long(use_pickle=True):
    filepath = DATA_DIR / 'processed/long.pkl'
    if use_pickle:
        try:
            df_long = pd.read_pickle(filepath)
        except FileNotFoundError:
            df_long = make_long(use_pickle=False)
    else:
        df = make_raw()
        df_home = (df
               .rename(columns={'home_score': 'goals_scored', 'away_score': 'goals_admitted', 'away_team': 'opponent'})
               .pipe(lambda df: df.drop(labels = [col for col in list(df) if 'away' in col], axis=1))
               .rename(columns = lambda name: name.replace('home_', ''))
               .assign(home_field = lambda df: ~df.neutral.values)
               .sort_values(by=['team', 'date'], ascending=[True, True])
               .reset_index(drop=True)
              )
        df_away = (df
               .rename(columns={'away_score': 'goals_scored', 'home_score': 'goals_admitted', 'home_team': 'opponent'})
               .pipe(lambda df: df.drop(labels = [col for col in list(df) if 'home' in col], axis=1))
               .rename(columns = lambda name: name.replace('away_', ''))
               .assign(home_field = False)
               .sort_values(by=['team', 'date'], ascending=[True, True])
               .reset_index(drop=True)
              )

        df_long = (pd
               .concat([df_home, df_away])
               .sort_values(by=['team', 'date'], ascending=[True, True])
               .reset_index(drop=True)
              )

        df_long.to_pickle(filepath)
    return df_long


def all_teams(use_pickle=True):
    filepath = DATA_DIR / 'processed/all_teams.pkl'
    if use_pickle:
        try:
            teams = pd.read_pickle(filepath)
        except FileNotFoundError:
            teams = all_teams(use_pickle=False)
    else:
        df_long = make_long()
        teams = (df_long
             .assign(games=1)
             .groupby('team')
             [['games', 'win']]
             .sum()
             .reset_index()
             .sort_values(by='team', ascending=True)
             .reset_index()
             .rename(columns={'index': 'team_idx'})
            )
        teams.to_pickle(filepath)
    return teams


def make_groups(use_pickle=True):
    filepath = DATA_DIR / 'processed/groups.json'
    if use_pickle:
        try:
            groups = json.loads(filepath.read_text())
        except FileNotFoundError:
            groups = make_groups(use_pickle=False)
    else:
        fixture = pd.read_excel(DATA_DIR / 'processed/raw_fixture.xlsx')
        groups = (fixture
                  .append(
                      fixture
                      .drop(columns='home')
                      .rename(columns={'away': 'home'})
                      )
                  .loc[fixture.group.str.contains('Group'), ['group', 'home']]
                  .rename(columns={'home': 'team'})
                  .drop_duplicates()
                  .sort_values(by=['group', 'team'])
                  .reset_index(drop=True)
                  .apply(lambda col: col.str.replace('Group ', ''))
                  .groupby('group').agg(lambda vals: list(vals))
                  .to_dict()['team']
                 )
        filepath.write_text(json.dumps(groups))
    return groups



def make_all():
    make_raw()
    make_team_idx()
    all_teams()
