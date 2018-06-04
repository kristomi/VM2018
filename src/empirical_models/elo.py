import pandas as pd
import numpy as np
import math
import os
from pathlib import Path
import pickle
from sklearn.linear_model import LogisticRegression

from src.tournament import Outcome, stats
from src.make_data import make_team_idx, make_long

ROOT_DIR = Path(os.environ.get('ROOT_DIR'))
DATA_DIR = ROOT_DIR / 'data'


class TeamElo:
    """

    """
    def __init__(self, team):
        self.team = team
        self.elos = []
        self.last_elo = 1500.0

    def add_elo(self, date, change):
        self.elos.append({'team': self.team,
                          'date': date,
                          'elo': self.last_elo,
                          'elo_new': self.last_elo + change})
        self.last_elo += change

    @property
    def df(self):
        return pd.DataFrame(self.elos).sort_values(by='date', ascending=True)

    def plot(self):
        df = self.df
        ax = df.plot(x='date', y='elo')
        return ax


class Elo:
    def __init__(self):
        """
        Class for generating Elo scores and predicting game outcomes. Based off of
        - Fivethirtyeight: https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py
        - World Football Elo ratings: https://www.eloratings.net/about
        - This academic prediction paper: http://www.collective-behavior.com/publ/ELO.pdf
        Has to be trained first
        """
        self.pickle_path = DATA_DIR / 'models/elo_pickle.pkl'
        unique_teams = set(make_long().team.values)

        self.teams = {team: TeamElo(team) for team in unique_teams}
        self.HFA = 200.0    # Home field advantage
        self.K = 20.0       # The speed at which Elo ratings change
        self.elo_df = None

        self.lr = LogisticRegression()  # Logistic regression for predicting outcome of games

    def save(self):
        self.pickle_path.write_bytes(pickle.dumps(self))

    def train(self, df=None):
        "Generates win probabilities and estimates Elo scores for each country"
        if df is None:
            df = make_team_idx()

        games = (df
                 .sort_values(by='date',
                              ascending=True)
                 .to_dict(orient='index')
                 )

        for _, game in games.items():

            out = self.update(date=game['date'],
                              home_team=game['home_team'],
                              away_team=game['away_team'],
                              home_score=game['home_score'],
                              away_score=game['away_score'],
                              neutral=game['neutral']
                              )

            game['home_elo'], game['away_elo'], game['elo_home_win_prob'] = out['home_elo'], out['away_elo'], out['p_home']

        out = pd.DataFrame(games).T
        out['draw'] = 1-out['home_win'] - out['away_win']
        X = (out['home_elo'] - out['away_elo'] + self.HFA*(1-out['neutral'])).values
        y = out[['home_win', 'draw', 'away_win']].astype(int).values

        self.lr.fit(X.reshape(-1, 1), np.argwhere(y == 1)[:, 1])

        return out

    def __call__(self, home_team, away_team, date=None, can_draw=True):
        out = self.predict_update(home_team, away_team, can_draw=can_draw)
        return Outcome(home=home_team,
                       away=away_team,
                       home_goals=out['home_score'],
                       away_goals=out['away_score'],
                       date=date,
                       can_draw=can_draw,
                       to_overtime=out['to_overtime']
                       )

    def predict_update(self, home_team, away_team, date=None, neutral=True, can_draw=True):
        winner, to_overtime = self.pred(home_team, away_team, neutral, can_draw)
        if winner == home_team:
            home_score, away_score = 1, 0
        elif winner == away_team:
            home_score, away_score = 0, 1
        else:
            home_score, away_score = 0, 0

        out = self.update(date, home_team, away_team, home_score, away_score, neutral)
        return {**out, **{'home_score': home_score, 'away_score': away_score, 'to_overtime': to_overtime}}

    def update(self, date, home_team, away_team, home_score, away_score, neutral):
        home_elo, away_elo = self.teams[home_team].last_elo, self.teams[away_team].last_elo
        p_home = self._predict_from_elo(home_elo, away_elo, neutral)
        shift = self.elo_change(home_score - away_score, p_home)

        self.teams[home_team].add_elo(date, shift)
        self.teams[away_team].add_elo(date, -shift)

        return {'home_elo': home_elo, 'away_elo': away_elo, 'p_home': p_home}

    def pred(self, home_team, away_team, neutral=True, can_draw=True, to_overtime=False):
        home_elo, away_elo = self.teams[home_team].last_elo, self.teams[away_team].last_elo
        method = 'regression'
        if method == 'fivethirtyeight': # Bruker bare Elo - seier uansett
            p_home = 1.0 / (math.pow(10.0, (-(home_elo - away_elo)/400.0)) + 1.0)
            if np.random.random() < p_home:
                outcome = home_team
            else:
                outcome = away_team
        elif method == 'regression': # Logistisk regresjon, også uavgjort mulig
            ar = self.lr.predict_proba(home_elo - away_elo)
            outcome = [home_team, 'draw', away_team][np.argmax(np.random.random() < ar.cumsum())]

        if outcome == 'draw' and not can_draw:
            return self.pred(home_team, away_team, neutral, can_draw, to_overtime=True)
        return outcome, to_overtime

    @property
    def teams_df(self):
        if self.elo_df is None:
            teams = (pd
                     .DataFrame
                     .from_records([val for team in self.teams.values() for val in team.elos])
                     .sort_values(by=['team', 'date'], ascending=[True, True])
                     .reset_index(drop=True)
                     )
            dfs = []
            for country in teams.team.unique():
                dfs.append(teams
                           .loc[teams.team == country, ['elo', 'date']]
                           .set_index('date')
                           .rename(columns={'elo': country})
                           .resample('Q')
                           .mean()
                           )
            self.elo_df = pd.concat(dfs, axis=1).fillna(method='ffill')
        return self.elo_df

    def _predict_from_elo(self, home_elo, away_elo, neutral=True):
        """
        Given two teams, will predict probability for home win
        """
        elo_diff = home_elo - away_elo + (0 if neutral == 1 else self.HFA)

        p_home = 1.0 / (math.pow(10.0, (-elo_diff/400.0)) + 1.0)

        return p_home

    def elo_change(self, score_diff, estimated_p_home):
        if abs(score_diff) == 1:
            mult = 1
        elif abs(score_diff) == 2:
            mult = 1.5
        elif abs(score_diff) == 3:
            mult = 1.75
        else:
            mult = 1 + (abs(score_diff)-3)/8

        if score_diff == 0:
            result = .5
        elif score_diff > 0:
            result = 1
        else:
            result = 0

        # Elo shift based on K and the margin of victory multiplier
        shift = (self.K * mult) * (result - estimated_p_home)

        return shift
