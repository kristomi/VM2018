import json
import os
import pickle
import pandas as pd
import numpy as np
import pymc3 as pm
import collections
from pathlib import Path

from src.tournament import Outcome, stats

ROOT_DIR = Path(os.environ.get('ROOT_DIR'))
DATA_DIR = ROOT_DIR / 'data'



class Team:
    __slots__ = ['name', 'atts', 'defs', '_hash']

    def __init__(self, name: str, atts: float = 0, defs: float = 0):
        self.name = name
        self.atts = atts
        self.defs = defs
        self._hash = hash((self.name, self.atts, self.defs))

    def __hash__(self):
        return self._hash

    def __str__(self):
        return f"{self.name} - atts: {self.atts}, defs: {self.defs}"

    def __repr__(self):
        return f"Team('{self.name}', {self.atts}, {self.defs})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        else:
            return False


class ConstantPar:
    def __init__(self, method='random', russian_home_advantage=True):
        self.team_list, self.intercept, self.home_advantage = self.get_pars(method=method, seed=None, russian_home_advantage=russian_home_advantage)

    @staticmethod
    def get_team_idx(team_name):
        teams = pd.read_pickle(DATA_DIR / 'processed/all_teams.pkl')
        return teams.loc[teams['team'] == team_name, 'team_idx'].values[0]

    def get_pars(self, method='random', seed=None, russian_home_advantage=True):
        pars = json.loads(Path(DATA_DIR / 'processed/pymc_pars.json').read_text())

        all_teams = pd.read_pickle(DATA_DIR / 'processed/all_teams.pkl').team.values

        if method == 'random':
            idx = str(np.random.choice(list(range(100))))
        elif method == 'median':
            idx = str(50)
        else:
            raise KeyError("Method has to be either 'random' or 'median'")

        intercept = pars['intercept'][idx]
        home_advantage = pars['home_advantage'][idx]
        atts = pars['atts'][idx]
        defs = pars['defs'][idx]

        team_list = {}
        for team in all_teams:
            atts = pars['atts'][idx][self.get_team_idx(team)]
            defs = pars['defs'][idx][self.get_team_idx(team)]
            if (team == 'Russia') and russian_home_advantage:
                atts += home_advantage
            team_list[team] = Team(name=team, atts=atts, defs=defs)
        return team_list, intercept, home_advantage

    def __call__(self, home_team, away_team, date=None, can_draw=True):
        home = self.team_list[home_team]
        away = self.team_list[away_team]

        home_theta = np.exp(self.intercept + home.atts + away.defs + self.home_advantage )
        away_theta = np.exp(self.intercept + away.atts + home.defs)

        # Ordinary time
        home_goals = np.random.poisson(home_theta)
        away_goals = np.random.poisson(away_theta)

        if (home_goals == away_goals) and (not can_draw): # Assumes golden goal
            to_overtime = True
            first_home_goal = np.random.exponential(1/home_theta)
            first_away_goal = np.random.exponential(1/away_theta)
            if first_home_goal < first_away_goal:
                home_goals += 1
            else:
                away_goals += 1
        else:
            to_overtime = False

        return Outcome(home=home_team,
                       away=away_team,
                       home_goals=home_goals,
                       away_goals=away_goals,
                       date=date,
                       can_draw=can_draw,
                       to_overtime=to_overtime)
