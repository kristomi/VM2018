# Module to hold the tournament stuff
import os
import requests
import re
import json
import itertools as it
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import collections
import menon_styles
from pathlib import Path
from multiprocessing import cpu_count, Pool

from src.make_data import make_groups


ROOT_DIR = Path(os.environ.get('ROOT_DIR'))
DATA_DIR = ROOT_DIR / 'data'


def get_fixture():
    df = pd.read_excel(DATA_DIR / 'processed/current_fixture.xlsx')
    return df


def scrape_fixture():
    r = requests.get('http://www.skysports.com/football/news/12098/11154890/world-cup-fixtures-the-full-schedule-for-russia-2018')
    soup = BeautifulSoup(r.text, 'lxml')
    outer = soup.find("div", {"class": "article__body article__body--lead"})
    games = [p.text for p in outer.findAll("p") if ' v ' in p.text]
    def get_info(game):
        info = {}
        game = game.replace('Tues', 'Tue')#.replace('South Korea', 'Korea')
        date, rest = game.split(':')
        teams, rest = rest.strip().split(' - ', maxsplit=1)
        home, teams_rest = teams.split(' v ')
        away = teams_rest.split(' (')[0]
        try:
            group = re.search(r'\((.*)\)', teams).group(1)
        except AttributeError:
            try:
                group = re.findall(r'\((.*)\)', game)[-1]
            except IndexError:
                group = ''
        return dict(date=pd.to_datetime(date + ' 2018').date(), home=home, away=away, group=group)

    games_df = pd.DataFrame([get_info(game) for game in games])
    games_df.loc[games_df['group'] == 'Luzhniki), 3pm (Match 51', 'group'] = 'Match 51'
    games_df.loc[games_df['group'] == 'Spartak), 7pm (Match 56', 'group'] = 'Match 56'
    games_df.loc[60, 'group'] = 'Match 61'
    games_df.loc[61, 'group'] = 'Match 62'
    finals = [{'home': 'Loser match 61', 'away': 'Loser match 62', 'date': pd.to_datetime('Sat July 14 2018').date(), 'group': 'Match 63'},
              {'home': 'Winner match 61', 'away': 'Winner match 62', 'date': pd.to_datetime('Sun July 15 2018').date(), 'group': 'Match 64'}]
    games_df = pd.concat([games_df, pd.DataFrame(finals)], ignore_index=True)
    games_df['match'] = games_df.index + 1
    games_df.loc[games_df['group'].str.contains('Match'), 'group'] = 'Playoffs'
    games_df['home_score'] = None
    games_df['away_score'] = None
    games_df.to_pickle(ROOT_DIR / 'data/raw/raw_fixture.pkl')
    games_df.to_excel(ROOT_DIR / 'data/processed/raw_fixture.xlsx')

    return games_df


stats = collections.namedtuple('stats', 'team goals_scored goals_admitted points')


class Outcome:
    """
    Stores game outcomes for further processing
    """
    def __init__(self, home, away, home_goals, away_goals, date=None, can_draw=True, to_overtime=False):
        self.home = home
        self.away = away
        self.home_goals = home_goals
        self.away_goals = away_goals
        self.date = date
        self.can_draw = can_draw
        self.to_overtime = to_overtime

        if home_goals > away_goals:
            self.winner, self.loser = self.home, self.away
        elif away_goals > home_goals:
            self.winner, self.loser = self.away, self.home
        else:
            self.winner = None


    def __repr__(self):
        return f"Outcome(home={self.home}, away={self.away}, home_goals={self.home_goals}, away_goals={self.away_goals}, can_draw={self.can_draw}, to_overtime={self.to_overtime})"

    def __str__(self):
        return f"{self.home} {self.home_goals} - {self.away_goals} {self.away}"

    @property
    def stats(self):
        winner = self.winner
        return {self.home: self.home_goals, self.away: self.away_goals, 'stats': {'winner': winner, 'date': self.date, 'can_draw': self.can_draw, 'to_overtime': self.to_overtime}}


    @staticmethod
    def _score_points(own_goals, other_goals):
        if own_goals > other_goals:
            return 3
        elif own_goals < other_goals:
            return 0
        return 1

    @property
    def teams(self):
        return [self.home, self.away]

    @property
    def home_stats(self):
        return stats(self.home, self.home_goals, self.away_goals, self._score_points(self.home_goals, self.away_goals))

    @property
    def away_stats(self):
        return stats(self.away, self.away_goals, self.home_goals, self._score_points(self.away_goals, self.home_goals))


class WorldCup:
    """
    Simulates the world cup. Requires a match instance which with its call
    method precicts the winner of a match.
    """
    def __init__(self, match):
        self.fixture = pd.read_excel(DATA_DIR / 'processed/current_fixture.xlsx')

        grouping = make_groups()
        self.team_list = [team for teams in grouping.values() for team in teams]
        self.match = match
        self.group_play()
        self.playoff()

    def group_play(self):
        groups = pd.DataFrame(json.loads((DATA_DIR/'processed/groups.json').read_text())).melt().rename(columns={'variable': 'group', 'value': 'team'})
        results = []
        for game in self.fixture.loc[self.fixture.group.str.contains('Group')].to_dict(orient='index').values():
            if np.isnan(game['home_score']): # Not yet played
                g = self.match(game['home'], game['away'], date=game['date'], can_draw=True)
            else: # Game is played, just recording the outcome
                g = Outcome(game['home'], game['away'], game['home_score'], game['away_score'])
            results.append(g.home_stats)
            results.append(g.away_stats)
        results = (pd
                   .DataFrame(results)
                   .assign(goals_diff = lambda df: df.goals_scored - df.goals_admitted)
                   .merge(right=groups, on='team')
                   .groupby(['group', 'team'])
                   .sum()
                   .reset_index()
                   .sort_values(by=['group', 'points', 'goals_scored', 'goals_diff'], ascending=[True, False, False, False])
                   .assign(rank = list(it.chain.from_iterable([[1, 2, 3, 4]*8])))
                  )

        self.group_outcome = results

    def adv(self, key):
        poss_keys = [''.join([el[0], el[1]]) for el in it.product(list('ABCDEFGH'), list('12'))]
        assert key in poss_keys, "key must be a string with group and placement, like 'A1'"
        group, place = key[0], int(key[1])
        return self.group_outcome.query("group == @group and rank == @place").team.values[0]

    def fix_or_new(self, game_num, home, away):
        """
        Helper method to either read game from fixture (if played) or to simulated
        """
        game = self.fixture.to_dict(orient='index')[game_num-1]
        if np.isnan(game['home_score']): # Not yet played
            return self.match(home, away, can_draw=False)
        else:
            Outcome(game['home'], game['away'], game['home_score'], game['away_score'])

    def playoff(self):
        playoffs = {}
        # 1/8 finals
        playoffs[49] = self.match(self.adv('C1'), self.adv('D2'), can_draw=False)
        playoffs[50] = self.match(self.adv('A1'), self.adv('B2'), can_draw=False)
        playoffs[51] = self.match(self.adv('B1'), self.adv('A2'), can_draw=False)
        playoffs[52] = self.match(self.adv('D1'), self.adv('C2'), can_draw=False)
        playoffs[53] = self.match(self.adv('E1'), self.adv('F2'), can_draw=False)
        playoffs[54] = self.match(self.adv('G1'), self.adv('H2'), can_draw=False)
        playoffs[55] = self.match(self.adv('F1'), self.adv('E2'), can_draw=False)
        playoffs[56] = self.match(self.adv('H1'), self.adv('G2'), can_draw=False)

        # 1/4 finals
        playoffs[57] = self.fix_or_new(game_num=57, home=playoffs[50].winner, away=playoffs[49].winner)
        playoffs[58] = self.fix_or_new(game_num=58, home=playoffs[53].winner, away=playoffs[54].winner)
        playoffs[59] = self.fix_or_new(game_num=59, home=playoffs[55].winner, away=playoffs[56].winner)
        playoffs[60] = self.fix_or_new(game_num=60, home=playoffs[51].winner, away=playoffs[52].winner)

        # Semi finals
        playoffs[61] = self.fix_or_new(game_num=61, home=playoffs[58].winner, away=playoffs[57].winner)
        playoffs[62] = self.fix_or_new(game_num=62, home=playoffs[60].winner, away=playoffs[59].winner)

        # Bronze final
        playoffs[63] = self.fix_or_new(game_num=63, home=playoffs[61].loser, away=playoffs[62].loser)

        # Final
        playoffs[64] = self.fix_or_new(game_num=64, home=playoffs[61].winner, away=playoffs[62].winner)

        self.playoffs = playoffs

        self.winner = playoffs[64].winner

    def get_placement(self, team):
        if team == self.playoffs[64].winner:
            return 1
        elif team == self.playoffs[64].loser:
            return 2
        elif team == self.playoffs[63].winner:
            return 3
        elif team == self.playoffs[63].loser:
            return 4
        elif team in list(it.chain.from_iterable([self.playoffs[x].teams for x in [57, 58, 59, 60]])):
            return 8
        elif team in list(it.chain.from_iterable([self.playoffs[x].teams for x in [49, 50, 51, 52, 53, 54, 55, 56]])):
            return 16
        else:
            return 24

    @property
    def stats(self):
        outcome = [(self.get_placement(team), team) for team in self.team_list]
        outcome = pd.DataFrame(outcome, columns=['rank', 'team']).sort_values(by='rank').set_index('team', drop=True)
        return outcome

def data_fun(x): return WorldCup(x).stats.T

class Simulation:
    """
    Class for simulating the World Cup 2018 with different parameters
    """
    def __init__(self, predictors: list, n: int = 1, multi_cores: bool = True):
        """
        Runs n simulations for each set of parameters in pars, and returns the aggregate statistics
        """
        self.predictors = predictors
        self.n = n
        self.multi_cores = multi_cores
        self.sim = self.simulate()

    def simulate(self):
        groups = make_groups()

        if not self.multi_cores:
            out = pd.concat([WorldCup(pred).stats.T for pred in self.predictors for _ in range(self.n)], axis=0).reset_index(drop=True)
        else:
            pool = Pool(cpu_count())
            data = [pred for pred in self.predictors for _ in range(self.n)]
            out = pd.concat(pool.map(data_fun, data))
        return (pd
                 .concat([
                     out.describe().T.sort_values(by='50%'),
                     out.agg(lambda col: np.mean(col==1)).rename('share wins'),
                     out.agg(lambda col: np.mean(col<=4)).rename('share top 4'),
                     out.agg(lambda col: np.mean(col<24)).rename('share playoff')
                         ], axis=1)
                 .drop('count', axis='columns')
                 .merge(right=pd.DataFrame([(group, country) for group in groups.keys() for country in groups[group]], columns = ['group', 'country']),
                        left_index=True,
                        right_on='country'
                       )
                 .sort_values(by='share wins', ascending=False)
                 .reset_index(drop=True)
                 [['country', 'group', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'share wins', 'share top 4', 'share playoff']]
                )

    @property
    def group(self):
        colors = []
        for _, color in menon_styles.menon_farger.items():
            colors.append(tuple([int(255*el) for el in color]))

        return (self.sim
                .sort_values(by=['group', 'share playoff'], ascending=[True, False])
                .reset_index(drop=True)
                .style
                 .apply(lambda row: [f"background-color: {'#%02x%02x%02x'%colors[(ord(row.group)-1)%len(colors)]}"]*len(row), axis=1)
                 .format({'share wins': "{:.0%}", 'share top 4': "{:.0%}", 'share playoff': "{:.0%}"})
               )

    @property
    def style(self):
        return (self.sim
                .style
                .bar(subset=['share wins', 'share top 4', 'share playoff'], align='left', color=['#5fba7d'])
                .format({'share wins': "{:.0%}", 'share top 4': "{:.0%}", 'share playoff': "{:.0%}"})
                .set_properties(**{'border-color': 'black'})
               )
