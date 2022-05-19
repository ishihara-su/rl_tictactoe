# tictactoe.py - Reinformcement learning-based Tic-Tac-Toe
#   Susumu Ishihara <ishihara.susumu@shizuoka.ac.jp>

from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

verbose = False


def error_exit(msg):
    print(f'Error: {msg}', file=sys.stderr)
    sys.exit(1)


class Board:
    WIN = 1
    DRAW = 0
    LOSE = -1
    CONTINUE = -2
    INVALIED = -3

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._cells = [0] * 9  # 0 is for a free cell
        self._count = 0

    @property
    def code(self) -> int:
        """Return the state code of the board

        :returns: state code (0 - 3**9-1)
        """
        s = 0
        for v in self._cells:
            s = s * 3 + v
        return s

    def is_occupied(self, pos) -> bool:
        return self._cells[pos] != 0

    def mark_and_judge(self, v: int, pos) -> int:
        """Mark and return true if it is a winner

        :parameter v: player (1: first player / 2: second)
        :parameter x: x
        :parameter y: y
        :returns: Board.{WIN/DRAW/CONTINUE}
        """
        if self._count >= 9:
            error_exit('The board is full')
        if v < 1 or 2 < v:
            raise ValueError(f'v must be 1 or 2.: Board.put() v={v}')
        x = pos % 3
        y = pos // 3
        if self._cells[pos] != 0:
            error_exit('Marked on a closed cell')
        self._count += 1
        self._cells[pos] = v
        if (self._cells[y * 3 + 0] == self._cells[y * 3 + 1] == self._cells[y * 3 + 2] == v):
            return Board.WIN
        if (self._cells[x] == self._cells[3 + x] == self._cells[6 + x] == v):
            return Board.WIN
        if pos == 0 or pos == 4 or pos == 8:
            if self._cells[0] == self._cells[4] == self._cells[8]:
                return Board.WIN
        if pos == 2 or pos == 4 or pos == 6:
            if self._cells[2] == self._cells[4] == self._cells[6]:
                return Board.WIN
        if self._count == 9:
            return Board.DRAW
        return Board.CONTINUE

    def show(self, show_pos: bool = False):
        print('+---+---+---+')
        for y in range(3):
            print('|', end='')
            for x in range(3):
                pos = y * 3 + x
                v = self._cells[pos]
                if v == 0:
                    s = str(pos) if show_pos else ' '
                elif v == 1:
                    s = 'o'
                elif v == 2:
                    s = 'x'
                else:
                    raise ValueError(f'Undefined cell value: v: {v}')
                print(f' {s} |', end='')
            print()
            print('+---+---+---+')


class GameRecord:
    def __init__(self):
        self._wins = 0.0
        self._games = 0

    def add(self, win: int):
        self._games += 1
        if win == Board.WIN:
            self._wins += 1.0
        elif win == Board.DRAW:
            self._wins += 0.2
        elif win == Board.LOSE:
            pass
        else:
            raise ValueError(f'Invalid value - win: {win}')

    @property
    def games(self) -> int:
        return self._games

    @property
    def wins(self) -> int:
        return self._wins

    @property
    def winstr(self) -> str:
        return f'{self.wins:.2f}/{self._games}'

    @property
    def wp(self) -> float:
        if self.games == 0:
            return 0.0
        return self._wins/self._games

    @property
    def qv(self) -> float:
        if self.games == 0:
            return '0.0000'
        rate = self._wins/self._games * (1.0 - math.exp(- 0.0001 * self.games))
        return f'{rate:.4f}'


class Agent:
    def __init__(self, name: str = ''):
        if name == '':
            self.name = self.__repr__().split()[0].split('.')[-1]
        else:
            self.name = name
        self.learning = True
        self.init_stat()

    def init_stat(self):
        self.games = 0
        self.wins = 0
        self.draws = 0
        self.wp = []  # Winning percentage

    def mark(self, board: Board) -> int:
        error_exit('Not implemented: This class (Agent) must be inhereted')

    def record_game_result(self, result: int) -> None:
        self.games += 1
        if result == Board.WIN:
            self.wins += 1
        elif result == Board.DRAW:
            self.draws += 1
        elif result == Board.LOSE:
            pass
        else:
            raise ValueError('Invalid result value')
        self.wp.append(self.wins/self.games)

    def start_game(self, turn: int, learning: bool = True) -> None:
        """starts game

        :parameter turn: 1: first, 2: second
        :parameter learning: Is learning mode?
        """
        if turn < 1 or 2 < turn:
            raise ValueError('turn must be 1 or 2.')
        self.turn = turn
        self.learning = learning

    def show_stat(self) -> None:
        print(
            f'{self.name:12s} WP: {self.wp[-1]:.3f} Win: {self.wins:5d} Draw: {self.draws:5d} Games: {self.games}')


class RLAgent(Agent):
    def __init__(self, name: str = ''):
        super().__init__(name)
        self.records = [[[GameRecord() for i in range(9)]
                         for j in range(3**9)] for k in range(2)]

    def start_game(self, turn: int, learning=True):
        super().start_game(turn, learning)
        self.history = []

    def mark(self, board: Board) -> int:
        DEFAULT_MIN_GAMES = 100
        MIN_SUM_RATE = 0.001
        records = self.records[self.turn-1][board.code]
        best_way = -1
        min_game_way = -1
        best_rate = -0.001
        min_games = DEFAULT_MIN_GAMES
        candidates = []
        sum_rate = 0.0
        for i in range(len(records)):
            if board.is_occupied(i):
                continue
            wins, games = records[i].wins, records[i].games
            if games <= min_games:
                min_game_way, min_games = i, games
            rate = 0.0 if games == 0 else wins/games
            if rate > best_rate:
                best_way, best_rate = i, rate
            if verbose:
                print(f'{i}: {records[i].wp:.3f}, ', end='')
            candidates.append((i, rate))
            sum_rate += rate
        if self.learning and min_games < DEFAULT_MIN_GAMES:
            best_way = min_game_way
        elif sum_rate < MIN_SUM_RATE:
            (best_way, _) = random.choice(candidates)
        elif self.learning:
            thresholds = []
            cum_r = 0.0
            for (c, r) in candidates:
                cum_r += r
                thresholds.append((c, cum_r/sum_rate))
            r = random.random()
            for (c, thresh) in thresholds:
                if r < thresh:
                    best_way = c
                    break
            assert r < thresh
        if verbose:
            print(f' Selected: {best_way}')
        if best_way < 0:
            error_exit('No valid hand is found.')
        self.history.append((board.code, best_way))
        return board.mark_and_judge(self.turn, best_way)

    def record_game_result(self, result: int) -> None:
        super().record_game_result(result)
        for (code, pos) in self.history:
            self.records[self.turn-1][code][pos].add(result)


class RandomAgent(Agent):
    def __init__(self, name: str = '') -> None:
        super().__init__(name)

    def mark(self, board: Board) -> int:
        candidates = []
        for i in range(9):
            if board.is_occupied(i):
                continue
            candidates.append(i)
        if len(candidates) == 0:
            error_exit('No valid hand is found.')
        c = random.choice(candidates)
        return board.mark_and_judge(self.turn, c)


class HumanAgent(Agent):
    def __init__(self, name: str = '') -> None:
        super().__init__(name)

    def mark(self, board: Board) -> int:
        candidates = []
        for i in range(9):
            if board.is_occupied(i):
                continue
            candidates.append(i)
        if len(candidates) == 0:
            error_exit('No valid hand is found.')
        board.show(show_pos=True)
        while True:
            try:
                c = int(input(f'Input one of {candidates}: '))
            except ValueError:
                print('Invalid input', file=sys.stderr)
                continue
            if c not in candidates:
                print('Invalid input', file=sys.stderr)
                continue
            break
        return board.mark_and_judge(self.turn, c)


def play_games(agents: list[Agent], n_trials: int = 100, learning=False) -> None:
    for i in range(n_trials):
        first = random.randrange(2)
        step = 0
        board = Board()
        agents[0].start_game(1 + first, learning)
        agents[1].start_game(1 + (first+1) % 2, learning)
        while True:
            ego = (step + first) % 2
            r = agents[ego].mark(board)
            if verbose:
                board.show()
            if r != Board.CONTINUE:
                assert r == Board.WIN or r == Board.DRAW
                agents[ego].record_game_result(r)
                other = (ego + 1) % 2
                agents[other].record_game_result(-r)
                winner_char = 'o' if agents[ego].turn == 1 else 'x'
                # print(f'            Winner:  {agents[ego].name} ({winner_char})')
                break
            step += 1
        if verbose:
            for a in agents:
                a.show_stat()


if __name__ == '__main__':
    n_trials = 100000
    n_battles = 500
    random.seed(1)
    if len(sys.argv) >= 2:
        n_trials = int(sys.argv[1])
    if len(sys.argv) >= 3:
        n_battles = int(sys.argv[2])
    verbose = False
    rl1 = RLAgent('RL1')
    rl2 = RLAgent('RL2')
    ra = RandomAgent('Rand')
    agents = [rl1, rl2]
    play_games(agents, n_trials, learning=True)

    fig, (ax1, ax2) = plt.subplots(2)
    x1 = np.arange(len(agents[0].wp))
    for a in agents:
        ax1.plot(x1, a.wp, label=a.name)
    ax1.legend()
    ax1.set_ylabel('Winning Percentage')
    ax1.set_xlabel('# Trials')

    #verbose = True
    ha = HumanAgent('You')
    rl1.init_stat()
    ra.init_stat()
    #agents = [rl1, ha]
    agents = [rl1, ra]
    play_games(agents, n_battles, learning=False)

    x2 = np.arange(len(agents[0].wp))
    for a in agents:
        ax2.plot(x2, a.wp, label=a.name)
    ax2.legend()
    ax2.set_ylabel('Winning Percentage')
    ax2.set_xlabel('# Battles')
    plt.show()
