# tictactoe.py - Reinformcement learning-based Tic-Tac-Toe
#   Susumu Ishihara <ishihara.susumu@shizuoka.ac.jp>

from enum import Enum
import math
import random
import sys

def error_exit(msg):
    print(f'Error: {msg}', file=sys.stderr)
    sys.exit(1)

class GameResult(Enum):
    WIN = 1
    DRAW = 2
    CONTINUE = 0

class Board:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._cells = [0] * 9
        self._count = 0

    @property
    def code(self) -> int:
        """Return the state code of the board

        :returns: state code (0 - 3**9-1)
        """
        s = 0
        for v in self.cells:
            s = s * 3 + v
        return s

    def state(self, pos) -> int:
        return self._cells[pos]

    def mark_and_judge(self, v: int, pos) -> int:
        return self.mark_and_judge(v, pos % 3, pos / 3)

    def mark_and_judge(self, v: int, x: int, y:int) -> int:
        """Mark and return true if it is a winner

        :parameter v: player (1 or 2)
        :parameter x: x
        :parameter y: y
        :returns: 1 win / 0 / -1 not allowed
        """
        if self._count >= 9:
            error_exit('The board is full')
        if v < 1 or 2 < v:
            raise ValueError(f'v must be 0 or 1.: Board.put() v={v}')
        pos = y * 3 + x
        if self._cells[pos] != 0:
            error_exit('Marked on a closed cell')
        self._count += 1
        self._cells[pos] = v
        if (self._cells[y * 3 + 0] == self._cells[y * 3 + 1] == self._cells[y * 3 + 2] == v):
            return GameResult.WIN
        if (self._cells[x] == self._cells[3 + x] == self._cells[6 + x] == v):
            return GameResult.WIN
        if pos == 0 or pos == 4 or pos == 8:
            if self._cells[0] == self._cells[4] == self._cells[8]:
                return GameResult.WIN
        if pos == 2 or pos == 4 or pos == 6:
            if self._cells[2] == self._cells[4] == self._cells[6]:
                return GameResult.WIN
        if self._count == 9:
            return GameResult.DRAW
        return GameResult.CONTINUE

class Agent:
    def __init__(self, turn):
        if turn < 1 or 2 < turn:
            raise ValueError('turn must be 1 or 2')
        self.turn = turn # 1 - first / 2 second

    def mark(self, board: Board) -> int:
        error_exit('Not implemented: This class (Agent) must be inhereted')

    def record_game_result(self, result: GameResult) -> None:
        pass


class GameRecord:
    def __init__(self):
        self._wins = 0
        self._games = 0

    def add(self, win: GameResult):
        self._games += 1
        if win == GameResult.INVALID_HAND:
            self._wins = -1000
        elif win == GameResult.WIN:
            self._wins += win

    @property
    def games(self) -> int:
        return self._games

    @property
    def wins(self) -> int:
        return self._wins

class RLAgent(Agent):
    def __init__(self, turn: int, stabilizing_rate = 0.1):
        super().__init__(turn)
        self.results = [[GameRecord() for i in range(9)] for j in 3**9]
        self.stabilizing_rate = stabilizing_rate

    def start_game(self):
        self.history = []

    def mark(self, board: Board) -> int:
        results = self.results[board.code]
        best_way = -1
        best_rate = -0.001
        for i in len(results):
            if board.state(i) != 0:
                continue
            wins, games = results[i].wins, results[i].games
            rate = wins/games * (1.0 - math.exp(- self.alpha * games))
            if rate > best_rate:
                best_way = i
        if best_way < 0.0:
            error_exit('No valid hand is found.')
        self.history.append(board.code)
        return board.mark_and_judge(best_way)

    def record_game_result(self, result: GameResult) -> None:
        for code in self.history:
            self.results[code].add(result)


class RLAgent(Agent):
    def __init__(self, turn: int, stabilizing_rate = 0.1):
        super().__init__(turn)

    def mark(self, board: Board) -> int:
        candidates = []
        for i in range(9):
           if board.state(i) == 0:
               candidates.append(i)
        if len(candidates) == 0:
            error_exit('No valid hand is found.')
        return random.choice(candidates)
