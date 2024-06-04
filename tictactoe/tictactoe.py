from copy import deepcopy
"""
Tic Tac Toe Player
"""

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count = 0
    for row in board:
        for value in row:
            if value == EMPTY:
                count += 1
    if count % 2 == 0:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    turn = player(board)
    new_board = deepcopy(board)
    if new_board[action[0]][action[1]] == EMPTY:
        new_board[action[0]][action[1]] = turn
    else:
        raise Exception
    
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for row in board:
        if len(set(row)) == 1 and row[0] != EMPTY:
            return row[0]

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != EMPTY:
            return board[0][col]

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != EMPTY:
        return board[0][2]

    return None



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    count = 0
    for row in board:
        for value in row:
            if value == EMPTY:
                count += 1
    if count == 0 or winner(board) != EMPTY:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def eval(board, depth, isMaxing):
    if terminal(board):
        win = utility(board)
        return win
    if isMaxing:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j] = X
                    new_value = eval(board, depth + 1, False)
                    board[i][j] = EMPTY
                    max_eval = max(max_eval, new_value)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                        board[i][j] = O
                        new_value = eval(board, depth + 1, True)
                        board[i][j] = EMPTY
                        min_eval = min(min_eval, new_value)
        return min_eval


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    turn = player(board)
    possible_moves = actions(board)
    best_move = None
    if turn == X:
        best_val = float('-inf')
        for move in possible_moves:
            new_board = result(board, move)
            val = eval(new_board, 0, False)
            if val > best_val:
                best_val = val
                best_move = move
    else:
        best_val = float('inf')
        for move in possible_moves:
            new_board = result(board, move)
            val = eval(new_board, 0, True)
            if val < best_val:
                best_val = val
                best_move = move
    return best_move
                