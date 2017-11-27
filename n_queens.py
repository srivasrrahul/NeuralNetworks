
BOARD_SIZE = 30


def init_state(state):
    for i in range(0,BOARD_SIZE):
        state[i] = {}
        for j in range(0,BOARD_SIZE):
            state[i][j] = 0

def is_queen_present(t):
    return t == 1

def add_covered_state(state,row_index,col_index):
    #for horizontol
    for i in range(col_index+1,BOARD_SIZE):
        state[row_index][i] += 2

    for j in range(row_index+1,BOARD_SIZE):
        state[j][col_index] += 2

    #diagnal
    for i in range(1,BOARD_SIZE):
        if row_index + i < BOARD_SIZE and col_index + i < BOARD_SIZE:
            state[row_index + i][col_index + i] += 2

    for i in range(1,BOARD_SIZE):
        if row_index + i < BOARD_SIZE and col_index - i > -1:
            state[row_index + i][col_index - i] += 2


def removed_covered_state(state,row_index,col_index):
    #for horizontol
    for i in range(col_index+1,BOARD_SIZE):
        state[row_index][i] -= 2

    for j in range(row_index+1,BOARD_SIZE):
        state[j][col_index] -= 2

    #diagnal
    for i in range(0,BOARD_SIZE):
        if row_index + i < BOARD_SIZE and col_index + i < BOARD_SIZE:
            state[row_index + i][col_index + i] -= 2

    for i in range(0,BOARD_SIZE):
        if row_index + i < BOARD_SIZE and col_index - i > -1:
            state[row_index + i][col_index - i] -= 2

def goal_state(local_state,row_index,col_index):
    #print(str(row_index) + " "  + str(col_index))
    if local_state[row_index][col_index] == 0:
        return True
    else:
        return False



def put_queen(local_state,x,y):
    local_state[x][y] = 1
    add_covered_state(local_state,x,y)
    local_state[x][y] = 1
    return local_state


def remove_queen(local_state,x,y):
    removed_covered_state(local_state,x,y)
    local_state[x][y] = 0
    return local_state


def transition_model(state,row_index,removed_index = -1):
    smaller_range = 0
    if removed_index != -1:
        smaller_range = removed_index + 1
    for i in range(smaller_range,BOARD_SIZE):
        #print("Row is " + str(smaller_range) + " Col is " + str(i))
        if goal_state(state, row_index, i):
            put_queen(state,row_index,i)
            return (True,(row_index,i))
    return (False,(-1,-1))



def transition(state):
    old_state_lst = []
    i = 0
    last_backward_index = -1
    while i < BOARD_SIZE:
        print("Row is " + str(i))
        result,(row_index,col_index) = transition_model(state,row_index=i,removed_index=last_backward_index)
        if result:
            old_state_lst.append((row_index,col_index))
            print("Adding for row " + str(i))
            i += 1
            last_backward_index = -1
        else:
            print("Row is bad " + str(i))
            print_state_row(state,i)
            old_row,old_col = old_state_lst.pop()
            remove_queen(state,old_row,old_col)
            print_state_row(state,i)
            i -= 1
            last_backward_index = old_col


def print_state_row(state,row_index):
    for i in range(0,BOARD_SIZE):
        print state[row_index][i],
    return ""

def print_state(state):
    for i in range(0,BOARD_SIZE):
        for j in range(0,BOARD_SIZE):
            print state[i][j],
        print "\n"

def normalize_print_state(state):
    for i in range(0,BOARD_SIZE):
        for j in range(0,BOARD_SIZE):
            if state[i][j] != 1:
                print 0,
            else:
                print 1,
        print "\n"


state = {}
init_state(state)
print_state(state)
transition(state)
normalize_print_state(state)
#init_state(state)
#transition(state)