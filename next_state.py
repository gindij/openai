def frozen_lake_v0_next_state(state, action):
    if action == 0:
        if state % 4 == 0:
            return state
        return state - 1
    elif action == 1:
        if state % 4 == 3:
            return state
        return state + 1
    elif action == 2:
        if state >= 12:
            return state
        return state + 4
    else:
        if state < 4:
            return state
        return state - 4
