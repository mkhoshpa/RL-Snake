Python Implementation of Snake. This implementation comes with a DQN agent.
Interested in Knowing How it works? Check out here.

Usage:

#### Train:
    board_size = (15,15)
    dqn_agent = Agent.DQNAgent(board_size, path = False)


#### Load a pre-trained agent:
    dqn_agent = Agent.DQNAgent(board_size, path = PATH)


#### View GUI of the game
    run_gui_game(board_size,dqn_agent)