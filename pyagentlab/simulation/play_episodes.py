"""
pyagentlab/simulation/play_episodes.py
---
this file defines functions to run episodes of the custom defined-environment.
 
"""


from collections import deque
from pyagentlab.constants import CONST
from pyagentlab.environment.outcome import OUTCOME


def play_episodes(N_EPISODES, env, players, is_training, history_offset=0):
    player_1_outcomes = deque(maxlen=100)
    for n_episodes in range(N_EPISODES):
        scores, prev_state_str = play_episode(env, players, is_training)
        outcomes = env.return_outcomes()
        ranks = env.return_win_ranks()

        env.append_score_and_outcome_history(scores)

        # local variable is appended to for quick logging.
        player_1_outcomes.append(outcomes[0])

        # prints out log information every 1000 episodes.
        if n_episodes > 0 and (n_episodes + 1) % 1000 == 0:
            print("\n\n\n" + prev_state_str)
            print(
                f"episode #{n_episodes + 1:>7d}\t\t"
                f"avg loss: {players[0].calc_avg_loss():.5f}\t\t"
                f"lr: {players[0].get_current_lr():.1e}\t\t"
                f"epsilon: {players[0]._epsilon:.5f}"
            )
            print("outcome:   ", end="")

            for outcome in env.return_outcomes():
                status = ""
                if outcome == OUTCOME.WIN:
                    status = "win"
                elif outcome == OUTCOME.DRAW:
                    status = "draw"
                elif outcome == OUTCOME.LOSS:
                    status = "loss"
                elif outcome == OUTCOME.INTERRUPTED:
                    status = "interrupted"
                elif outcome == OUTCOME.FORFEIT_BY_ILLEGAL:
                    status = "illegal"
                print(status.ljust(8) + " ", end="")
            print("\n", end="")

            # prints average amount of wins/losses/draws for player 1.
            num_wins = 0
            num_draws = 0
            num_losses = 0
            for outcome in player_1_outcomes:
                if outcome == OUTCOME.WIN:
                    num_wins += 1
                elif outcome == OUTCOME.DRAW:
                    num_draws += 1
                elif outcome == OUTCOME.LOSS:
                    num_losses += 1
            avg_wins = int((num_wins / len(player_1_outcomes)) * 100)
            avg_draws = int((num_draws / len(player_1_outcomes)) * 100)
            avg_losses = int((num_losses / len(player_1_outcomes)) * 100)
            print(f"W/L/D:     {avg_wins:>2d}/{avg_losses:>2d}/{avg_draws:>2d}")


def play_episode(env, players, is_training):
    prev_state, info = env.reset()

    while not env.entirely_done():
        player_num = env.get_player_num()
        player = players[player_num - 1]

        # creates the convolutional and additional FC observations.
        prev_conv_obs = prev_state.to_conv_obs(player_num)
        prev_add_fc_obs = prev_state.to_add_fc_obs(player_num)

        # the current player chooses an action.
        action = player.choose_action(
            prev_state, prev_conv_obs, prev_add_fc_obs, player_num
        )

        # the environment has an action applied and steps forward.
        next_state, reward, done, trunc, info, legal = env.step(action)

        # the player saves the transition to its memory.
        player.process_step_and_learn(
            player_num,
            prev_state,
            prev_conv_obs,
            prev_add_fc_obs,
            action,
            reward,
            done,
            legal,
        )

        # the previous state becomes the next state.
        prev_state = next_state

    # finalizes scores.
    scores = [0.0 for _ in range(CONST.N_PLAYERS)]
    for i, player in enumerate(players):
        player_scores = player.finalize_episodes(
            env.return_outcomes(), env.return_win_ranks()
        )
        player.reset()
        for i, player_score in enumerate(player_scores):
            if player_score is not None:
                scores[i] = player_score

    return scores, prev_state.to_str()
