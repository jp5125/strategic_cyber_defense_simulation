# scripts/train_qlearn.py
"""
Train + evaluate the tabular Q-learner policy (qlearn_v1).

Run from repo root:

macOS/Linux:
  PYTHONPATH=src python scripts/train_qlearn.py

Windows PowerShell:
  $env:PYTHONPATH="src"
  python scripts\train_qlearn.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cyber_sim.parameters import default_parameters, apply_defaults
from cyber_sim.sim import run_sim, run_one
from cyber_sim.state import make_initial_state
from cyber_sim.rl import QLearner
from cyber_sim.metrics import summarize_run, rolling_action_freq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=100_000)
    parser.add_argument("--eval_steps", type=int, default=25_000)
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--eval_seed", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=None)  
    parser.add_argument("--p_attack_low", type=float, default=0.10)
    parser.add_argument("--p_attack_high", type=float, default=0.60)
    parser.add_argument("--print_action_mix", action="store_true")
    args = parser.parse_args()

    #Parameter values to be used during test execution
    P = apply_defaults(default_parameters())
    if args.epsilon is not None:
        P["rl_epsilon"] = float(args.epsilon)

    #Create an instance of the QLearner agent
    agent = QLearner(n_actions=3)

   #Initial training run
    train_df = run_one(
        P,
        seed=args.train_seed,
        policy="qlearn_v1",
        T=args.train_steps,
        agent=agent,
        learn=1,
        epsilon=float(P["rl_epsilon"]),
    )
    train_summary = summarize_run(train_df)

    #Evaluation runs
    eval_q_df = run_one(
        P,
        seed=args.eval_seed,
        policy="qlearn_v1",
        T=args.eval_steps,
        agent=agent,
        learn=0,
        epsilon=0.0,  # greedy eval
    )

    eval_thr_df = run_one(
        P, seed=args.eval_seed, policy="threshold_v1", T=args.eval_steps, agent=None
    )
    eval_pas_df = run_one(
        P, seed=args.eval_seed, policy="always_passive", T=args.eval_steps, agent=None
    )
    eval_rnd_df = run_one(
        P, seed=args.eval_seed, policy="random", T=args.eval_steps, agent=None
    )

    eval_summary = {
        "qlearn_greedy": summarize_run(eval_q_df),
        "threshold_v1": summarize_run(eval_thr_df),
        "always_passive": summarize_run(eval_pas_df),
        "random": summarize_run(eval_rnd_df),
    }

    print("\nTraining Summary:")
    print(json.dumps(train_summary, indent=2, sort_keys=True))

    print("\nEvaluation Summary:")
    print(json.dumps(eval_summary, indent=2, sort_keys=True))

    #Additional diagnostics
    if args.print_action_mix:
        train_action_mix = rolling_action_freq(train_df, window=500)
        eval_action_mix_q = rolling_action_freq(eval_q_df, window=250)

        print("\nQ size end (train):", train_summary.get("q_size_end", None))
        print("\nTrain action mix by window (head):")
        print(train_action_mix.head())

        print("\nEval action mix (qlearn greedy) by window (head):")
        print(eval_action_mix_q.head())

    #Threat Sensitivity Analysis, compares QLearn vs. threshold_v1 vs. random policy
    def eval_qlearn_under(p_attack: float, seed: int = 123) -> tuple[dict, dict]:
        P2 = P.copy()
        P2["defender_policy"] = "qlearn_v1"
        P2["rl_learn"] = 0
        P2["rl_epsilon"] = 0.0
        P2["p_attack"] = float(p_attack)
        P2["Seed"] = int(seed)

        rng = np.random.default_rng(int(P2["Seed"]))
        df = run_sim(P2, make_initial_state(P2), rng, agent=agent)
        return summarize_run(df), df["action_name"].value_counts(normalize=True).to_dict()

    low_sum, low_mix = eval_qlearn_under(args.p_attack_low)
    high_sum, high_mix = eval_qlearn_under(args.p_attack_high)

    print("\nThreat Check")
    print("LOW THREAT:", low_sum)
    print("LOW THREAT action mix:", low_mix)
    print("HIGH THREAT:", high_sum)
    print("HIGH THREAT action mix:", high_mix)

    # Compare heuristic policies under low/high threat
    def eval_policy_under(policy: str, p_attack: float, seed: int = 123) -> dict:
        P2 = P.copy()
        P2["defender_policy"] = policy
        P2["p_attack"] = float(p_attack)
        P2["Seed"] = int(seed)

        rng = np.random.default_rng(int(P2["Seed"]))
        df = run_sim(P2, make_initial_state(P2), rng, agent=None)
        return summarize_run(df)

    print("\n=== HEURISTICS THREAT CHECK ===")
    print("THRESH low :", eval_policy_under("threshold_v1", args.p_attack_low))
    print("THRESH high:", eval_policy_under("threshold_v1", args.p_attack_high))
    print("RAND low   :", eval_policy_under("random", args.p_attack_low))
    print("RAND high  :", eval_policy_under("random", args.p_attack_high))


if __name__ == "__main__":
    main()
