#!/usr/bin/env python3

from pantograph.server import Server
from pantograph.expr import TacticDraft

root = """
theorem add_comm_proved_formal_sketch : ∀ n m : Nat, n + m = m + n := sorry
"""

sketch = """
by
  -- Consider some n and m in Nats.
  intros n m
  -- Perform induction on n.
  induction n with
  | zero =>
    sorry
  | succ n ih =>
    -- Inductive step: Assume n + m = m + n, we need to show succ n + m = m + succ n.
    -- By the inductive hypothesis, we have n + m = m + n.
    have h_inductive: n + m = m + n := sorry
    -- 1. Note we start with: Nat.succ n + m = m + Nat.succ n, so, pull the succ out from m + Nat.succ n on the right side from the addition using addition facts Nat.add_succ.
    have h_pull_succ_out_from_right: m + Nat.succ n = Nat.succ (m + n) := sorry
    -- 2. then to flip m + S n to something like S (n + m) we need to use the IH.
    have h_flip_n_plus_m: Nat.succ (n + m) = Nat.succ (m + n) := sorry
    -- 3. Now the n & m are on the correct sides Nat.succ n + m = Nat.succ (n + m), so let's use the def of addition to pull out the succ from the addition on the left using Nat.succ_add.
    have h_pull_succ_out_from_left: Nat.succ n + m = Nat.succ (n + m) := sorry
    -- Combine facts to close goal
    sorry
"""

if __name__ == '__main__':
    server = Server()
    unit, = server.load_sorry(root)
    print(unit.goal_state)

    # Send the draft payload using `TacticDraft`
    state1 = server.goal_tactic(
        unit.goal_state,
        tactic=TacticDraft(sketch))
    print(state1)
