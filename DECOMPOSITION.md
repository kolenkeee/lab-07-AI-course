# DECOMPOSITION — the task in named parts, before any code

**The task.** Given the shop, get an ordering policy by each of three routes (order-up-to sweep, value
iteration, Q-learning), score all three by one rule, see what changes when one condition of the shop
changes (a delivery fee), and recommend one route in writing.

I wrote this file from the task as the README states it, before I wrote or ran any code.

## 1. The subproblems

I split the task into eight subproblems. S1 and S2 are shared by everything; S3, S4 and S5 are the three
routes; S6, S7 and S8 turn the routes into a comparison and a choice.

| # | Name | What it does |
|---|---|---|
| S1 | **Shop model** | Defines one shop: stock levels, legal orders, prices, demand, delivery fee, and the rules of one day (sell, pay, restock). It can *simulate* a day and can *write out* the full transition table `P` and reward table `R`. |
| S2 | **Scoring rule** | Gives one number to any policy: its exact expected discounted return from the start stock (5 units), endless shop, γ = 0.95. Also computes **regret** = optimum − score. |
| S3 | **Route A: order-up-to sweep** | For each level `S`, builds the rule "order up to `S`", estimates it from simulated days, and keeps the best `S`. Uses only the simulator. |
| S4 | **Route B: value iteration** | From `P` and `R`, repeats the Bellman optimality backup until `V` stops moving, then takes the greedy policy. Uses the model, no days. |
| S5 | **Route C: Q-learning** | From simulated days only, keeps a table `Q(s, a)`, chooses orders ε-greedy and updates one entry per day, then takes the greedy policy. |
| S6 | **Comparison under one rule** | Runs S3, S4, S5 on the **same** shop, γ and seeds, scores each final policy with S2, and reports regret, spread across seeds and cost (days, backups, seconds). |
| S7 | **Sensitivity** | Changes one condition and runs S6 again: (a) a delivery fee `K > 0`; (b) the same budget of days for S3 and S5, from small to large. |
| S8 | **Write-up** | Turns the numbers from S6 and S7 into a recommendation for a named shop manager, with the conditions that would reverse it and the limits of the comparison (`JUSTIFICATION.md`, `results.json`). |

**No overlaps, no gaps.** Each step of the task maps to exactly one row: "get a policy by each of three
routes" = S3, S4, S5; "score all three by one rule" = S2 + S6; "change one condition" = S7; "recommend in
writing" = S8; and S1 is what all of them act on. A reader can check completeness by walking the task
sentence word by word against the table, and can check that nothing appears twice because **only S2
scores a policy** (the routes return policies, never scores they computed themselves) and **only S1
knows the rules of a day** (the routes call it, they do not copy it).

**Could the set be smaller?** Yes: S2 could be folded into S6, and S7 could be a loop inside S6. I keep
them apart on purpose:
- S2 apart from the routes, because the whole comparison is fair only if one piece of code scores every
  policy. If each route scored itself (S3 by simulation, S5 by its own `Q` values), the numbers would mean
  different things.
- S7 apart from S6, because S6 must be finished and trusted on the lecture shop *before* a condition is
  changed; otherwise I cannot tell whether a change in the ranking comes from the fee or from a bug.
- S8 apart, because the recommendation depends on who the manager is and what a day costs them, which no
  number in S6 contains.

## 2. Inputs, outputs and data flow

| # | Inputs | Outputs | When the input is wrong or the result is not usable |
|---|---|---|---|
| S1 | `ShopParams` (max stock, prices, demand distribution, `order_fixed_cost` = fee K ≥ 0) | `step(stock, order) -> (next_stock, reward)`; `P`: array `(n_actions, n_states, n_states)`; `R`: array `(n_states, n_actions)`; `mask`: bool array `(n_states, n_actions)` of legal orders | Rejects a negative fee, an order that overflows the shelf, or a demand distribution that does not sum to 1. |
| S2 | a policy (table `(n_states, n_actions)` of probabilities), `P`, `R`, γ, start stock | `score: float` (tenge); `regret = optimum − score: float` | Rejects a policy that puts probability on an illegal order or whose rows do not sum to 1. |
| S3 | S1 simulator, γ, list of levels `S`, days per level, seeds | best level `S*`, its policy, days used | If two levels tie within noise, reports both; the final number is the S2 score, not the noisy estimate. |
| S4 | `P`, `R`, `mask`, γ, tolerance, max iterations; my function `bellman_backup(P, R, mask, V, γ) -> Q` with `Q` of shape `(n_states, n_actions)` and illegal orders set to −∞ | `V*`: `(n_states,)`, greedy policy, number of sweeps and backups | If `V` does not settle within the max iterations, it reports "not converged" and the result is not used as the optimum. |
| S5 | S1 simulator, γ, number of days, seed, ε, step size α and its decay; my functions `epsilon_greedy(Q, s, ε, mask, rng) -> a` and `q_update(Q, s, a, r, s2, α, γ, mask)` (in place) | `Q`: `(n_states, n_actions)`, greedy policy, days used, backups | Must only ever choose legal orders; a state never visited keeps its initial values, and the greedy choice there is still legal. |
| S6 | lecture shop, γ = 0.95, the same 10 seeds, the policies from S3–S5 | table per route: mean regret, sd, min, max over seeds, days, backups, sec/run | If S4 did not converge there is no trusted optimum, so regrets are not reported. |
| S7 | S6 plus a fee K (chosen from a table of fees vs. the sweep's gap), and day budgets (1,000 to 100,000) | the same table at fee K; regret vs. budget for S3 and S5 | A budget too small for the sweep to try every level (fewer than 2,200 days) is reported as such, not as a result. |
| S8 | the tables of S6 and S7 | `JUSTIFICATION.md`; `results.json` with the numbers | Every number quoted must appear in the notebook output. |

**Data flow, in the order to follow:**
1. S1 builds the shop → gives the simulator to S3 and S5, and `P`, `R`, `mask` to S2 and S4.
2. S4 runs first → its policy, scored by S2, is the **optimum**.
3. S3 and S5 run on the same seeds → their policies go to S2 → **regret** against the optimum.
4. S6 collects all three into one table.
5. S7 changes the fee in S1 and repeats steps 1–4; then repeats S3 and S5 for each day budget.
6. S8 reads the tables of S6 and S7.

```
S1 shop ──P,R,mask──► S4 value iteration ──policy──┐
   │                                                ├──► S2 score/regret ──► S6 table ──► S7 (new fee, budgets) ──► S8 write-up
   └──simulator──► S3 sweep, S5 Q-learning ──policy─┘
```

**Can one subproblem be built from this alone?** Yes for S4 and S5: their inputs, outputs and shapes are
fixed above, so `bellman_backup`, `epsilon_greedy` and `q_update` can be written and tested on a tiny
hand-made table without the rest of the shop.

## 3. Constraints, assumptions and the reason for the split

**Constraints.**
- *Same for all routes* (S6, S7): one shop, one γ = 0.95, one scoring rule (S2), the same ten seeds for
  the routes that use randomness. Without this the regrets are not comparable.
- *What each route may use*: S3 and S5 may use **only the simulator** (days), never `P` or `R`; S4 may use
  **only `P` and `R`**, no days. This is the real difference the lab measures.
- *Legal actions*: every route must return a policy that only orders what fits on the shelf (S1 `mask`).
- *Every number from the run* (S8): nothing in the write-up may be typed by hand.

**Assumptions, and where each matters.**
- The shop is a finite, stationary Markov decision process: the next stock depends only on today's stock,
  order and demand (S1, S4). Value iteration's guarantee rests on this.
- Demand is independent from day to day and its distribution does not change (S1, S3, S5).
- γ < 1, so the discounted sum is finite and the backup is a contraction (S2, S4).
- Q-learning visits every state-and-order pair often enough and its step size decays (S5); with a fixed
  budget of days this may fail for rare states.
- One simulated day costs nothing here, but a real day costs a day (S7, S8).

**Why this split.** It follows what each route *needs* (simulator vs. model) and separates "make a
policy" from "judge a policy". That is exactly the comparison the task asks for, and it lets each route
be replaced without touching the others.

**Alternative split I considered and rejected.** Split by *time phase*: (1) "setup" = shop + all route
code, (2) "run everything", (3) "analyse". I rejected it because it puts the three routes and the scoring
rule into one block: a route could then score its own policy, and a bug in one route would hide inside the
"run" phase together with the others. It also hides the key difference (model vs. days), which my split
makes visible as separate inputs.

**The assumption that would hurt most if wrong:** that the transition table `P` is the true shop. Value
iteration is "exactly optimal" only for the model it is given, and I use its result as the optimum that
every regret is measured against. If `P` were wrong (as it would be for a real shop whose demand is only
estimated), the "optimum" would be wrong too, and every regret in the comparison would be measured from the
wrong point.
