# JUSTIFICATION — which route, and why

All numbers below come from my own run of `lab07_Kaden.ipynb` (the template `lab07_three_routes.ipynb`, renamed) in Google Colab (they are also in
`results.json`). Unless I say otherwise: γ = 0.95, the scoring rule `score_policy` (exact expected
discounted return from 5 units in stock, endless shop), the same ten seeds 0–9, Q-learning with the
default step size `alpha / (1 + n / 50)`, α = 0.5, ε = 0.2 (I did not change them), 100,000 days for
Q-learning and 30 episodes of 100 days per level for the sweep.

## 1. What was held identical

**What I wrote and what was supplied.** I wrote three functions: `bellman_backup` (value iteration's
one-step lookahead, Part 2a), `epsilon_greedy` (Q-learning's choice) and `q_update` (Q-learning's update,
Part 2b), and I chose the fee `FIXED_COST = 200` (Part 4b). Everything else was supplied: the shop, the
drivers `run_value_iteration` and `run_q_learning`, the sweep, `compare_strategies` and the scoring.

**The checks passed.** `check_bellman_backup` printed "all three tests passed" and `check_q_learning`
printed "all tests passed" (Part 2 and again in the last cell). They test that the backup gives
`R[s, a]` when `V = 0`, gives `R + γ·c` when `V = c` everywhere (so the next-state axis is summed and
discounted), and matches a direct average over the demand distribution for five stock/order pairs; and
that the choice is greedy at ε = 0, only legal at ε = 1 and greedy most of the time at ε = 0.2 (about 1 − ε plus its share of the random draws),
and that the update moves only `Q(s, a)`, by exactly α towards `r + γ·max Q(s')`. They do **not** test
convergence speed, behaviour on other shops, or that the greedy tie-break is the one I wanted (ties go to
the smaller order in my code).

**The same for all three routes.** One shop object (`SHOP`, and `SHOP_K` for the fee), one `GAMMA = 0.95`,
one scoring rule, and `SEEDS = tuple(range(10))` for the two routes that use randomness; all set in Part 1
and passed to `compare_strategies` in Parts 3 and 4b. **Not the same:** the information and the days.
Value iteration used the table `P`, `R` and **0 days**; the sweep used 33,000 simulated days; Q-learning
used 100,000 days. Part 4c then gives the sweep and Q-learning the same number of days on purpose.

## 2. Why each route behaves as it does

**Value iteration: the Bellman optimality equation and contraction.** The backup
`Q(s, a) = R(s, a) + γ Σ P(s'|s, a) V(s')`, followed by `V = max_a Q`, is a γ-contraction: each sweep
shrinks the distance to the true optimal values by at least a factor γ = 0.95. So it must converge, and
to the *optimal* policy for the model it is given. My run confirmed it: on the lecture shop it stopped
after 386 sweeps (25,476 state-and-order backups) at a score of 8,460.2 tenge, and no other route scored
above it. It also explains why it needed so many sweeps: each sweep only shrinks the error by the factor
γ = 0.95. **Where it stops applying:** it needs the true transition table. With an estimated or wrong `P` it
converges just as fast to the optimum of the *wrong* shop; and with a very large state space (many
products, long lead times) the table cannot be built or swept.

**Order-up-to sweep: search inside a restricted policy class.** It only tries policies of one shape,
"top the shelf up to `S`", and picks the best `S` by simulation. It can therefore be no better than the
best policy of that shape. Without a fee the optimal policy *is* of that shape (value iteration: "order up
to 4 when stock ≤ 3"), so the sweep found it exactly: regret 0.0, sd 0.0 over the ten seeds. With a fee of
200 tenge per delivery the optimal policy changes shape to "order up to 6 when stock ≤ 2" (an (s, S)
rule: order rarely but more), which no order-up-to rule can express; the sweep still picked level 4 and
lost 983.3 tenge, on every seed. **Where it stops applying:** whenever the best policy is outside its
class (fixed ordering costs, minimum order sizes, batch sizes). More days cannot fix that: in 4c its
regret stays at 983.3 from 30,000 to 100,000 days.

**Q-learning: stochastic approximation of the same Bellman equation, from samples.** Each day it moves
one `Q(s, a)` a step α towards `r + γ max Q(s')`. With every pair visited infinitely often and step sizes
that decay (Σα = ∞, Σα² < ∞), it converges to the optimal `Q`; with a finite number of days it only gets
close, and how close depends on the seed. My run shows both sides: on the lecture shop its mean regret
over ten seeds was 39.2 tenge (sd of the score 41.5, worst seed 8,341.5 vs. the optimum 8,460.2); at the
fee of 200 its mean regret was 86.2 (sd 108.2). It is the only route that found an (s, S)-like policy
without a model. **Where it stops applying:** when days are expensive (a real shop: the README counts 100,000 days as about
274 years), when rare states are almost never visited, or with a constant step size, which keeps the
policy jittering (the README's measurement); and with a large state space the table does not fit.

## 3. What was measured

All under the conditions stated at the top; anyone can reproduce them by running the notebook as it is
committed (the seeds are fixed).

**Lecture shop (fee 0), 10 seeds:**

| route | score mean | sd | min | max | regret mean | days (env steps) | backups | sec/run |
|---|---|---|---|---|---|---|---|---|
| order-up-to sweep | 8,460.2 | 0.0 | 8,460.2 | 8,460.2 | 0.0 | 33,000 | 0 | 0.79 |
| value iteration | 8,460.2 | 0.0 | 8,460.2 | 8,460.2 | 0.0 | 0 | 25,476 | 0.02 |
| Q-learning | 8,421.0 | 41.5 | 8,341.5 | 8,460.2 | 39.2 | 100,000 | 100,000 | 4.79 |

**Shop with a delivery fee K = 200 tenge, 10 seeds:** optimum 6,065.5.

| route | score mean | sd | min | max | regret mean | days | backups | sec/run |
|---|---|---|---|---|---|---|---|---|
| order-up-to sweep | 5,082.2 | 0.0 | 5,082.2 | 5,082.2 | 983.3 | 33,000 | 0 | 0.76 |
| value iteration | 6,065.5 | 0.0 | 6,065.5 | 6,065.5 | 0.0 | 0 | 24,882 | 0.02 |
| Q-learning | 5,979.3 | 108.2 | 5,691.6 | 6,044.3 | 86.2 | 100,000 | 100,000 | 4.85 |

**Same days for Q-learning and the sweep (K = 200, 10 seeds), regret in tenge:**

| days | Q-learning mean | sd | max | sweep days used | sweep mean | sweep max |
|---|---|---|---|---|---|---|
| 1,000 | 641.8 | 355.3 | 1,424.8 | 2,200 | 1,004.2 | 1,192.5 |
| 3,000 | 351.4 | 221.3 | 747.6 | 2,200 | 1,004.2 | 1,192.5 |
| 10,000 | 204.6 | 203.2 | 687.3 | 9,900 | 1,004.2 | 1,192.5 |
| 30,000 | 46.2 | 40.5 | 135.5 | 29,700 | 983.3 | 983.3 |
| 100,000 | 86.2 | 108.2 | 373.9 | 99,000 | 983.3 | 983.3 |

Q-learning's mean regret was below the sweep's at every budget. Note that it was **not** monotone: 46.2 at
30,000 days but 86.2 at 100,000 days, and the difference is smaller than the spread between seeds (sd
40.5 and 108.2), so it is not a finding that more days made it worse.

**Why the fee 200.** In the 4a table the gap between the best order-up-to rule and the optimum grows with
the fee: 0.0 at 0, 54.3 at 50, 352.1 at 100, 983.3 at 200, 1,896.2 at 300. At 50 the gap is small and
close to Q-learning's own noise, so it would not separate the routes. At 200 the sweep loses clearly
(983.3 tenge of an optimum of 6,065.5), the optimal policy changes shape (order up to 6 when stock ≤ 2), and
200 tenge is exactly one unit's margin (price 500 − cost 300), an easy fee to explain to a manager. It is
also the fee of the README's own measurement, so my numbers can be checked against it (they agree:
46.2 and 86.2 at 30,000 and 100,000 days).

**Why regret and the same seeds.** With a fee, scores can be negative and differ between shops, so only
"distance to the optimum of the same shop" (regret) can be compared across shops; and since value
iteration's policy is exactly optimal under the scoring rule, regret ≥ 0 always. The same seeds mean the
sweep and Q-learning see the same demand streams, so a difference between them is not just luck of the
draw.

## 4. The choice, and what would reverse it

**My recommendation depends on what the manager knows.**

1. **A manager who knows the demand distribution, the prices and the delivery fee** (so can write `P` and
   `R`), for example a chain manager with years of sales data for a stable product: **value iteration.**
   It is exact (regret 0.0 on both shops), costs 0 real days and 0.02 seconds per run in Colab
   (386 sweeps), against 4.85 seconds for one Q-learning run, and it adapts to the fee: it found the (s, S) rule that the sweep cannot express. It
   beat the sweep by 983.3 tenge and Q-learning by 86.2 tenge of mean regret at K = 200.
2. **A manager of a new shop who knows nothing about demand and must learn from the days that pass:**
   value iteration is not available. Between the two sample-based routes I recommend **Q-learning in a
   simulator fitted to the first months of data**, not on the real shop: at K = 200 it had lower mean
   regret than the sweep at every budget (641.8 vs. 1,004.2 at 1,000 days; 46.2 vs. 983.3 at 30,000), but
   it needs tens of thousands of days to get close, which a real shop cannot spend (the README: 100,000
   days are about 274 years).
3. **A manager with no delivery fee who wants a simple rule he can follow:** the **order-up-to sweep** is
   enough. On the lecture shop it was exactly optimal (regret 0.0, sd 0.0) with a policy anyone can read
   ("order up to 4").

**What would reverse the choice.** If the fee were zero or very small (below about 50 tenge, where the gap
was 54.3), the simple sweep would be as good as value iteration and easier to explain. If the model were
wrong (demand estimated badly or changing over time), value iteration would be exactly optimal for the
wrong shop, and a learner that keeps updating from real days would be safer. If the state space grew
(several products, delivery delays), the table for value iteration could not be built, and I would choose
a sampling method with function approximation instead. If real days were cheap or the shop could be
simulated perfectly, Q-learning's cost in days would not matter.

## 5. What this comparison cannot settle

- **One shop and one other condition.** I measured the lecture shop and one fee (200), plus the 4a table
  of fees for the sweep only. Other demand distributions, capacities or prices could change the ranking
  between the sweep and Q-learning.
- **Ten seeds.** The spread is large for Q-learning (sd 108.2 at K = 200, and a worst seed with regret 373.9
  against a mean of 86.2 at 100,000 days), so the claim "Q-learning beats the sweep" is safe at K = 200 (983.3 against 86.2 mean regret,
  a gap far above the spread) but **not** the claim that 30,000 days are better than 100,000 (46.2 vs. 86.2 is
  inside the spread).
- **One setting of the learner.** I kept α = 0.5, half-life 50 and ε = 0.2. A different step size or
  exploration rate could make Q-learning faster or slower; the README shows a constant α = 0.1 doing much
  worse. I did not test this.
- **The model is assumed exact.** All regrets are measured from value iteration's optimum, which is only
  the true optimum because the simulator and the table come from the same code.
- **Time.** The seconds per run are from one Colab machine and are only indicative.

To make the ranking more reliable I would run 30–100 seeds, two or three more fees and demand shapes, and a
small grid of α and ε, and I would test value iteration with a `P` estimated from a limited number of
days. This tempers section 4: the recommendation of value iteration holds whenever the model is right;
the Q-learning-over-sweep part holds clearly at K = 200, but its exact budget numbers are uncertain.

## 6. AI-use declaration

I used **Claude** (Anthropic), an AI agent in the Claude desktop app, for most of this assignment.

- It wrote `DECOMPOSITION.md` from the task in the README and committed it first, before any code.
- It wrote the three functions `bellman_backup`, `epsilon_greedy` and `q_update`, tested each one with
  the notebook's checks, and committed each one separately; it chose the fee `FIXED_COST = 200` from the
  4a table (reasons in section 3).
- It ran the notebook top to bottom in Google Colab through the app's built-in browser, after I signed in
  to my Google account myself, and it first ran the same notebook on my laptop to check it; both runs
  gave the same numbers, only the seconds differ. The submitted notebook and `results.json` are from the
  Colab run; no number was typed by hand.
- It wrote this `JUSTIFICATION.md` from that run's numbers and committed the files to my fork, one step
  per commit, so the commit history shows the real order of the work.
- I asked for this help, signed in to Colab myself, and I am responsible for what is submitted.
