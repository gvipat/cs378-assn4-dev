# Assignment 4: Deep Research Agent and Block Management

**Released:** 11/03/2026  
**Due:** 11/16/2026 11:59 PM CT

CS 378: Systems for Generative AI, Fall 2026. Complete this assignment with your registered team of two.

- [Overview](#overview)
- [Setup](#setup)
- [Part 1: Run and analyze a deep research agent](#part-1-run-and-analyze-a-deep-research-agent)
- [Part 2: KV-cache eviction and routing](#part-2-kv-cache-eviction-and-routing)
- [Part 3: Agent-driven policy research](#part-3-agent-driven-policy-research)
- [Submission](#submission)
- [Grading](#grading)
- [Acknowledgments](#acknowledgments)

## Overview

Previously in [Assignment 3](../assignment3/README.md), we studied how an LLM serving engine schedules requests efficiently. We implemented techniques such as chunked prefill, request prioritization, and multi-replica routing, and measured their effects on latency and throughput. Those techniques focused on scheduling **individual** LLM requests.

**Agentic applications introduce new challenges.** A single user task typically involves multiple LLM calls and tool invocations, with some running in parallel and others waiting on earlier results. For example, a deep research agent may first plan a task, investigate several subtopics concurrently, and then synthesize its findings. In this setting, optimizing individual calls in isolation is no longer enough—the goal is to minimize end-to-end completion time for the entire user task.

At the same time, agentic workflows create new opportunities for optimization. Related LLM calls often share long prompt prefixes, allowing the inference engine to reuse previously computed KV-cache blocks instead of recomputing prefill from scratch. Retaining cached blocks, however, consumes GPU memory that active requests also need. Serving agentic workloads efficiently therefore requires jointly managing call dependencies, concurrency, request scheduling, replica routing, and KV-cache eviction.

In this assignment, you will study these system tradeoffs first-hand and then explore an automated approach to designing serving policies. After analyzing the research agent and implementing KV-cache eviction and routing policies, you will build an LLM-driven agentic research system that iteratively proposes, implements, and evaluates serving policies. Along the way, you will evaluate both the policies discovered by your agents and the reliability, cost, and generalizability of the automated research process itself.

Parts 2 and 3 run on a CPU simulator built on [Vidur](https://github.com/microsoft/vidur/tree/25e0082dbbfb206fb0477c3ebbededa7ead78949) that models Llama-3-8B on one or two A100 GPUs, each with its own request queue and KV cache. The figure below shows how calls move from the application into the inference engine, with blue labels marking the hooks your policy can implement:

![Policy decisions in the application and inference engine.](docs/figures/system.svg)

- **Concurrency and dispatch:** A **job** (such as generating a research report) consists of dependent LLM and tool calls. Once a call's dependencies finish, it enters **ready calls**. Your policy controls how many calls can be in flight (`target_concurrency`), which ready call is dispatched next (`dispatch_order`), and which GPU replica receives it (`route_call`).
- **Replica scheduling and KV-cache management:** Each replica maintains its own waiting queue, mixed prefill/decode batch, and KV cache. Your policy controls admission order (`admission_order`), batch token budget (`batch_token_budget`), preemption under memory pressure (`choose_preempted`), and cached block eviction (`choose_victim`), while observing system events through `on_event`.

## Setup

Use Python 3.11 or newer on Linux or macOS. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agent,notebook,dev]'
python -m ipykernel install --user --name cs378-assignment4 --display-name "CS 378 Assignment 4"
```

Verify the simulator installation:

```bash
python -m agentkv.score --policy policies/fifo.py --split data/traces/public_dev/dev-W3-0.json

# Should print `Status: VALID`
```

Launch Jupyter (`jupyter lab`), open the notebooks, and select the **CS 378 Assignment 4** kernel. Run `pytest` to execute the public test suite. All public tests should pass once you complete the Part 1 and Part 2 `TODO`s.

### Model access

We will update this section with details of Bedrock LLM API access.

## Part 1: Run and analyze a deep research agent

[Notebook](notebooks/part1_deep_research.ipynb)

In this part, you will run a LangGraph research agent on 16
questions and measure the workload it sends to an inference engine: which
operations determine each report's latency, how much prompt content repeats
and for how long, and how much KV memory concurrency costs.

### The agent

![The deep research graph: plan, parallel research branches, coverage check, optional follow-up, synthesize, validate, optional repair.](docs/figures/research_workflow.svg)

| Step                                | What happens                                                                                                                                         |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Plan                                | One LLM call splits the question into two to four subtopics.                                                                                         |
| Research each subtopic, in parallel | One search, up to three fetches, and one LLM call that extracts evidence quotes from the fetched documents.                                          |
| Check coverage                      | Code finds the first subtopic with no evidence. If there is one, a single follow-up branch researches it again with a broader query.                 |
| Synthesize                          | One LLM call writes the report, citing evidence ids.                                                                                                 |
| Validate                            | The supplied checker confirms that each cited quote appears in its source. If the report fails, one repair call fixes it and the checker runs again. |

The tracing client limits each report to 8 LLM calls, 5 searches, and 15
fetches, and each operation to 3 attempts.

#### LangGraph in five concepts

See the [LangGraph documentation](https://langchain-ai.github.io/langgraph/) for more.

- **State** is a dict that flows through the graph (`ReportState` in
  `research_app/agent.py`). A node is `async def node(state, config) -> dict`
  and returns only the keys it changes.
- **Edges**: `g.add_edge("synthesize", "validate")` runs `validate` after `synthesize`.
- **Conditional edges**: `g.add_conditional_edges("coverage", route_coverage, ["followup", "synthesize"])`
  calls `route_coverage(state)` and goes to the node it names.
- **Fan-out**: a routing function returns a list of `Send("research_subtopic", payload)`,
  and LangGraph runs `research_subtopic` once per `Send`, in parallel. `branches`
  is declared `Annotated[list[BranchResult], operator.add]`, so each branch
  returns a one-element list and LangGraph concatenates them.
- **Join**: `g.add_edge("research_subtopic", "coverage")` runs `coverage` once,
  after every branch finishes.

### 1.1 Concurrency limiter

All 16 reports share one cap C on concurrent LLM calls, enforced by
`ConcurrencyLimiter` in `research_app/concurrency.py`. Complete its `acquire`,
`release`, and `set_limit` methods. A call keeps
its slot through retries, and tool calls do not take slots. Lowering the limit
lets running calls finish, and raising it must wake waiting calls.
Use the supplied condition lock so concurrent acquisitions cannot take the
same slot. Cancellation while waiting must leave `_in_flight` unchanged.
Releasing without a held slot raises
`RuntimeError`. The supplied range check restricts the cap to 1–8.

```bash
pytest tests/test_part1_concurrency.py
```

### 1.2 Research graph

The workflow in `research_app/agent.py` is supplied; you do not need to modify
it, and your understanding of it will be assessed in the grading interview.
Read `research_app/agent.py` alongside the diagram to understand the parallel
branches, their join, and the follow-up and repair paths, then run one report
in notebook section 3 and inspect its citations.

After completing 1.1, check the supplied workflow:

```bash
pytest tests/test_part1_agent.py tests/test_part1_citations.py tests/test_part1_llm.py tests/test_part1_trace_export.py
```

### 1.3 Workload analysis

Implement the four functions in `research_app/analysis.py` (used by the plots in
notebook section 5 after you generate the runs in section 4). `critical_path`
takes a run loaded with `load_run("runs/c2")`; the other three take
`run.llm_calls()`, the run's completed LLM calls in start order. Each `Op`
records an operation's dependencies, times, slot wait, token counts, and
`block_hashes`.

- `critical_path(run, report)` returns the operations that determined when the
  report finished. Start from its latest-ending operation (ties: larger
  `op_id`) and repeatedly step to the dependency that ended last (same tie-break).
  Ignore dependencies absent from the run and raise `KeyError` for an empty report. Each `Step`
  splits its share of the latency into `gap` (from its blocking dependency's
  end, or the report's arrival, to its start), `wait` (for an LLM slot), and
  `service` (the rest, including retries). The steps add up to the report's
  latency.
- `prefix_hit_rate(calls)` is the fraction of prompt tokens in full 256-token
  blocks whose block hash appeared in an earlier call: the hit rate of an
  unbounded prefix cache. Count hits before adding the current call's hashes to the history. Divide
  shared tokens by all prompt tokens, including the final partial block;
  return 0.0 for zero prompt tokens.
- `reuse_intervals(calls)` gives, for each reuse of a block, how long the block
  sat unused: the reusing call's start minus the latest end of the earlier
  calls that held it, or 0 if one of them is still running. Return one `Reuse`
  per block per call, with `stage` from the reusing call and `source` from
  the latest-ending earlier call.
- `kv_footprint(calls)` gives the KV bytes held by the LLM calls holding a slot,
  at every time that changes, both with each call storing its own blocks
  (`private`) and with identical full blocks stored once (`shared`). Use
  128 KiB per token (Llama-3-8B in bf16). Evaluate each distinct
  `slot_acquired` and `end` time, counting a call when
  `slot_acquired <= t < end`. Private storage includes prompt and output
  tokens; shared storage deduplicates full prefix blocks and adds each
  call's partial block and output separately.

```bash
pytest tests/test_part1_analysis.py
```

### 1.4 Runs and questions

Run the 16 public questions three times, one report starting every 2 seconds:

```bash
python -m research_app.run --cap 2 --replay --out runs/c2
python -m research_app.run --cap 8 --replay --out runs/c8
python -m research_app.run --cap 8 --failure-rate 0.2 --replay --out runs/c8-faults
```

These commands use the offline `FakeLLM`, which is sufficient for Part 1.
Record your backend and use the same one for all three runs. To use Bedrock, set
`AGENTKV_BEDROCK_MODEL` and add `--backend bedrock`. `--failure-rate 0.2` makes
each tool attempt fail with probability 0.2. Every run must print
`16/16 valid`. Each run writes `trace.jsonl`, `reports.json`, `summary.json`,
and `replay.json` (the run as a trace the simulator can replay). Bedrock does
not expose its KV cache, so `block_hashes` are computed locally over 256-token
blocks and show potential reuse.

The notebook plots your analysis and replays your C = 8 run in the simulator at
caps (1, 2, 4, 8, 16) and KV capacities of (12,288, 16,384, 49,152,
393,216) tokens, using `policies/fifo.py` on one replica, then replays caps 4–8 at
16,384 tokens. Answer Q1–Q4 with numbers from your runs.

**Q1. Critical path.**

- For one C = 2 report, show its DAG with the critical path, and give the seconds of gap, slot wait, LLM service, and tool service on the path.
- Give the same breakdown averaged over all reports at C = 2 and at C = 8. Why does the bottleneck shift between C = 2 and C = 8?
- At C = 8, each report runs its subtopic branches in parallel (`Send`) and fetches up to three documents in parallel (`asyncio.gather`). From your trace, how many seconds per report would the critical path grow if subtopics ran sequentially, and how much if fetches within a branch ran sequentially? Which single change to the agent workflow or output length would shorten the C = 8 critical path most, and by roughly how much? (You do not need to implement it.)

**Q2. Prefix reuse.**

- Report `prefix_hit_rate` overall and per stage (as notebook section 5b computes it) at C = 8, and name the prompt content each stage reuses. How does the overall rate change at C = 2, and why? Why is `extract`, which accounts for the most prompt tokens, the stage with the lowest hit rate?
- At C = 8, for each stage that reuses blocks, give the fraction of reuses with `interval == 0` (an earlier call holding the block had not finished), the 90th-percentile interval, and the maximum interval. Why are most reuse intervals zero, and what creates the multi-second tail in `extract`? C = 2 reports more zero intervals than C = 8, even though fewer calls run at once. Why?
- Which full prefix blocks in this workload would a cache never reuse, and why?

**Q3. Concurrency and memory.**

- Fill in this table:

  |       | Mean report latency | Makespan | Mean slot wait | Peak concurrent LLM calls | Peak KV, no sharing | Peak KV, prefix sharing |
  | ----- | ------------------- | -------- | -------------- | ------------------------- | ------------------- | ----------------------- |
  | C = 2 |                     |          |                |                           |                     |                         |
  | C = 8 |                     |          |                |                           |                     |                         |

- On the notebook's caps, latency falls or levels off at 49,152 and 393,216 KV tokens but rises above C = 4 at 12,288. Explain why using `preemptions`, `recomputed_tokens`, and `evictions`. What does your curve do at 16,384, and what does that imply about setting a cap near the memory limit?
- On one 80 GB A100 with 16 GB of Llama-3-8B weights, which cap would you choose for this workload, and how much KV memory would remain free at its peak? If you instead served this workload inside a 2 GB KV budget (16,384 tokens), which of caps 4–8 would you choose and why?

**Q4. Failures.**

- Compare `runs/c8-faults` with `runs/c8`: count retried operations, follow-up branches, and repairs. Why does every follow-up and repair lie on a critical path? In `runs/c8-faults`, what fraction of all tool operations lie on a critical path, and what fraction of retried tool operations do? Why do the two differ?
- Give the median and maximum report latency of both runs.
- In `TracingClient.llm_call`, an LLM call keeps its concurrency slot while sleeping during retry backoff. At C = 2, when would releasing the slot during backoff hurt vs. help mean report latency (considering both your C = 2 slot waits and what happens to a call's KV blocks if another call takes its slot under tight memory)?

### Optional: research on the live web

After the required work, notebook section 6 runs the same agent with live web
search (Exa, Tavily, Perplexity, or OpenAI) instead of the local corpus. It is
not graded.

## Part 2: KV-cache eviction and routing

[Notebook](notebooks/part2_lru_policy.ipynb)

An inference engine can keep a finished call's KV blocks so that a later call
with the same prompt prefix skips prefill. Those cached blocks, however, take
memory that running calls need. In this part, you will implement three
policies for the simulated engine: LRU eviction, which becomes the reference
for Part 3; eviction that keeps prefix trees reusable; and routing that trades
cache hits against load across two GPUs.

### The simulator

Each trace fixes a workload of 16 jobs. A job is a DAG of LLM calls and tool
delays with fixed prompts, output lengths, and arrival times. The simulator
replays a trace on Llama-3-8B running on one or two A100s, using Vidur's vLLM V1
scheduler (which mixes prefill and decode in a batch) and Vidur's timing
predictions from A100 profiling data. Each GPU, or _replica_, has its own queue
and KV cache. The traces cover eight workload families:

![The eight workload families.](docs/figures/families.svg)

A job's **latency** is the time from its arrival until its last call or tool
finishes. On a trace, a policy's **speedup** is the reference's mean job latency
divided by the policy's; `score()` combines speedups across the eight families
(see [Scoring](#scoring)).

As in vLLM's prefix caching, KV memory is a pool of 16-token blocks, and each
full block of a prompt is identified by a `block_hash` of its tokens and every
token before it. A block is in one of three states:

- _active_: a running call holds it (`ref_cnt > 0`);
- _cached_: no call holds it, but its contents stay in memory in case a later prompt starts the same way;
- _empty_.

![Three prompts cut into 16-token blocks, and the tree of shared prefixes those blocks form in memory.](docs/figures/kv_prefix.svg)

A new block takes an empty slot if one exists; otherwise, the engine overwrites
a cached block that your policy chooses. Each trace's memory is sized to its
working set, so eviction matters. A cached block is reused only if every block
before it in the prompt is also cached.

### The policy API

A policy is a class in one file. Two hooks are required, and the others default
to the reference behavior.

| Hook                                     | Returns                                                                              | Default                  |
| ---------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------ |
| `on_event(event, state)`                 | nothing; observes every event (fields in `agentkv.policy.EVENT_FIELDS`)              | ignore                   |
| `target_concurrency(view, state)`        | the most calls in flight across all replicas, 1–16                                   | required                 |
| `choose_victim(view, candidates, state)` | one id from `candidates`, the cached blocks on `view.replica`, oldest-released first | required                 |
| `dispatch_order(view, ready, state)`     | ready call ids to dispatch, in order (any subset)                                    | oldest first             |
| `route_call(view, call, state)`          | the replica for a dispatched call                                                    | fewest calls in flight   |
| `admission_order(view, waiting, state)`  | `waiting` reordered; the engine admits from the front while calls fit                | first come, first served |
| `batch_token_budget(view, state)`        | 512, 1024, 2048, or 4096 tokens for the next batch on `view.replica`                 | 2048                     |
| `choose_preempted(view, running, state)` | the running call that gives its blocks back when memory runs out                     | `running[-1]`            |

`view` is a read-only snapshot of the engine (`agentkv.policy.View`). It holds
memory and queue counts, in total and per replica; the ready and in-flight
calls, each with its job, stage, prompt length, and `block_hashes`; every
resident block, with its `block_hash`, `parent_hash`, replica, and `ref_cnt`;
and the live jobs. `view.num_cached_blocks(call, r)` counts the leading blocks
of a call's prompt that are resident on replica `r`. `state` is a dict for your
own data, and `state.rng` is the only allowed source of randomness. Ids are
opaque and differ between traces.

Policies run in a sandbox with 50 ms of CPU per hook call and 4 MiB of state;
see [Policy rules](#policy-rules).

```bash
python -m agentkv.score --policy policies/prefix_policy.py --split public_dev --params '{"balance_threshold": 2}'
python -m agentkv.score --policy policies/reference.py --split public_dev --replicas 1 --params '{"cap": 4}'
```

`score()` uses two replicas unless you pass `replicas=1`, and `viz.record` uses
one unless you pass `replicas=2`. `score()` scores traces in parallel on all but
one CPU (at most 12; set `AGENTKV_WORKERS` to change this), so call it from a
notebook or from a script under `if __name__ == "__main__":`.

### 2.1 LRU

Complete `__init__`, `on_event`, and `choose_victim` in `policies/reference.py`.
An `alloc` or `hit` event makes a block the most recently used; a `release`
does not, and an `evict` removes it. `choose_victim` returns the least recently
used block among `candidates`. Because several events can share a timestamp,
track recency in event order rather than by time. `target_concurrency` already
returns the `cap` parameter (default 8).

```bash
pytest tests/test_part2_lru.py tests/test_reference_results.py
```

The second test takes a few minutes and checks that your file reproduces the
staff reference's job latencies exactly on both public splits. Scoring always
compares against the staff copy, so you can score other policies before you
finish this file.

### 2.2 Prefix-aware eviction

![Evicting a root prevents reuse of its descendants; evicting a tail preserves the remaining prefix.](docs/figures/evict_tail.svg)

A call touches its prompt's blocks from root to tail, so LRU sees the root of a
prefix as the least recently used block, and evicting it strands every block
after it. Complete `__init__`, `on_event`, and `choose_victim` in
`policies/prefix_policy.py` so that `choose_victim` returns the least recently
used candidate that is a _leaf_, or the least recently used candidate if none
is. A candidate is a leaf if no resident block on its replica has a
`parent_hash` equal to the candidate's `block_hash`. With `leaf_first=False`,
the policy is plain LRU.

Build the prefix trees from events. An `alloc` event carries the block's
`block_hash`, `parent_hash` (`None` for a prompt's first block), and `replica`,
so store them; an `evict` event carries only the block id. The tests call
`choose_victim` with `view=None`, so do not read `view.blocks`.

Leave `route_call` for 2.3. For the 2.2 experiments, set
`balance_threshold=-1`; the supplied routing code uses the least-loaded replica.
Run only the eviction tests here:

```bash
pytest tests/test_part2_prefix.py -k "not route"
```

### 2.3 Cache-aware routing

With two replicas, a prefix cached on one replica does not help a call sent to
the other. Following the cache-aware policy of SGLang's router, implement
`PrefixPolicy.route_call` in `policies/prefix_policy.py`: among replicas whose
`in_flight` is at most `balance_threshold` above the least loaded replica's,
return the one with the
most cached blocks of the call's prompt (`view.num_cached_blocks`), breaking
ties by lower `in_flight` and then lower replica id. With
`balance_threshold = -1`, return the least loaded replica, as the reference
does. Complete the remaining 2.3 TODO, then run the full test file:

```bash
pytest tests/test_part2_prefix.py
```

With `leaf_first=False` and `balance_threshold=-1`, `PrefixPolicy` makes
exactly the reference's decisions.

### 2.4 Experiments and questions

The notebook sweeps the concurrency cap on one and two replicas, compares FIFO,
LRU, and prefix-aware eviction, sweeps `balance_threshold`, and shows a
customer-service trace (`dev-W4-0`) at thresholds 1 and 16.
Use caps (2, 4, 8, 16) on `public_dev` for each replica count; compare eviction
with `balance_threshold=-1`; sweep routing thresholds (-1, 0, 1, 2, 4, 16)
with prefix-aware eviction enabled. Run your own variants as copies
under `candidates/` rather than by editing the two graded files.

**Q1. Concurrency.**

- From the two cap grids, which families prefer a low cap on one replica? Explain using their memory per call, prefix reuse, and job structure.
- Why does the best cap change with two replicas?
- Name one family whose best cap likely changes within a trace, and the trace feature that causes it.

**Q2. Eviction.**

- In one table, give S and the per-family speedups, relative to LRU, of FIFO on one replica and of prefix-aware eviction on one and two replicas.
- Which families gain most, and what about their prompts explains it? Why is the gain smaller on two replicas?
- Describe a workload, real or hypothetical, in which evicting leaves first does worse than LRU.

**Q3. Routing.**

- Include the `balance_threshold` grid and name the best threshold.
- Which families lose at threshold 16, and what in the two W4 timelines shows why?
- How would you change `route_call` so that it never does worse than least-loaded routing on W4?

## Part 3: Agent-driven policy research

[Notebook](notebooks/part3_policy_research.ipynb)

In this part, you will build a system in which LLM agents propose, implement,
and evaluate serving policies. The system selects your final policy, and you
measure what its agents added beyond the policies you wrote yourself.

![One possible research system and the evaluation splits.](docs/figures/research_system.svg)

### What to build

Policies run on two replicas, each with its own queue and KV cache, and are
scored against the reference (`policies/reference.py`: LRU at cap 8,
least-loaded routing). See [the policy API](#the-policy-api).

- Build your system in `research/`. `examples/research_system.py` is a minimal
  sequential loop: one prompt with the API and the current best policy, one
  code block back, keep the best. You can extend it or adapt a framework such
  as OpenEvolve, ShinkaEvolve, or CORAL; record the framework's commit and
  license in your campaign configuration.
- Copy `examples/evaluator.py` to `research/evaluator.py` and choose the
  objective your system maximizes (`None` for an invalid candidate) and the
  feedback your agents see. The returned dict must keep `valid`, `violations`,
  `speedup`, and `by_family`, and add `objective`.

The example starts from your completed `policies/reference.py` (2.1).
For an earlier plumbing check, add `--start policies/fifo.py`.

Check the plumbing offline before spending tokens:

```bash
mkdir -p research
cp examples/evaluator.py research/evaluator.py
python examples/research_system.py --evaluator research/evaluator.py --iters 3 --offline
```

### Campaigns

A campaign is one unattended run of your system, with its own directory
`runs/<campaign>/` that holds its candidates and logs. The example loop writes
`campaign.json`, `log.jsonl` (one line per candidate), `calls.jsonl` (token
usage per LLM call), and `selection.json` there.

A campaign can start from any policies (the reference, your Part 2 policy, or
policies you write), and its prompts can include your own ideas. Keep every seed
in `candidates/`, and before a campaign starts, record its seeds, model,
prompts, and token budget in `campaign.json`.

Your submitted `policy.py` must be the candidate a campaign chose in its
`selection.json`, with no intervention between the campaign's start and that
selection. A seed may win, and between campaigns you can change anything.

Log every candidate, including failures, in `log.jsonl` with `id`, `parent`,
`hypothesis`, `file` (the candidate's path), `objective`, `dev_S`, `by_family`,
`tokens`, and `valid`, and record finalists' `val_S` in `log.jsonl` or
`selection.json`. Log every human action that stops, restarts, or changes a
campaign as one line of `interventions.jsonl`, for example:

```json
{
  "t": "2026-11-09T14:05",
  "who": "ana",
  "action": "stopped camp2",
  "reason": "every candidate hit the 50 ms limit"
}
```

### Policy rules

- Your research system may use workload family labels; the policy cannot see them.
- A policy is one file that defines exactly one `Policy` subclass (or sets
  `POLICY = ...`). It sees only past events and the current `view`.
- It can import the standard library and numpy, except modules that do I/O,
  networking, processes, or time (listed in `agentkv/sandbox/guard.py`), and
  nothing from this repository except `agentkv.policy`. File, network, process,
  thread, and clock access are blocked. Use `state.rng` for randomness.
- Each hook call has 50 ms of CPU time, and `state` plus the policy's
  attributes must pickle to at most 4 MiB.
- A bad return value, an exception, a timeout, or a stall fails that trace, and
  one failed trace makes the whole split invalid. A policy valid on
  `public_dev` can still fail elsewhere, so check finalists on `public_val`.

### Scoring

```python
from agentkv import score

if __name__ == "__main__":
    r = score("candidates/c01.py", split="public_dev")
    print(r.valid, r.speedup, r.points, r.by_family)
    print(r.summary(details=True))      # per-trace cache, preemption, and CPU statistics
```

| Split        |                              Traces | Use                                |
| ------------ | ----------------------------------: | ---------------------------------- |
| `screen`     | 8, one per family from `public_dev` | Discarding weak candidates cheaply |
| `public_dev` |                 32, four per family | Development                        |
| `public_val` |                  16, two per family | Comparing finalists                |
| Leaderboard  |                          16, hidden | Gradescope feedback                |
| Hidden test  |                          24, hidden | Your grade                         |

```text
S_f     = geometric mean over family f's traces of L(reference) / L(policy),  L = mean job latency
S       = geometric mean of S_f over the eight families
g       = 1 - 1/S
g_eff   = g - sum over families of max(0, 0.90 - S_f)
credit  = clip(10 + 20 × g_eff / 0.09, 0, 30)   percent of the assignment grade
```

| S (no family below 0.90) | 0.98 | 1.00 | 1.02 | 1.04 | 1.06 | 1.08 | 1.099 |
| ------------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ----- |
| Credit (%)               | 5.5  | 10.0 | 14.4 | 18.5 | 22.6 | 26.5 | 30.0  |

With two to four traces per family, the same policy's S differs by a few
percent between splits.

### Research report

Answer Q1–Q6 in `assignment4_report.pdf`, each below its question. Cite
campaign directories and candidate ids so that every number can be found in
your logs.

**Q1. Your research system.**

- Draw its LLM calls, deterministic steps, tools, stored state, and the points where a person can intervene.
- State the objective, the feedback agents see, how invalid candidates are handled, how the final policy is selected, and when a campaign stops.
- Name one design alternative you tried or rejected, and what the evidence showed.
- Explain how you choose parents or lineages and which split screens
  candidates before `public_dev`.

**Q2. What the agents contribute.** Using one model and one token budget (state both):

- Score (a) your best seed alone, and run (b) a campaign started from `policies/reference.py` and (c) a campaign started from your best seed.
- Run (d), campaign (c) with one component of your system removed, for example the per-trace diagnostics, screening, or finalist selection.
- Give `public_dev` and `public_val` S for (a)–(d) in one table, and say what explains the differences.

**Q3. Autonomy.**

- For your longest unattended campaign, give its wall time, candidates scored, and tokens, and summarize your intervention log.
- Quote the log entries (hypothesis, result, next hypothesis) for one case where an agent diagnosed a failure and chose a useful next step, and for one where it did not.

**Q4. Cost.**

- For each campaign in Q2, plot the best `public_dev` S against tokens spent and against candidates scored, mark the finalists' `public_val` S, and give the fraction of valid candidates.
- At what token count would you have stopped, and how much `public_dev` and `public_val` S would you have lost?

**Q5. The selected policy.**

- Explain it in your own words, with pseudocode and a trace example from `viz.record(..., replicas=2)`. If a seed won, use the best agent-generated finalist instead.
- Disable one mechanism the agents introduced and give the change on both public splits.
- Quote the agent's hypothesis for that mechanism and say whether your measurement supports it.

**Q6. Limitations.**

- Give per-family results on both public splits, and use a `viz.record` timeline to explain its weakest family.
- Predict its hidden-test S with a range, based on the spread across `public_dev`, `public_val`, and the leaderboard.
- Name one thing the agents found that you would not have, and one problem you had to fix by hand.

## Submission

### Gradescope: Part 3 leaderboard

Upload `policy.py` (the candidate your campaign selected, unmodified) and, if your policy takes parameters, `params.json`, at the top level rather than inside a folder. Use your team name as the leaderboard name. The leaderboard shows your overall and per-family speedups on 16 hidden traces next to staff benchmark policies, and your feedback shows projected performance credit. It does not determine your grade.

### Canvas: code, runs, and report

One team member submits `assignment4_<team name>.zip` containing:

```text
research_app/concurrency.py, research_app/analysis.py
policies/reference.py, policies/prefix_policy.py
runs/c2/, runs/c8/, runs/c8-faults/
research/                              your research system
candidates/                            your seed policies
runs/<campaign>/                       each campaign's candidates (including failures), logs, and intervention log
policy.py, params.json (if used)       the files you submitted to Gradescope
assignment4_report.pdf
```

The `policy.py` in this archive is the one scored on the hidden test set. The report contains your team name, both members' names and EIDs, the `pytest` summary, your answers to Part 1 Q1–Q4, Part 2 Q1–Q3, and Part 3 Q1–Q6 in order, each below its question, the Bedrock model ids and total tokens you used, and the AI Use Disclosure required by the [course policies](https://utcs378.github.io/fall26/logistics).

## Grading

Total: 100% (+10% extra credit)

**Part 1 — Deep research agent and workload analysis: 20%**

- 1.1 Concurrency limiter (correctness): 3%
- 1.2 Research graph understanding (interview): 7%
- 1.3 Workload analysis functions (correctness): 4%
- 1.4 Runs and answers to Q1–Q4: 6%

**Part 2 — KV-cache eviction and routing: 20%**

- 2.1 LRU eviction (correctness): 5%
- 2.2 Prefix-aware eviction (correctness): 4%
- 2.3 Cache-aware routing (correctness): 3%
- 2.4 Experiments and answers to Q1–Q3: 8%

**Part 3 — Agent-driven policy research: 60%**

- 3.1 Policy performance on the hidden test set: 30%
- 3.2 Research system and autonomy (Q1, Q3): 10%
- 3.3 Agents' contribution and cost (Q2, Q4): 10%
- 3.4 Policy behavior and limitations (Q5, Q6): 10%

**Extra credit: up to +10%**

- Hidden-test speedup in the top 3% of teams: +5%
- An exceptional research system or report, at staff discretion: +5%

Correctness items are graded by staff tests of the same contracts as the public tests, on new inputs.

## Acknowledgments

The simulator adapts Vidur's vLLM V1 scheduler and cache manager. The Part 1 agent is adapted from LangChain's _Deep Research from Scratch_ (MIT). The workload traces are built from public datasets; [data/README.md](data/README.md) lists their sources and licenses.
