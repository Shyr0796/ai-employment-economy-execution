# AI Employment Economy Execution Layer

This package is the independent implementation of `AI_Employment_Economy_Execution_Layer.html`.
It does not import the old `ai_economy_sim` economic core. It retains the prior project's
OpenAI-compatible provider convention and runs the three economic roles through AgentSociety 2.7.

## Architecture

- `EconomicAgent`: one AgentSociety class with `resident`, `firm`, and `government` profiles.
- `ExecutionEconomyEnv`: the only owner of the mutable economy and monthly clearing process.
- `ExecutionRouter`: deterministic JSON routing for observation and bounded intents; it makes no LLM call.
- `EconomyEngine`: common numerical core used by AgentSociety and fast common-random-number experiments.
- LLM decisions are optional and categorical. Agents cannot directly set wages, cash, employment totals,
  prices, tax rates, transfers, or production capacity.

The baseline creates 500 resident household agents, 30 firm agents, and one government agent. The
employment count is 474. Population scaling recomputes firms using the 15.8-worker average and scales
resident demand, government purchases, capacity, and fiscal totals proportionally.

## Institutional E2-E4 design

The current institutional design assigns:

- `E2`: employment-preserving AI responsibility, unchanged monthly wages,
  temporary 25% wage-cost sharing, restructuring grace, and auditable
  reductions in required work hours;
- `E3`: a graduated 20%-to-50% levy on bounded AI productivity rents, with a
  basic-consumption allowance, 20% price incidence, and accelerated 70/30
  recycling into public services and public investment;
- `E4`: AI-saved-time incubation followed by voluntary, non-dual
  solo-enterprise formation with substitution, B2B, induced, and bounded
  external demand reported separately.

The full design, evidence boundary, qualification gates, and network-free smoke
command are in [`INSTITUTIONAL_EXPERIMENTS.md`](../INSTITUTIONAL_EXPERIMENTS.md).
Historical artifacts produced under older E2-E4 meanings are not robustness
replications of this source version.

## Versioned E5-E6 definitions

E5 and E6 have explicit, recorded semantic versions:

- `institutional_v2` (default): E5 combines the E2 employment-responsibility
  mechanism, E3 AI-rent levy, E4 solo-enterprise channel, transfers,
  procurement, and government AI. E6 applies the same portfolio under a 2%
  annual-deficit limit and 40% debt limit.
- `legacy_v1`: reproduces the historical E5/E6 definitions with transfers,
  procurement, and government AI, but without the new E2-E4 institutional
  mechanisms.

Every resolved config, run manifest, and suite manifest records
`scenario_definition_version`. Select the historical semantics explicitly:

```bash
.venv/bin/python -m ai_economy_execution.run \
  --scenario E5 --scenario-definition-version legacy_v1 \
  --population 500 --months 120 \
  --provider offline --llm-roles "" --cognitive-regime R0 \
  --result-stage smoke
```

Do not pool `legacy_v1` and `institutional_v2` outputs in one estimand.

## Standard R x E result layout

New matrix runs use `matrix_v1` and are written automatically under:

```text
results/research_matrix/
  {smoke|pilot|formal}/
    {scenario_definition_version}/
      {R0_rules|R1_government|R2_firm_government|R3_resident_firm_government}/
        {provider}_{model}/
          N{population:05d}_M{months:03d}_S{seed:03d}/
            equilibrium/
            E0/ ... E6/
```

The role mappings are fixed: R0 has no LLM role, R1 uses government, R2 uses
firm plus government, and R3 uses resident plus firm plus government.
`--cognitive-regime` is an optional assertion and fails when it conflicts with
`--llm-roles`. Use `--result-stage smoke`, `pilot`, or `formal`; the default is
`smoke`. Omit `--output` for the standard layout. A supplied `--output` remains
an expert override for tests and migrations.

### Unified cognitive matrix runner

`cognitive_matrix_suite` is the authoritative R x E orchestrator. For every
seed it first creates one qualified offline R0 equilibrium for months 1-24,
then forks every requested E scenario from that exact checkpoint. R1-R3 roles
activate on month 25 together with the AI shock. The runner performs a
network-free API/module preflight before paid cells, stops at the first failed
cell, preserves completed cells, and skips matching completed cells when the
same command is rerun.

Without `--execute`, it only prints the complete plan:

```bash
.venv/bin/python -m ai_economy_execution.cognitive_matrix_suite \
  --population 100 --months 36 --seeds 1 \
  --regimes R0-R3 --scenarios E0-E6 \
  --provider hkust --model gpt-3.5-turbo \
  --result-stage smoke
```

Add `--execute` only after checking the plan and loading `HKUST_API_KEY`.
The complete formal five-seed command is:

```bash
.venv/bin/python -m ai_economy_execution.cognitive_matrix_suite \
  --population 500 --months 120 --seeds 1-5 \
  --regimes R0-R3 --scenarios E0-E6 \
  --provider hkust --model gpt-3.5-turbo \
  --result-stage formal --execute
```

Per-cell stdout/stderr is stored under the corresponding orchestration
directory's `logs/`; `matrix_manifest.json` is updated after every cell.

## Core behavior

- Resident consumption follows personal baseline disposable income, marginal propensity to consume,
  precautionary behavior after model-generated job loss, cash drawdown limits, and a minimum floor.
  Employed residents can also pay for personal AI; this raises effective productivity but is recorded
  separately from welfare consumption.
- Final demand is allocated by lagged firm market share. After the AI shock, relative price, productive
  capacity, and employment reputation move shares; supply shortage is proportionally rationed.
- Three configurable firm cultures (`augmentation`, `cost_cutter`, and `adaptive`) change retention floors,
  layoff delays, AI-complementary jobs, adjustment speed, and price pass-through. They do not directly set
  employment or profits.
- AI expands worker capacity gradually. Its gains pass through to prices and wages using configurable shares.
- Firms can borrow, invest, fail, exit, and enter. The current boundary still excludes inventories,
  household dividends, sectoral input-output links, and the export sector.
- Government policy observes a two-month lag. Five deterministic strategies compare passive safety nets,
  active demand management, an isolated productivity-dividend procurement channel, fiscal guarding, and active demand plus
  a targeted below-cost-pricing penalty. Employment support is allocated only to firms retaining at least
  85% of initial employment and its transition component sunsets over 24 months.
- The real-consumption deflator uses the fixed month-0 firm basket: entrants are excluded and an exited
  firm's final observed price is carried forward.

Every month validates sales, wage, and tax accounting identities.

## Bounded trend context and behavior audit

Agent observations now include deterministic changes over the latest three completed monthly
records. Residents see unemployment, aggregate-price, and real-consumption changes; firms see
aggregate sales, capacity-utilization, and market-concentration changes; government sees
unemployment, household-consumption, and debt-ratio changes. These fields are read-only context:
they add no agent memory, action, economic parameter, dependency, or extra LLM request. The rule-only
policy does not consume them, so the numerical E0--E6 paths remain unchanged.

Each AgentSociety run converts the existing `decision_audit.jsonl` into:

- `decision_behavior_summary.json`: role/status/action counts, conditioned action shares, and
  five directional response checks;
- `decision_behavior_summary.csv`: tidy role-by-condition action shares for later analysis.

The checks cover household income and unemployment stress, weak firm demand, government
unemployment response, and the government debt guard. A missing comparison group is reported as
`available: false`; it is not silently treated as a pass.

## Run AgentSociety

Offline rule mode makes no API request:

```bash
.venv/bin/python -m ai_economy_execution.run \
  --scenario E5 --population 500 --months 120 --seed 1 \
  --scenario-definition-version institutional_v2 \
  --provider offline --llm-roles "" --cognitive-regime R0 \
  --result-stage smoke
```

Use the inherited provider mapping and enable LLM decisions only for selected roles:

```bash
.venv/bin/python -m ai_economy_execution.run \
  --scenario E5 --provider hkust --llm-roles government \
  --cognitive-regime R1 --result-stage smoke
```

Full-role HKUST run at the requested 500-resident scale:

```bash
.venv/bin/python -m ai_economy_execution.run \
  --scenario E5 --population 500 --months 120 --seed 1 \
  --provider hkust --llm-roles resident,firm,government \
  --cognitive-regime R3 --result-stage pilot \
  --batch-size 16 --llm-max-workers 2 --llm-concurrency 2 \
  --max-fallback-rate 0.01 --max-role-fallback-rate 0.01 \
  --require-response-model-match
```

`--batch-size` controls how many agents share one Ray task; it is not an API concurrency limit.
Smaller batches create more Ray tasks, so a safe pilot must also set `--llm-max-workers` and
`--llm-concurrency`. The values above start at no more than four concurrent requests. With a
16-agent batch, the per-task adaptive semaphore does not reach its 20-completion adjustment round,
so this pilot setting stays bounded in practice. The resolved worker count, per-worker concurrency,
Ray task count, and initial global concurrency ceiling are written to `resolved_config.json`.

The decision quality gate defaults to a 1% overall and per-role fallback ceiling. A violation
finishes and preserves all artifacts but marks the run `quality_gate_failed`. Successful responses
record the requested model, response-reported model, safe deployment/provider identifiers, and
selected request IDs/region headers. `--require-response-model-match` rejects a gateway route whose
response model differs from the requested model or does not report a model identifier. The only
provider-specific exceptions are the HKUST gateway's documented legacy aliases:
`gpt-3.5-turbo` may report `gpt-4o-mini[-YYYY-MM-DD]`, and `gpt-4` may report
`gpt-4o[-YYYY-MM-DD]`. Exact, aliased, and mismatched counts plus the applied policy are persisted
under `decision_audit.model_provenance` in `run_manifest.json`.
Per-request LiteLLM response caching is disabled for economic agents so every accepted audit record
comes from an actual provider response rather than a prior identical prompt.
The HKUST gateway's non-standard string `created` timestamp is normalized before both OpenAI SDK
and aiohttp response serialization, while its original value is retained in safe provenance. The
unrelated LiteLLM cache-price warning is filtered only for this explicitly uncached agent path;
network, timeout, rate-limit, fallback, and model-identity diagnostics remain visible.
General unemployment duration and model-generated displacement duration are tracked separately.
`unemployment_duration` feeds descriptive persistent-unemployment rates, while
`shock_unemployment_duration` drives precautionary behavior and shock-stress metrics so the
calibrated baseline unemployment stock is not silently treated as a new layoff shock.
`decision_behavior_summary.json` and `run_manifest.json` include a non-terminal
`behavior_qualification`: incomplete stress-bucket coverage or a role choosing only one action is
reported explicitly and must not be presented as validated LLM behavioral responsiveness. Qualification
is role-aware: deterministic `rule_only` roles remain visible in the audit but do not invalidate the
behavioral qualification of roles that actually used the LLM.

E5 and E6 use the common `passive_safety_net` before month 25 and switch to
`active_demand` at the shock. E3 is the isolated AI-rent-levy treatment and does
not activate the procurement-response channel.

Months 1--24 are an LLM-active pre-equilibrium qualification stage. The run checks the maximum
departure from the calibrated initial state across the full window and, by default, stops before
month 25 when the gate fails. `--allow-unstable-equilibrium` is diagnostic-only and should not be
used for confirmatory results. Each qualified run writes `pre_equilibrium_audit.json`,
`pre_equilibrium_state.json`, `decision_audit.jsonl`, `resolved_config.json`, and a source
fingerprint alongside the monthly metrics. `run_manifest.json` is the authoritative terminal
record for design, completion, dependency versions, configuration/source hashes, and decision-log
closure; finalized `SOCIETY_STEP.json` counters must agree with it.

To qualify one HKUST equilibrium and automatically fork E0, E1, and E5 from that exact month-24
state, use the counterfactual suite:

```bash
.venv/bin/python -m ai_economy_execution.counterfactual_suite \
  --population 500 --months 120 --seed 1 \
  --provider hkust --llm-roles resident,firm,government \
  --cognitive-regime R3 --result-stage pilot \
  --batch-size 16 --llm-max-workers 2 --llm-concurrency 2 \
  --max-fallback-rate 0.01 --max-role-fallback-rate 0.01 \
  --require-response-model-match
```

The suite pays for months 1--24 only once, rejects an unqualified equilibrium, then starts all
three branches at month 24. It writes `counterfactual_comparisons.json` and `.csv` for E1-E0,
E5-E1, and E5-E0. A single branch can also be resumed manually with
`run --initial-state PATH_TO/pre_equilibrium_state.json`; population, seed, LLM roles, provider,
model, pre-equilibrium configuration, and source fingerprint must match the checkpoint evidence.

Supported providers and key variables:

| Provider | Key variable | Default model |
|---|---|---|
| `hkust` | `HKUST_API_KEY` | `gpt-4` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `dashscope` | `DASHSCOPE_API_KEY` | `qwen-plus` |
| `moonshot` | `MOONSHOT_API_KEY` | `moonshot-v1-8k` |
| `openai` | `OPENAI_API_KEY` | `gpt-4.1-mini` |

The key is loaded from the environment or the workspace `.env`; it is never written to model output.
For a custom OpenAI-compatible endpoint, use `--provider custom --key-env NAME --api-base URL --model NAME`.

Before a future API run, validate the key mapping, URL, model, selected roles, and AgentSociety custom
module registration without sending a request:

```bash
.venv/bin/python -m ai_economy_execution.api_preflight \
  --provider hkust --llm-roles government
```

The command prints `network_called: false` and never prints the key. For `custom`, pass the same
`--key-env`, `--api-base`, and `--model` arguments that will be used by the real run. After this
preflight, start with a new output directory and a small pilot before a paid full-scale run. The
provider setup also forces LiteLLM to use its bundled local model-cost map, preventing an unrelated
price-table download during import.

## Experiments

Run the innovation-focused firm-culture and government-strategy experiment without an API:

```bash
.venv/bin/python -m ai_economy_execution.strategy_experiment \
  --population 500 --seeds 1-5 --months 120 \
  --output ai_economy_execution/results/research_matrix/auxiliary/strategy/culture_policy/N00500_M120_S1-5
```

This runner produces seven firm regimes (three homogeneous cultures and four mixed-culture competition
channel settings) × five policies. Each seed/regime treatment has its own matched E0 control. It saves
monthly paths by default and writes `control_runs.csv`, `strategy_runs.csv`, `strategy_matrix.csv`,
seed-paired and aggregate competition diagnostics, seed-paired and aggregate policy diagnostics,
`strategy_results.json`, `run_manifest.json`, source hashes, the resolved baseline config, and a concise
`analysis.md`. Use `--no-paths` only for disposable diagnostics. These are exploratory model comparisons;
the design intentionally excludes exports for now.

The current corrected smoke artifact is
`results/20260722_corrected_smoke_N500_seed1_v6/` (500 households, one seed, 120 months, 42 paths).
It verifies execution and output structure, not substantive robustness. The earlier five-seed culture-policy
directory used an invalid control/price-index design and is retained only for audit; do not cite it as a
current result.

Regenerate the standalone dark-editorial curve dashboard from the saved v6 monthly paths:

```bash
.venv/bin/python -m ai_economy_execution.visualize_key_curves
```

This writes `key_curves.html` and the disclosed normalized source payload `key_curves_data.json` into
the v6 result directory. It compares five policies under the mixed-culture/full-competition regime using
unemployment, month-24-indexed real consumption, firm count, and real government procurement. Three-month
trailing means are used except for the unsmoothed firm-count step series.

The general E0--E6 runner remains available for the original scenario design:

Use the numerical-core batch runner for parameter calibration and common-random-number comparisons:

```bash
.venv/bin/python -m ai_economy_execution.experiments \
  --populations 500,1000,5000 --seeds 1,2,3,4,5
```

It writes each monthly path and seed-paired scenario effects. Confirmatory runs should then use
`ai_economy_execution.run`, which executes every entity through AgentSociety workspaces and Ray tasks.

The experiment runner automatically produces:

- full monthly paths for the overall population, five income groups, bottom 60%, and bottom 80%;
- means, medians, quartiles, interquartile ranges, and paired bootstrap confidence intervals;
- disposable-income and real-consumption Atkinson indices for epsilon 0.5, 1.0, and 1.5;
- `runs.csv`, `comparisons.csv`, `aggregate_statistics.csv`, `run_gate_audit.csv`,
  `experiment_summary.json`, and `research_report.md`.

The run-gate audit checks monthly accounting and numeric boundaries, last-12-month warmup stability,
same-seed initial-state identity, pre-shock path gaps, and cross-seed systematic pretrends. Ordinary
experiment runs always retain the audit result; they do not abort solely because a small calibration
run fails a formal-scale threshold.

Seed ranges are accepted directly:

```bash
.venv/bin/python -m ai_economy_execution.experiments \
  --populations 500,1000,5000 --seeds 1-50 --bootstrap-samples 5000
```

## Formal sensitivity analysis

Run the registered one-at-a-time boundary design independently:

```bash
.venv/bin/python -m ai_economy_execution.sensitivity \
  --populations 500,1000,5000 --seeds 1-50 \
  --result-stage formal
```

It scans private-AI productivity, household and firm cash buffers, precautionary consumption,
labor adjustment, price pass-through, transfer response, and procurement response. The center point
is the main experiment; low and high boundaries are run with the same paired seeds and all E0-E6
scenarios. Outputs include every resolved variant config, `sensitivity_effects.csv`, and
`sensitivity_report.md`.

## Complete automatic study

The single formal offline entry point runs the main matrix, statistical report, and sensitivity
analysis without any LLM or external API call:

```bash
.venv/bin/python -m ai_economy_execution.full_study \
  --populations 500,1000,5000 --seeds 1-50 \
  --bootstrap-samples 5000 --result-stage formal
```

The output directory must be new or empty. The formal entry point stops before sensitivity analysis
when the main matrix does not pass its run gates and retains `run_gate_audit.csv` for diagnosis.
`study_manifest.json` records the source hash, resolved design, timestamps, gate result, completion
status, and output counts so a failed or partial study cannot be mistaken for a completed run.

## Tests

```bash
.venv/bin/python -m unittest discover -s ai_economy_execution/tests -v
```

The numerical test suite checks initialization anchors, population scaling, all seven scenarios, accounting
closure, fixed-basket prices, personal AI, targeted regulation, matched controls, complete artifact writing,
and bounded actions. AgentSociety custom-module validation uses the project `.venv`; because AgentSociety
requires `AGENTSOCIETY_LLM_API_KEY` even at import time, an inert placeholder may be supplied for this
offline registry test. The test itself makes no API request.
