# E2-E4 Institutional Experiment Design v2

## Evidence boundary

E2-E4 are exploratory institutional counterfactuals. They are not empirical
estimates of a real city and are not policy rankings until the same qualified
checkpoint design is repeated across population seeds, LLM draws, model
providers, population scales, and registered parameter sensitivities.

All branches start from one qualified month-24 state. The private-AI shock is
identical to E1. Each treatment changes one institutional mechanism:

| Branch | Treatment | Primary contrast |
| --- | --- | --- |
| E0 | No new private AI | E1 - E0 |
| E1 | Laissez-faire private AI | Reference |
| E2 | Employment responsibility, lower work intensity, and temporary wage-cost sharing | E2 - E1 |
| E3 | Graduated AI-rent levy and accelerated public return | E3 - E1 |
| E4 | AI time dividend and mixed-demand solo-enterprise formation | E4 - E1 |

Historical outputs that used the old E2-E4 meanings remain historical
artifacts. New manifests identify this design through the scenario names
`employment_preserving_ai_with_cost_sharing`,
`ai_rent_levy_with_immediate_recycling`, and
`ai_time_dividend_mixed_demand_solo_enterprise`.

## E2: employment-preserving AI responsibility

### Hypothesis

AI-related labor savings can be converted into lower work intensity rather
than layoffs when firms face an employment-responsibility constraint.

### Mechanism

For each firm and month the engine computes:

- AI labor demand using the realized AI multiplier and personal-AI
  productivity;
- a shadow labor demand using the same realized demand and price but removing
  AI productivity from the labor requirement;
- a protected floor equal to a registered share of the shadow labor demand.

The v2 primary value protects 100% of shadow employment. Monthly wages are
preserved while required hours and work intensity fall. Government temporarily
shares 25% of the wage cost of the jobs between AI labor demand and shadow
labor demand. The ordinary window lasts six months and can extend for six
months when registered loss or cash-distress warnings activate.

Before an otherwise eligible bankruptcy, a two-month restructuring window
allows subsidy, internal reassignment, and hours adjustment to operate. A firm
receives a layoff exemption only after the registered persistent distress or
loss threshold. Routine layoffs, exemption layoffs, and firm-exit layoffs are
reported separately.

### Required outputs

- AI and shadow labor demand;
- protected job floor and blocked AI-attributable layoffs;
- required work hours and work intensity;
- wage-cost sharing, ordinary layoffs, exemption layoffs, firm-exit jobs,
  restructuring months, profits, distress, and exits.

## E3: AI infrastructure levy and social return

### Hypothesis

Part of the AI-induced price advantage can be captured as a public
infrastructure charge and recycled into public services and public investment.

### Mechanism

The charge is based on a bounded AI productivity-rent base: modeled sales
multiplied by the realized AI productivity fraction, capped by positive
operating surplus. A basic-consumption allowance exempts 25% of that base.
The capture rate rises from 20% to 50% over 12 months and only 20% of the
modeled charge enters price-setting.

Seventy percent is earmarked for public services and 30% for public
investment. Up to 80% of a conservative current-revenue forecast may be
advanced for same-month recycling. A negative restricted-fund balance is
reported explicitly as a bridge advance rather than silently clipped.

The accounting identity is:

`cumulative levy revenue = cumulative earmarked spending + fund balance`.

Recommended sensitivities vary the initial/target capture rates, price
pass-through, basic-consumption exemption, and same-month advance share.

### Required outputs

- productivity-rent base, capture rate, unit price surcharge, and revenue;
- public-service and public-investment recycling;
- restricted fund balance and bridge advance;
- prices, consumption, firm profit, AI adoption, employment, and debt.

## E4: AI time dividend and solo-enterprise formation

### Hypothesis

AI-generated time savings can incubate new solo enterprises and transform some
wage employment into self-employment rather than unemployment.

### Mechanism

Wage workers accumulate entrepreneurial readiness from AI-saved hours.
Eligible residents voluntarily leave their employer after meeting registered
AI-use, readiness, cash, and market-capacity conditions. They cannot remain
simultaneously wage-employed in the primary design.

Solo-enterprise demand is decomposed into four registered channels:

- substitution from existing household and government orders;
- B2B services within existing firm-investment orders;
- induced household demand funded only from cash above the registered target;
- bounded external/platform demand, disclosed as an open-economy inflow.

Only induced and external demand count as net additional demand. Substitution
and B2B demand are reported as incumbent displacement. Operating costs and
income taxes are deducted. Persistently weak businesses exit into unemployment
and may later return to wage employment.

Labor-force accounting is:

`population = wage employed + self-employed + unemployed`.

### Required outputs

- saved hours and entrepreneurial readiness;
- voluntary wage exits, solo entries, solo exits, and survival;
- sales by demand source, incumbent displacement, net additional demand,
  net income, and taxes;
- wage employment, self-employment, total employment, and unemployment.

## Qualification gates

1. Months 1-24 must be identical across E0-E4 for all registered comparable
   fields.
2. Existing accounting, boundary, model-identity, and decision-audit gates must
   pass.
3. E2 must activate the registered floor and wage-cost sharing; every
   exemption and firm-exit layoff remains auditable.
4. E3 levy revenue must equal earmarked spending plus the signed restricted
   balance, with bridge advances disclosed.
5. E4 labor status and the four-channel solo-sales identity must close exactly;
   induced plus external sales must be positive in the primary smoke.

## Offline smoke

The network-free smoke uses 100 residents, seed 1, months 1-36:

```bash
.venv/bin/python -m ai_economy_execution.institutional_suite \
  --population 100 \
  --months 36 \
  --seed 1 \
  --provider offline \
  --llm-roles "" \
  --cognitive-regime R0 \
  --result-stage smoke
```

Do not launch a paid or formal LLM run until the smoke artifacts, API
preflight, output path, provider/model identity policy, seed matrix, and
sensitivity matrix have been reviewed and explicitly approved.
