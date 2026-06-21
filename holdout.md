# Holdout Location Analysis

## Current (old) holdout: `florida, france, jakarta, kinshasa`

| location | satellite | n |
|---|---|---|
| florida | goes | 1,440 |
| france | meteosat | 7,167 |
| jakarta | himawari | 1,488 |
| kinshasa | meteosat | 1,421 |
| **total** | | **11,516 / 40,686 = 28.3%** |

### Problems

1. **france dominates val**: 7,167 samples = ~62% of total holdout. Val RMSE is
   heavily biased toward meteosat temperate-Europe, not representative of full diversity.

2. **No arid/dry-climate location**: cape_town, bahia_blanca, gaza_province, borno_state
   are all in training. The model's performance on low-rainfall regimes is never validated.

3. **jakarta + kinshasa are both humid tropics**: two of four holdout slots go to similar
   climate types (high frequency, high intensity rain), wasting coverage.

4. **28.3% val is too large**: france's excess pulls 5,700 samples away from training
   that could otherwise be used productively.

---

## Proposed holdout: `florida, jakarta, cape_town, friuli_venezia_giulia`

| location | satellite | climate type | n |
|---|---|---|---|
| florida | goes | subtropical humid | 1,440 |
| jakarta | himawari | tropical SE Asia | 1,488 |
| cape_town | meteosat | arid / Mediterranean | 1,465 |
| friuli_venezia_giulia | meteosat | temperate Europe | 1,423 |
| **total** | | | **5,816 / 40,686 = 14.3%** |

### Why this is better

- **Balanced sample counts**: all four locations have ~1,400–1,500 samples; no single
  location dominates val RMSE.
- **Climate diversity**: covers subtropical (florida), tropical (jakarta), arid
  (cape_town), and temperate (friuli) — spans the full range seen in training data.
- **All three satellites represented**: goes=1, himawari=1, meteosat=2 (meteosat has
  the most total data at 17,222 so two holdout slots is proportionate).
- **More training data**: train set grows from 29,170 → 34,870 (+20%).

### Per-satellite holdout ratio

| satellite | total | holdout | % |
|---|---|---|---|
| goes | 10,272 | 1,440 | 14.0% |
| himawari | 13,192 | 1,488 | 11.3% |
| meteosat | 17,222 | 2,888 | 16.8% |
