# Band Mapping: Per-Satellite Canonical Slot Design

Source of truth: official competition data spec (verified against user-provided documentation).
Decision: 12-slot canonical mapping. Meteosat swap in dataset.py will be removed.

---

## Official Band Order

### Himawari AHI (16 bands)
| Index | Name | Wavelength | Physical meaning |
|-------|------|-----------|-----------------|
| 0  | B01 | 0.47 um | Blue visible |
| 1  | B02 | 0.51 um | Green visible |
| 2  | B03 | 0.64 um | Red visible |
| 3  | B04 | 0.86 um | NIR |
| 4  | B05 | 1.6 um  | SWIR |
| 5  | B06 | 2.3 um  | SWIR |
| 6  | B07 | 3.9 um  | Mid-IR |
| 7  | B08 | 6.2 um  | Upper WV |
| 8  | B09 | 6.9 um  | Mid WV |
| 9  | B10 | 7.3 um  | Lower WV |
| 10 | B11 | 8.6 um  | Cloud phase |
| 11 | B12 | 9.6 um  | Ozone |
| 12 | B13 | 10.4 um | IR window |
| 13 | B14 | 11.2 um | IR window 2 |
| 14 | B15 | 12.3 um | Split window |
| 15 | B16 | 13.3 um | CO2 |

### GOES ABI (16 bands)
| Index | Name | Wavelength | Physical meaning |
|-------|------|-----------|-----------------|
| 0  | C01 | 0.47 um  | Blue visible |
| 1  | C02 | 0.64 um  | Red visible |
| 2  | C03 | 0.865 um | NIR |
| 3  | C04 | 1.38 um  | Cirrus |
| 4  | C05 | 1.61 um  | SWIR |
| 5  | C06 | 2.25 um  | SWIR |
| 6  | C07 | 3.9 um   | Mid-IR |
| 7  | C08 | 6.185 um | Upper WV |
| 8  | C09 | 6.95 um  | Mid WV |
| 9  | C10 | 7.34 um  | Lower WV |
| 10 | C11 | 8.5 um   | Cloud phase |
| 11 | C12 | 9.61 um  | Ozone |
| 12 | C13 | 10.35 um | IR window |
| 13 | C14 | 11.2 um  | IR window 2 |
| 14 | C15 | 12.3 um  | Split window |
| 15 | C16 | 13.3 um  | CO2 |

### Meteosat SEVIRI (16 bands, raw TIF order)
| Index | Name   | Wavelength | Physical meaning |
|-------|--------|-----------|-----------------|
| 0  | vis_04 | ~0.4 um  | Visible |
| 1  | vis_05 | ~0.5 um  | Visible |
| 2  | vis_06 | ~0.6 um  | Red visible |
| 3  | vis_08 | ~0.8 um  | NIR |
| 4  | vis_09 | ~0.9 um  | NIR |
| 5  | nir_13 | ~1.3 um  | Cirrus-like NIR |
| 6  | nir_16 | ~1.6 um  | SWIR |
| 7  | nir_22 | ~2.2 um  | SWIR |
| 8  | ir_38  | 3.8 um   | Mid-IR |
| 9  | wv_63  | 6.3 um   | Upper WV |
| 10 | wv_73  | 7.3 um   | Lower WV |
| 11 | ir_87  | 8.7 um   | Cloud phase |
| 12 | ir_97  | 9.7 um   | Ozone |
| 13 | ir_105 | 10.5 um  | IR window |
| 14 | ir_123 | 12.3 um  | Split window |
| 15 | ir_133 | 13.3 um  | CO2 |

---

## Canonical 12-Slot Mapping (raw indices, no swap)

All indices are 0-based. Meteosat uses raw TIF order directly (swap removed).

| Slot | Wavelength    | rho    | Himawari | GOES   | Meteosat (raw) |
|------|--------------|--------|---------|--------|----------------|
| 0  | 0.64 um  Red  | +0.129 | idx 2   | idx 1  | idx 2          |
| 1  | 0.8 um   NIR  | low    | idx 3   | idx 2  | idx 3          |
| 2  | 1.6 um   SWIR | low    | idx 4   | idx 4  | idx 6          |
| 3  | 2.25 um  SWIR | low    | idx 5   | idx 5  | idx 7          |
| 4  | 3.9 um   Mid-IR | -0.289 | idx 6 | idx 6  | idx 8          |
| 5  | 6.2 um   WV up  | -0.221 | idx 7 | idx 7  | idx 9          |
| 6  | 7.3 um   WV low | low    | idx 9 | idx 9  | idx 10         |
| 7  | 8.6 um   cloud  | low    | idx 10| idx 10 | idx 11         |
| 8  | 9.7 um   ozone  | low    | idx 11| idx 11 | idx 12         |
| 9  | 10.4 um  IR win | -0.293 | idx 12| idx 12 | idx 13         |
| 10 | 12.3 um  split  | -0.287 | idx 14| idx 14 | idx 14         |
| 11 | 13.3 um  CO2    | low    | idx 15| idx 15 | idx 15         |

Input dimensions: 12 slots x 3 frames + 3 masks = 39 channels

---

## Canonical 18-Slot Mapping (raw indices, no swap)

Slots 0-11 identical to 12-slot above. Slots 12-17 are satellite-specific bands.
Missing slots filled with 0 for that satellite.

| Slot | Wavelength      | rho    | Himawari | GOES   | Meteosat (raw) |
|------|----------------|--------|---------|--------|----------------|
| 0  | 0.64 um  Red    | +0.129 | idx 2   | idx 1  | idx 2          |
| 1  | 0.8 um   NIR    | low    | idx 3   | idx 2  | idx 3          |
| 2  | 1.6 um   SWIR   | low    | idx 4   | idx 4  | idx 6          |
| 3  | 2.25 um  SWIR   | low    | idx 5   | idx 5  | idx 7          |
| 4  | 3.9 um   Mid-IR | -0.289 | idx 6   | idx 6  | idx 8          |
| 5  | 6.2 um   WV up  | -0.221 | idx 7   | idx 7  | idx 9          |
| 6  | 7.3 um   WV low | low    | idx 9   | idx 9  | idx 10         |
| 7  | 8.6 um   cloud  | low    | idx 10  | idx 10 | idx 11         |
| 8  | 9.7 um   ozone  | low    | idx 11  | idx 11 | idx 12         |
| 9  | 10.4 um  IR win | -0.293 | idx 12  | idx 12 | idx 13         |
| 10 | 12.3 um  split  | -0.287 | idx 14  | idx 14 | idx 14         |
| 11 | 13.3 um  CO2    | low    | idx 15  | idx 15 | idx 15         |
| 12 | 0.47 um  Blue   | low    | idx 0   | idx 0  | 0              |
| 13 | 0.5 um   Green  | low    | idx 1   | 0      | idx 1          |
| 14 | 0.9 um   NIR2   | low    | 0       | 0      | idx 4          |
| 15 | 1.38 um  Cirrus | low    | 0       | idx 3  | idx 5          |
| 16 | 6.9 um   WV mid | low    | idx 8   | idx 8  | 0              |
| 17 | 11.2 um  IR win2| low    | idx 13  | idx 13 | 0              |

Input dimensions: 18 slots x 3 frames + 3 masks = 57 channels

Notes:
- GOES has no green band: C02 (0.64um) is red, already in slot 0. Slot 13 = 0 for GOES.
- Meteosat has no 6.9um or 11.2um bands. Slots 16, 17 = 0 for Meteosat.
- Himawari has no 0.9um or 1.38um bands. Slots 14, 15 = 0 for Himawari.

---

## Satellite One-Hot

| Satellite | Vector    |
|-----------|-----------|
| himawari  | [1, 0, 0] |
| goes      | [0, 1, 0] |
| meteosat  | [0, 0, 1] |

Integrated into FiLM conditioning:
- Current time_feat (4-dim): [sin_day, cos_day, sin_hour, cos_hour]
- New cond_feat (7-dim):     [sin_day, cos_day, sin_hour, cos_hour, sat_0, sat_1, sat_2]
- FiLM MLP input_dim: 4 -> 7

---

## Implementation Checklist

- [ ] Add CANONICAL_BANDS dict to dataset.py (12-slot raw indices per satellite)
- [ ] Replace raw 16-band stack with 12-slot selection in __getitem__
- [ ] Remove Meteosat arr[[12,13]] swap (no longer needed)
- [ ] Update IN_CHANNELS: 51 -> 39
- [ ] Add satellite one-hot to time_feat in __getitem__ (4-dim -> 7-dim)
- [ ] Update FiLM MLP input_dim: 4 -> 7 in model.py
- [ ] Recompute stats.json with canonical 12 bands (mean/std per slot per satellite)
- [ ] Run smoke test locally before pushing to Vast.ai
