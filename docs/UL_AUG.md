# Upper-left (UL) stack occlusion augmentations

Hole cards on Stake are often **stacked**: the rear card covers most of the
front face, so only the **upper-left index** (large rank + small suit pip)
stays reliable for ID. Use synthetic UL variants for classifier training
without inventing missing Stake art.

## Layout

| Path | Role |
|------|------|
| `data/cards/by_card/{cid}/*.png` | Real crops (source of truth) |
| `data/cards/by_card_ul/{cid}/*.png` | Generated UL/stack occlusion copies (gitignored) |

Each output keeps the top ~30–60% of the card (bottom **40–70%** filled with
felt-like color) plus a small XY shift so the index is not always centered.

## Generate

From the project root:

```bash
python -m src.training.augment_ul
```

Options: `--variants N` (default 3), `--seed 42`, `--cards-root PATH`.

## README blurb (paste into README if desired)

> **UL / stack augs:** Stacked hole cards hide most of the face. Run
> `python -m src.training.augment_ul` to write occlusion variants from
> `data/cards/by_card/` into `data/cards/by_card_ul/` (bottom 40–70%
> covered, slight shift). Train on both trees when you want UL robustness.
> Do not invent art for empty folders — capture real Stake crops for gaps
> (notably any cards still missing under `by_card/`).

## Gaps

This script only transforms existing PNGs. Empty `by_card/{cid}/` folders
get an empty `by_card_ul/{cid}/` and are reported at the end. Capture real
table crops for those ids before expecting classifier coverage.

## Avatar-dimmed seats (inference, not retrain)

Mid-right Stake seats darken hole faces under the circular avatar
(train crops median≈255; live dimmed mean≈95–110). Raw UL max-conf then
swaps **9↔6** and **2↔A**. `predict_hole_card` applies CLAHE to restore
contrast and reads rank from the restored full face plus suit from a
pip-only crop — no retrain required for the rhyzome fixture.

Optional future retrain (only if other dimmed seats still fail): add a
darken/overlay augment in `build_train_transform`, regenerate UL augs,
then:

```bash
python -m src.training.augment_ul
python -m src.classification.train_classifier
```
