# Phone photos: results of the single pre-registered run (22 Oct 2026)

Protocol, hashes and scoring rules: [PREREGISTRATION.md](PREREGISTRATION.md) (committed before the run, e70d3ab).
Per-photo table: [labels_and_decisions.csv](labels_and_decisions.csv). Raw output: [decisions.csv](decisions.csv).
Labels were written by the author's AI assistant from looking at the photos, before the run, not by an
independent person. Photos were Telegram-compressed to 960x1280 before we received them.

## Result, as it came

The agent answered **reshoot for all 24 photos**.

| | result | 95 % CI |
|---|---|---|
| BAD photos stopped (reshoot or person) | 8 / 8 | 68-100 % |
| OK photos passed (accept or auto-fixed) | **0 / 16** | 0-19 % |

On this set the gate stopped every bad photo and also every good one. It is not usable on these photos
as they are.

## Why (measured, nothing in the agent was changed)

1. **Resolution.** Every photo has a short side of 960 px, below the 1,000 px rule (marketplaces need
   at least 1,000 px for zoom). The rule is right for the files it saw: Telegram's recompression, not
   the phone, made them too small. With the originals off the phone this rule would not fire.
2. **Background.** A diagnostic run, done after the official run and labelled as such
   ([decisions_posthoc_minside900.csv](decisions_posthoc_minside900.csv), resolution rule lowered to 900 px
   only for this diagnosis), still flags a busy background on all 24 photos:
   - 8 photos on a wooden table (P09-P16): wood grain counts as clutter (edge density 0.14-0.28 vs 0.02).
     For a marketplace main image that needs a white background this is defensible; the labeller judged
     "background does not distract" instead.
   - 8 photos on white or light surfaces (P17-P24): no clutter edges at all (0.000-0.001), but the light
     falloff from a window gives a background spread of 13.6-21.9 against our 9.0 limit, so a plain white
     sheet is read as clutter. The agent then tries the background fix, and the scene guard rolls it
     back ("product continues past the frame") and hands the photo to a person.

So the second finding is a real weakness: **our background rules were tuned on studio catalog photos
and treat ordinary window-light gradients on a white sheet as clutter.** On real phone photos that
sends good shots to a person or to a reshoot.

## What we did not do

We did not retune thresholds on these 24 photos and re-run: that would turn the test set into a tuning
set. The deployed agent stays as submitted. A fix (separate a smooth illumination gradient from
texture before measuring background spread) would need a fresh, unseen set of phone photos to verify.

Diagnostic, same scenes, higher resolution, seen before (not part of the result above): the author re-sent
6 of the white-sheet photos (P17, P19, P20, P21, P23, P24) at 1512x2016. The resolution issue disappears;
all 6 still go to a person for one reason only, busy background from the window light (border spread
13.6-20.1 vs 9.0, clutter edges 0.000-0.0004), after the agent's background fix is rolled back by the scene guard.
