# Phone photos: protocol fixed before the run

Written 2026-10-22T08:05:56Z before the agent saw any of these photos and before the runner read the labels.

- Photos: 24 product photos shot on the author's phone on 22 Oct 2026, sent through Telegram
  (so recompressed to 960x1280 JPEG; EXIF and GPS absent, checked: 0 tags on every file).
- Labels: OK / BAD per photo, written by the author's AI assistant from looking at the photos, before
  any run. Not an independent annotator. sha256 of LABELS.md: `346676c2d8345ffc9dddc2db9cefab7000fe0f89470b0b526d398f286d0a46c5`
- sha256 of all 24 photo bytes concatenated in name order: `bd23d15291eb982d896b3d9b7c3b0923281980f57a7ffd5d413f946c2f122768`
- Agent code: commit `2ab02a1` (thresholds unchanged since the 11 Oct submission); runner sha256 `31d4bc39669519282b19a621e4cff94e55ca1f5d2387afafa97868cd5e72ad09`.
- One run only. Results are published as they come, including failures.

Scoring, decided now:
- A BAD photo is "stopped" if the agent says reshoot or human. auto_fixed on a BAD photo is reported
  separately (it may or may not be a real fix) and does NOT count as stopped. accept on BAD = missed.
- An OK photo "passes" if the agent says accept or auto_fixed; human or reshoot on OK = false alarm.
- 95 % Wilson intervals for every share. Open-loop comparison is not run on this set.
