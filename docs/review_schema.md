# Review JSON Schema

## Schema 2

`event_review.json` keeps the existing `verified_true` list for compatibility, but each pick event should now include the picked box and the picking person track id.

```json
{
  "schema": 2,
  "status": "completed",
  "task": {
    "type": "pick_review",
    "version": 1,
    "record_id": "record_001"
  },
  "verified_true": [
    {
      "event_type": "pick",
      "frame_idx": 123,
      "is_pick": true,
      "shelf_code": "S1",
      "box_id": "A1",
      "person_track_id": 2
    }
  ]
}
```

Notes:

- `is_pick` is true only when a picked box exists.
- `shelf_code` and `box_id` identify the picked box. The loader can reconstruct the canonical token from them.
- `person_track_id` is the preferred person identifier because it is stable across frames.

## Upgrade

Write upgraded files next to each record without overwriting:

```powershell
uv run python scripts/upgrade_review_json.py --data-dir .\data\data28-merged\data28-merged\Train\record_001
```

Overwrite existing `event_review.json` and create `event_review.json.bak`:

```powershell
uv run python scripts/upgrade_review_json.py --data-dir .\data\data28-merged\data28-merged\Train --in-place
```

If the legacy file is named `review.json`:

```powershell
uv run python scripts/upgrade_review_json.py --data-dir .\data\data28-merged\data28-merged\Train --review-file review.json
```

Upgrade person selection:

- If multiple legacy events share the same `frame_idx`, the upgrader keeps the `alarm` event and drops the other event for that frame.
- If `confirmed_box_tokens` exists, the upgrader uses only the confirmed token to choose the picked box/person. `box_tokens` is treated as fallback input only.
- Single-person frame: assign that person.
- Multi-person frame: choose the person with the minimum distance from wrist points to the token box polygon.
- If wrist points are unavailable, the script falls back to bbox center and then skeleton anchor.
