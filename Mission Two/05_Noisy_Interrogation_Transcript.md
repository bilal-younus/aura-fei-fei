# Ten-turn noisy conversation

Deterministic FST, not LLM-generated dialogue
Words are counted by whitespace. Wire messages include mandatory substituted noise.

| Turn | Speaker | Received wire message | Decoded payload | Words |
|---|---|---|---|---|
| 1 | Cadet | Sirius Sirius Banana | Sirius | 3 |
| 2 | AURA | Green Speed Speed please please please | Speed please | 6 |
| 3 | Cadet | warp warp Labyrinth 7 7 7 | warp 7 | 6 |
| 4 | AURA | Clearance Labyrinth Clearance please please please | Clearance please | 6 |
| 5 | Cadet | ALPHA-7 ALPHA-7 Banana | ALPHA-7 | 3 |
| 6 | AURA | Confirm Confirm Confirm Sirius/7/ALPHA-7 Sirius/7/ALPHA-7 Green | Confirm Sirius/7/ALPHA-7 | 6 |
| 7 | Cadet | confirm confirm Banana | confirm | 3 |
| 8 | AURA | Navigation Navigation Navigation authorized Midnight authorized | Navigation authorized | 6 |
| 9 | Cadet | Banana status status | status | 3 |
| 10 | AURA | Labyrinth Session Session closed closed closed | Session closed | 6 |

## FST state trace

- `INIT -> COLLECT_SLOTS`: Sirius / Speed please; slots={'destination': 'Sirius'}
- `COLLECT_SLOTS -> COLLECT_SLOTS`: warp 7 / Clearance please; slots={'destination': 'Sirius', 'speed': '7'}
- `COLLECT_SLOTS -> CONFIRMATION`: ALPHA-7 / Confirm Sirius/7/ALPHA-7; slots={'destination': 'Sirius', 'speed': '7', 'clearance_code': 'ALPHA-7'}
- `CONFIRMATION -> TERMINAL`: confirm / Navigation authorized; slots={'destination': 'Sirius', 'speed': '7', 'clearance_code': 'ALPHA-7'}
- `TERMINAL -> TERMINAL`: status / Session closed; slots={'destination': 'Sirius', 'speed': '7', 'clearance_code': 'ALPHA-7'}
