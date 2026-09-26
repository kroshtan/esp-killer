# RCON response fixtures

None of these come from a real server; nobody has published a complete raw player-data response. They are
written to match the format as documented by other Evrima tools' parsers, regexes, test fixtures and logs (see
NOTES.md, "RCON format evidence"), with made-up names and ids:

- `player_data_2025.txt`: 2025 builds. The first player is glued to the header (`PlayerDataName: ...`), classes
  are blueprint names (`BP_Carnotaurus_C`), no end marker.
- `player_data_2026.txt`, `player_data_2026_empty.txt`: 2026 builds. Header on its own line, `Gender`, mutation
  slot lists containing commas, bare class names, and a `PlayerDataEnd` marker.
- `player_data_expected.txt`, `player_data_variants.txt`, `player_data_broken.txt`, `player_data_empty.txt`:
  tolerance cases (unusual but plausible variants, and broken lines).

When `espk-agent capture` has been run against a real server, add its output here **with names and ids replaced
by fake ones**: captures are personal data and must not be committed as-is.
