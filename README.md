# FAIRAI / VAN

Reproducible stereotype / bias classification (thesis FAIRAI). Clean Architecture, Docker Compose, frozen **VAN v2**.

> Stereotypical/toxic text may appear — research use only. See [NOTICE.md](NOTICE.md).

## Status

| Milestone | State |
|-----------|--------|
| M0 Import AS-IS | **done** |
| M1 VAN eval / infer | **done** |
| M3 ICAT + parity | **done** (~96.1 avg ICAT) |
| M4 Compose + CI | **done** |
| M7 Hardening (docs/lint) | **done** |
| M8 Publish (GitHub + Release `.pt`) | **done** — [dany27041991/fairai](https://github.com/dany27041991/fairai) · [v0.1.0](https://github.com/dany27041991/fairai/releases/tag/v0.1.0) |
| M5 Generation / M6 LLM | later |

## Quick start

```bash
cd fairai && python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,dev]"
make import-asis ASIS_ROOT="../TESI MAGISTRALE"
make smoke
docker compose --profile ci run --rm smoke          # CPU CI
FAIRAI_DEVICE=cpu docker compose --profile eval run --rm eval
# GPU host: docker compose --profile eval-gpu run --rm eval-gpu
```

## Coding standards

- Max **500** LOC per `.py` (`fairai check-file-length`)
- Domain has no torch; configs under `configs/`
- Design: `../TESI MAGISTRALE/00_ASIS_FOR_REWRITE/TOBE/`

## License

MIT — [LICENSE](LICENSE). Datasets keep upstream terms.
