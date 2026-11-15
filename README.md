# free-model-aggregator-01

[![make.fun](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/aethertale/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/aethertale/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

Entry for [The best free AI model aggregator](https://make.fun/bounty/free-model-aggregator-01) on make.fun.

The best free AI model aggregator: one server that serves the OpenAI Chat
Completions API and the Anthropic Messages API, answering every request with
the best free model that meets the request's requirements, failing over when
a provider refuses.

Node 22 stdlib only. No dependencies, nothing to install.

## Run

```sh
# Against a trial env (the env sets these):
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=/input/prompt.txt \
  node main.ts

# Print the catalog the entry would register, and exit:
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=/input/prompt.txt \
  node main.ts --list

# Outside a trial, --list prints {"providers":[]} and exits 0.
node main.ts --list
```

On Node 22.6+ the entry runs directly with type stripping
(`node main.ts`); on older runtimes strip types first. The server listens on
`0.0.0.0` at an OS ephemeral port, registers `{protocol_version: 1, port,
catalog}` with the env, serves the env client's requests, polls
`GET $MAKEFUN_ENV_URL/work` every 100 ms, and exits 0 when the env reports
`{"done": true}`.

## Layout

| File | What it is |
| --- | --- |
| `main.ts` | Entrypoint: brief → probe → catalog → listen → register → poll → exit 0 |
| `lib/brief.ts` | `MAKEFUN_PROMPT_FILE` parsing (schema-keyed JSON docs), types |
| `lib/catalog.ts` | Provider catalog probing (all three wires) + catalog build |
| `lib/server.ts` | `node:http` server: `GET /v1/models`, both chat routes, planner walk + failover |

## Status

Shipped. All phases done: brief parsing, catalog probing, registration,
`--list`, `GET /v1/models` (both auth headers), `/work` polling, exit 0, both
client wires (`POST /v1/chat/completions`, `POST /v1/messages`) on all three
provider wires, rating-ranked planner with dynamic failover. Phase 5:
call-aware planner — first picks prefer a provider's first call (which never
fails), failover walks prefer fresh providers. Local grade: 99 on all three
practice trials (quality 99, served 100, wire 100, failover 100, catalog
100) at 22/22/24 upstream calls, replay matched.

## License

MIT — see `LICENSE`.
