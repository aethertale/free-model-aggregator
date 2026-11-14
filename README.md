# free-model-aggregator-01

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
| `lib/server.ts` | `node:http` server: `GET /v1/models`, 501 stubs for the two chat routes |

## Status

Phase 1: brief parsing, catalog probing, registration, `--list`,
`GET /v1/models` (both auth headers), `/work` polling, exit 0. The two
client wires (`POST /v1/chat/completions`, `POST /v1/messages`) answer 501
until Phase 2; provider model-call clients and the planner land in Phase 3.

## License

MIT — see `LICENSE`.
