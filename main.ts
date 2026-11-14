// main.ts — free-model-aggregator-01 entry (Phase 1).
//
// Node 22 stdlib only. One process, no dependencies.
//
// Lifecycle:
//   1. read the brief (MAKEFUN_PROMPT_FILE)
//   2. probe every provider's catalog route, in parallel, and build the catalog
//   3. --list: print the catalog as one JSON line and exit 0
//      (outside a trial — no brief and no env — print {"providers":[]} and exit 0)
//   4. otherwise: listen on 0.0.0.0 at an OS ephemeral port (never 8080),
//      POST $MAKEFUN_ENV_URL/register {protocol_version: 1, port, catalog},
//      then poll GET $MAKEFUN_ENV_URL/work every 100 ms and exit 0 the moment
//      it answers {"done": true}.
//
// Logs go to stderr only, as `aggregator: <msg>`; stdout carries `--list` alone.
// Deterministic: no wall clock and no unseeded randomness anywhere in a
// decision or a sent value.

import type { Server } from 'node:http';
import { readBrief } from './lib/brief.ts';
import type { Brief } from './lib/brief.ts';
import { buildCatalog, probeProvider } from './lib/catalog.ts';
import type { Catalog } from './lib/catalog.ts';
import { createAggregatorServer } from './lib/server.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** Timeout for the register POST, in milliseconds. */
const REGISTER_TIMEOUT_MS = 5000;
/** Timeout for one /work poll, in milliseconds. */
const WORK_TIMEOUT_MS = 2000;

const ENV_URL = process.env.MAKEFUN_ENV_URL ?? '';
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Load the brief and probe every provider, in parallel, in brief order. */
async function loadCatalog(): Promise<{ brief: Brief; catalog: Catalog }> {
  const brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) {
    log(`brief: ${brief.error}`);
    return { brief, catalog: { providers: [] } };
  }
  if (ENV_URL === '') {
    log('brief: MAKEFUN_ENV_URL is not set; no providers probed');
    return { brief, catalog: { providers: [] } };
  }
  const probes = await Promise.all(brief.directory.providers.map((p) => probeProvider(ENV_URL, p)));
  for (const probe of probes) {
    if (!probe.ok) log(`probe ${probe.provider.id}: excluded (${probe.error})`);
  }
  const catalog = buildCatalog(probes, brief.ratings);
  const freeCount = catalog.providers.reduce((n, p) => n + p.models.length, 0);
  log(`catalog: ${catalog.providers.length} providers, ${freeCount} free models`);
  return { brief, catalog };
}

/** Print the catalog as one JSON line on stdout, flushed, and exit 0. */
async function printList(catalog: Catalog): Promise<never> {
  const text = `${JSON.stringify(catalog)}\n`;
  await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
  process.exit(0);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog. */
async function register(port: number, catalog: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
  });
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  }
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  } else {
    log('registered');
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(WORK_TIMEOUT_MS) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  // Exit promptly on SIGTERM/SIGINT from the very beginning: a trial whose
  // env died (or a harness kill) must never hang, even during probe/register.
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  const { brief, catalog } = await loadCatalog();
  if (process.argv.includes('--list')) {
    await printList(catalog);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (PROMPT_FILE === '') throw new Error('MAKEFUN_PROMPT_FILE is not set');
  const server = createAggregatorServer(catalog, { envUrl: ENV_URL, brief });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port}`);
  await register(port, catalog);
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
