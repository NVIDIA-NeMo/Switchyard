import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { spawn } from 'node:child_process'
import { appendFile, mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { once } from 'node:events'
import { createHash } from 'node:crypto'
import { fileURLToPath } from 'node:url'
import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from '@earendil-works/pi-coding-agent'
import { conductor } from '../src/conductor.ts'
import { git, saveJson } from '../src/evidence.ts'

const output = await mkdtemp(join(tmpdir(), 'conductor-pi-smoke-'))
const live = process.argv.includes('--live')
const workerError = process.argv.includes('--worker-error')
const cli = process.argv.includes('--cli')
if (live && !process.env.SMOKE_UPSTREAM) throw new Error('Set SMOKE_UPSTREAM for a live Responses endpoint')
console.log(`Starting ${live ? 'live' : 'scripted'} smoke. Evidence: ${output}`)
const cwd = join(output, 'workspace')
await mkdir(cwd)
await git(cwd, 'init')
await git(cwd, 'config', 'user.name', 'Smoke test')
await git(cwd, 'config', 'user.email', 'smoke@example.com')
await writeFile(join(cwd, 'sum.mjs'), 'export const sum = (a, b) => a - b\n')
await writeFile(
  join(cwd, 'test.mjs'),
  "import assert from 'node:assert/strict'\nimport { sum } from './sum.mjs'\nassert.equal(sum(2, 3), 5)\nconsole.log('PASS')\n",
)
await git(cwd, 'add', '.')
await git(cwd, 'commit', '-m', 'fixture')

const upstreamRequests: any[] = []
let count = 0
let parent = 0
let child = 0
const upstream = createServer(async (req, res) => {
  try {
    let raw = ''
    for await (const part of req) raw += part
    const body = JSON.parse(raw)
    upstreamRequests.push(body)
    await appendFile(join(output, 'upstream-requests.jsonl'), JSON.stringify(body) + '\n', { mode: 0o600 })
    if (live) {
      body.model =
        body.model === 'capable' ? process.env.SMOKE_CAPABLE_MODEL : process.env.SMOKE_EFFICIENT_MODEL
      assert(body.model, 'Set SMOKE_CAPABLE_MODEL and SMOKE_EFFICIENT_MODEL')
      const upstreamResponse = await fetch(`${process.env.SMOKE_UPSTREAM}/responses`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${process.env.SMOKE_API_KEY ?? 'local'}`,
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(300000),
      })
      res.writeHead(upstreamResponse.status, {
        'content-type': upstreamResponse.headers.get('content-type') ?? 'text/event-stream',
      })
      for await (const chunk of upstreamResponse.body!) res.write(chunk)
      res.end()
      return
    }
    const worker = body.model === 'efficient'
    const index = worker ? ++child : ++parent
    assert.equal(body.reasoning?.effort, 'max')
    if (worker && workerError) {
      res.writeHead(400, { 'content-type': 'application/json' })
      res.end(
        JSON.stringify({
          error: { message: 'fixture worker is unavailable', type: 'invalid_request_error' },
        }),
      )
      return
    }
    const tool = (name: string, args: unknown) => ({
      type: 'function_call',
      id: `fc_${count}`,
      call_id: `call_${count}`,
      name,
      arguments: JSON.stringify(args),
      status: 'completed',
    })
    const message = (text: string) => ({
      type: 'message',
      id: `msg_${count}`,
      role: 'assistant',
      status: 'completed',
      content: [{ type: 'output_text', text, annotations: [] }],
    })
    count += 1
    let answer: any
    if (!worker) {
      assert.deepEqual(body.tools.map((tool: any) => tool.name).sort(), [
        'conductor_delegate',
        'find',
        'grep',
        'ls',
        'read',
      ])
      answer =
        index === 1
          ? tool('conductor_delegate', { instruction: 'Fix sum and run node test.mjs' })
          : index === 2
            ? tool('conductor_delegate', {
                instruction: 'Keep the fix. Add README.md explaining sum, then rerun node test.mjs',
              })
            : message('Implemented and verified both milestones.')
    } else {
      assert(!body.tools.some((tool: any) => tool.name === 'conductor_delegate'))
      answer =
        index === 1
          ? tool('write', { path: 'sum.mjs', content: 'export const sum = (a, b) => a + b\n' })
          : index === 2
            ? tool('bash', { command: 'node test.mjs' })
            : cli && index === 3
              ? message('First milestone complete')
              : index === (cli ? 4 : 3)
                ? tool('write', { path: 'README.md', content: 'sum adds two numbers.\n' })
                : index === (cli ? 5 : 4)
                  ? tool('bash', { command: 'node test.mjs' })
                  : message('Done')
    }
    const reasoning = {
      type: 'reasoning',
      id: `rs_${count}`,
      summary: [],
      encrypted_content: `opaque-${worker ? 'worker' : 'supervisor'}-${count}`,
    }
    const response = {
      id: `resp_${count}`,
      object: 'response',
      model: body.model,
      status: 'completed',
      output: [reasoning, answer],
      usage: {
        input_tokens: 100,
        output_tokens: 20,
        total_tokens: 120,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens_details: { reasoning_tokens: 10 },
      },
    }
    res.writeHead(200, { 'content-type': 'text/event-stream' })
    const emit = (type: string, data: any) =>
      res.write(`event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`)
    emit('response.created', { response: { ...response, status: 'in_progress', output: [] } })
    for (const [output_index, item] of response.output.entries()) {
      emit('response.output_item.added', {
        output_index,
        item: { ...item, arguments: item.type === 'function_call' ? '' : undefined },
      })
      emit('response.output_item.done', { output_index, item })
    }
    emit('response.completed', { response })
    res.end()
  } catch (error) {
    console.error('Smoke upstream:', String(error))
    await appendFile(
      join(output, 'upstream-errors.jsonl'),
      JSON.stringify({ time: new Date().toISOString(), error: String(error) }) + '\n',
    )
    if (!res.headersSent) {
      res.writeHead(502)
      res.end(String(error))
    } else res.destroy()
  }
})
upstream.listen(0, '127.0.0.1')
await once(upstream, 'listening')
const upstreamPort = (upstream.address() as { port: number }).port
const portServer = createServer()
portServer.listen(0, '127.0.0.1')
await once(portServer, 'listening')
const port = (portServer.address() as { port: number }).port
await new Promise<void>((resolve) => portServer.close(() => resolve()))
const config = `schema_version = 1
[llm_clients.fixture]
format = "openai_responses"
base_url = "http://127.0.0.1:${upstreamPort}/v1"
[targets.supervisor]
id = "capable"
llm_client = "fixture"
[targets.worker]
id = "efficient"
llm_client = "fixture"
[routes.switchyard]
id = "switchyard"
type = "passthrough"
target = "supervisor"
[routes.switchyard.subagents]
type = "passthrough"
target = "worker"
`
await writeFile(join(output, 'switchyard.toml'), config)
const server = spawn(
  process.env.SWITCHYARD_SERVER ?? 'switchyard-server',
  ['--config', join(output, 'switchyard.toml'), '--host', '127.0.0.1', '--port', String(port)],
  { stdio: ['ignore', 'pipe', 'pipe'] },
)
let serverLog = ''
server.stdout.on('data', (data) => {
  serverLog += data
})
server.stderr.on('data', (data) => {
  serverLog += data
})
const agentDir = join(output, 'parent')
const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } })
const runtime = await ModelRuntime.create({
  authPath: join(agentDir, 'auth.json'),
  modelsPath: null,
  modelsStorePath: join(agentDir, 'models.json'),
})
const loader = new DefaultResourceLoader({
  cwd,
  agentDir,
  settingsManager: settings,
  noExtensions: true,
  disabledBuiltinExtensions: ['mcp', 'codemode'],
  noSkills: true,
  noThemes: true,
  noPromptTemplates: true,
  noContextFiles: true,
  extensionFactories: [
    conductor({
      baseUrl: `http://127.0.0.1:${port}/v1`,
      output: join(output, 'evidence'),
      checkpointTurns: live ? 40 : 2,
    }),
  ],
})
let session: Awaited<ReturnType<typeof createAgentSession>>['session'] | undefined
try {
  let attempt = 0
  while (attempt < 40) {
    attempt += 1
    if (server.exitCode !== null) throw new Error(serverLog)
    try {
      await fetch(`http://127.0.0.1:${port}/health`)
      break
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
  }
  if (cli) {
    const process = spawn(
      'pi',
      [
        '--offline',
        '--no-extensions',
        '--no-mcp',
        '--no-context-files',
        '--no-skills',
        '--no-prompt-templates',
        '--no-themes',
        '-e',
        fileURLToPath(new URL('../index.ts', import.meta.url)),
        '--mode',
        'json',
        '-p',
        '/conductor Fix sum and test it, then add a README and test again in a second delegation.',
      ],
      {
        cwd,
        timeout: 45000,
        env: {
          ...globalThis.process.env,
          PI_CODING_AGENT_DIR: agentDir,
          CONDUCTOR_BASE_URL: `http://127.0.0.1:${port}/v1`,
          CONDUCTOR_OUTPUT_DIR: join(output, 'evidence'),
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    )
    let stdout = ''
    let stderr = ''
    process.stdout.on('data', (data) => {
      stdout += data
    })
    process.stderr.on('data', (data) => {
      stderr += data
    })
    const [code] = await once(process, 'exit')
    await writeFile(join(output, 'pi-cli.jsonl'), stdout)
    assert.equal(code, 0, stderr)
  } else {
    await loader.reload()
    const created = await createAgentSession({
      cwd,
      agentDir,
      resourceLoader: loader,
      modelRuntime: runtime,
      settingsManager: settings,
      sessionManager: SessionManager.create(cwd, join(agentDir, 'sessions')),
    })
    session = created.session
    await session.bindExtensions({ onError: (error) => console.error('Extension:', error) })
    session.subscribe((event) => {
      if (
        event.type === 'message_end' &&
        event.message.role === 'assistant' &&
        event.message.stopReason === 'error'
      )
        console.error(event.message.errorMessage)
    })
    const settled = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Smoke timed out')), live ? 720000 : 45000)
      session!.subscribe((event) => {
        if (event.type === 'agent_settled') {
          clearTimeout(timer)
          resolve()
        }
      })
    })
    await session.prompt(
      '/conductor Fix sum and test it, then add a README and test again in a second delegation.',
    )
    await settled
  }
  await saveJson(join(output, 'upstream.json'), upstreamRequests)
  if (workerError) {
    assert.equal(parent, 1)
    assert.equal(child, 1)
    const failure = JSON.parse(await readFile(join(output, 'evidence/failure.json'), 'utf8'))
    assert.match(failure.message, /fixture worker is unavailable/)
    assert.equal(await readFile(join(cwd, 'sum.mjs'), 'utf8'), 'export const sum = (a, b) => a - b\n')
    await saveJson(join(output, 'result.json'), { passed: true, workerError: true, requestsStopped: true })
    console.log(`PASS: worker error stopped parent without retries. Evidence: ${output}`)
  } else if (live) {
    const code = await readFile(join(cwd, 'sum.mjs'), 'utf8')
    const mod = await import(join(cwd, 'sum.mjs'))
    assert.equal(mod.sum(2, 3), 5)
    assert.equal(mod.sum(-3, 8), 5)
    assert((await readFile(join(cwd, 'README.md'), 'utf8')).length > 0)
    const handoffs = JSON.parse(await readFile(join(output, 'evidence/handoffs.json'), 'utf8'))
    assert(handoffs.rounds >= 2)
    assert(handoffs.supervisor.length > 0)
    assert(handoffs.worker.length > 0)
    const usage = (await readFile(join(output, 'evidence/usage.jsonl'), 'utf8'))
      .trim()
      .split('\n')
      .map((row) => JSON.parse(row))
    assert.equal(usage.length, upstreamRequests.length)
    assert(usage.every((row) => row.usage?.input_tokens > 0))
    const parentRequests = upstreamRequests.filter((row) => row.model === process.env.SMOKE_CAPABLE_MODEL)
    const workerRequests = upstreamRequests.filter((row) => row.model === process.env.SMOKE_EFFICIENT_MODEL)
    assert(parentRequests.length > 0 && workerRequests.length > 0)
    const parentIds = new Set(handoffs.supervisor.map((row: any) => row.id))
    const workerIds = new Set(handoffs.worker.map((row: any) => row.id))
    const hashes = new Map(
      [...handoffs.supervisor, ...handoffs.worker].map((row: any) => [row.id, row.sha256]),
    )
    for (const row of upstreamRequests) {
      assert.equal(row.reasoning.effort, 'max')
      for (const item of row.input) {
        if (hashes.has(item.id))
          assert.equal(createHash('sha256').update(item.encrypted_content).digest('hex'), hashes.get(item.id))
      }
    }
    assert(workerRequests.some((row) => row.input.some((item: any) => parentIds.has(item.id))))
    assert(parentRequests.some((row) => row.input.some((item: any) => workerIds.has(item.id))))
    await saveJson(join(output, 'result.json'), {
      passed: true,
      live: true,
      parentRequests: parentRequests.length,
      workerRequests: workerRequests.length,
      checkpoints: handoffs.rounds,
      reasoningBothDirections: true,
      usageRequests: usage.length,
      code,
    })
    console.log(`PASS: live Pi + Switchyard. Evidence: ${output}`)
  } else {
    assert.equal(parent, 3)
    assert.equal(child, cli ? 6 : 4)
    const handoffs = JSON.parse(await readFile(join(output, 'evidence/handoffs.json'), 'utf8'))
    assert.equal(handoffs.rounds, 2)
    const first = await readFile(join(output, 'evidence/checkpoint-1.patch'), 'utf8')
    const second = await readFile(join(output, 'evidence/checkpoint-2.patch'), 'utf8')
    assert.match(first, /a \+ b/)
    assert(!first.includes('README'))
    assert.match(second, /a \+ b/)
    assert.match(second, /README/)
    const workerRequests = upstreamRequests.filter((row) => row.model === 'efficient')
    assert(JSON.stringify(workerRequests[2]).includes('a + b'))
    assert(workerRequests[0].input.some((item: any) => item.encrypted_content === 'opaque-supervisor-1'))
    assert(
      upstreamRequests
        .filter((row) => row.model === 'capable')[1]
        .input.some((item: any) => item.encrypted_content?.startsWith('opaque-worker-')),
    )
    const usage = (await readFile(join(output, 'evidence/usage.jsonl'), 'utf8'))
      .trim()
      .split('\n')
      .map((row) => JSON.parse(row))
    assert.equal(usage.length, cli ? 9 : 7)
    assert(usage.every((row) => row.usage?.input_tokens === 100))
    await saveJson(join(output, 'result.json'), {
      passed: true,
      parentRequests: parent,
      workerRequests: child,
      checkpoints: 2,
      reasoningBothDirections: true,
      usageRequests: usage.length,
    })
    console.log(`PASS: real Pi + real Switchyard, scripted upstream. Evidence: ${output}`)
  }
} finally {
  await session?.abort()
  session?.dispose()
  server.kill('SIGTERM')
  upstream.closeAllConnections()
  upstream.close()
  await writeFile(join(output, 'switchyard.log'), serverLog)
}
