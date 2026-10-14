import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { capturePatch, git } from '../src/evidence.ts'
import { Handoff } from '../src/handoff.ts'

test('patch includes committed, staged and untracked changes without changing the real index', async () => {
  const cwd = await mkdtemp(join(tmpdir(), 'conductor-test-'))
  await git(cwd, 'init')
  await git(cwd, 'config', 'user.email', 'test@example.com')
  await git(cwd, 'config', 'user.name', 'Test')
  await writeFile(join(cwd, 'tracked.txt'), 'before\n')
  await git(cwd, 'add', '.')
  await git(cwd, 'commit', '-m', 'initial')
  const base = await git(cwd, 'rev-parse', 'HEAD')
  await writeFile(join(cwd, 'tracked.txt'), 'committed\n')
  await git(cwd, 'commit', '-am', 'change')
  await writeFile(join(cwd, 'staged.txt'), 'staged\n')
  await git(cwd, 'add', 'staged.txt')
  await writeFile(join(cwd, 'new.txt'), 'new\n')
  const before = await readFile(join(cwd, '.git/index'))
  const patch = await capturePatch(cwd, base)
  assert.match(patch, /\+committed/)
  assert.match(patch, /\+staged/)
  assert.match(patch, /\+new/)
  assert.deepEqual(await readFile(join(cwd, '.git/index')), before)
})

test('reasoning items remain intact, are deduplicated, and carry recipient role context', () => {
  const handoff = new Handoff()
  const item = { type: 'reasoning', id: 'rs_1', encrypted_content: 'opaque-ciphertext', summary: [] }
  handoff.receive({ type: 'response.output_item.done', item })
  handoff.receive({
    type: 'response.completed',
    response: { output: [{ ...item, encrypted_content: 'different' }] },
  })
  const result = handoff.inject({ input: [] }, 'worker')
  assert.deepEqual(result.input[0], item)
  assert.match(result.input[1].content, /implementation worker/)
  assert.equal(
    handoff.inject(result, 'worker').input.filter((entry: any) => entry.type === 'reasoning').length,
    1,
  )
  assert.throws(() => handoff.inject({}, 'worker'), /Responses/)
})
