import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const exec = promisify(execFile)

export async function git(cwd: string, ...args: string[]) {
  return (await exec('git', args, { cwd, maxBuffer: 16 * 1024 * 1024 })).stdout.trimEnd()
}

export async function capturePatch(cwd: string, base: string) {
  const root = await git(cwd, 'rev-parse', '--show-toplevel')
  const scratch = await mkdtemp(join(tmpdir(), 'conductor-index-'))
  const env = { ...process.env, GIT_INDEX_FILE: join(scratch, 'index') }
  const run = (...args: string[]) => exec('git', args, { cwd: root, env, maxBuffer: 16 * 1024 * 1024 })
  try {
    await run('read-tree', 'HEAD')
    await run('add', '-A', '--', '.')
    return (await run('diff', '--cached', '--binary', base, '--')).stdout
  } finally {
    await rm(scratch, { recursive: true, force: true })
  }
}

export async function saveJson(path: string, data: unknown) {
  await writeFile(path, JSON.stringify(data, null, 2) + '\n', { mode: 0o600 })
}

export async function checkpoint(cwd: string, base: string, output: string, index: number, tools: unknown[]) {
  await mkdir(output, { recursive: true, mode: 0o700 })
  const patch = await capturePatch(cwd, base)
  const path = join(output, `checkpoint-${index}.patch`)
  await writeFile(path, patch, { mode: 0o600 })
  await saveJson(join(output, `checkpoint-${index}-tools.json`), tools)
  return { path, patch }
}
