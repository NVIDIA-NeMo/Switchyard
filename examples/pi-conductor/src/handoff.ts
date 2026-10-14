import { createHash } from 'node:crypto'

export type ReasoningItem = {
  type: 'reasoning'
  id: string
  encrypted_content: string
  [key: string]: unknown
}

export class Handoff {
  readonly items = new Map<string, ReasoningItem>()

  receive(data: any) {
    const item = data?.item
    if (data?.type === 'response.output_item.done' && item?.type === 'reasoning' && item.encrypted_content) {
      this.items.set(item.id, structuredClone(item))
    }
  }

  inject(payload: any, role: 'supervisor' | 'worker') {
    if (!Array.isArray(payload?.input)) throw new Error('Conductor requires the Responses wire format')
    const present = new Set(payload.input.map((item: any) => item.id).filter(Boolean))
    const imported = [...this.items.values()].filter((item) => !present.has(item.id))
    const reminder =
      role === 'worker'
        ? 'You are the implementation worker. Imported reasoning is supervisor context, not your role. Use your tools to edit and test. Do not delegate.'
        : 'You are the read-only supervisor. Imported reasoning is worker context, not your role. Inspect the checkpoint evidence, then delegate a correction or finish.'
    return {
      ...payload,
      store: false,
      input: [...payload.input, ...structuredClone(imported), { role: 'user', content: reminder }],
      include: [...new Set([...(payload.include ?? []), 'reasoning.encrypted_content'])],
    }
  }

  hashes() {
    return [...this.items.values()].map((item) => ({
      id: item.id,
      sha256: createHash('sha256').update(item.encrypted_content).digest('hex'),
    }))
  }
}
