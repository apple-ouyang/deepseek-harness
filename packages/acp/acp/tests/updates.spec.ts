import { describe, expect, it, vi } from 'vitest'
import type { Context } from '@deepseek-ai/cordis'
import { ToolCallId, MessageId } from '@deepseek-ai/dsh-llm'
import { SessionSeq, type Session, type SessionEvent } from '@deepseek-ai/dsh-session'
import { assistantUpdates, toolCallUpdate, toolResultUpdate, userMessageUpdates } from '../src/updates.ts'

/** Minimal committed assistant event for pure update projection tests. */
function assistantEvent(
  content: SessionEvent<'assistant/message'>['data']['message']['content'],
  usage?: SessionEvent<'assistant/message'>['data']['usage'],
): SessionEvent<'assistant/message'> {
  return {
    type: 'assistant/message',
    surfaceOp: 'append',
    seq: SessionSeq(0),
    time: 0,
    data: {
      stream: [],
      turn: 1,
      step: 1,
      message: {
        id: MessageId('message-1'),
        role: 'assistant',
        source: { kind: 'model', provider: 'mock', model: 'mock' },
        content,
      },
      ...usage === undefined ? {} : { usage },
    },
  }
}

/** Minimal committed tool-call event for kind/location/diff projection tests. */
function toolCallEvent(name: string, args: Record<string, unknown>): SessionEvent<'tool/call'> {
  return {
    type: 'tool/call',
    seq: SessionSeq(0),
    time: 0,
    data: { turn: 1, step: 1, callId: ToolCallId('call-1'), name, arguments: JSON.stringify(args) },
  }
}

/** Minimal committed tool-result event for result projection tests. */
function toolResultEvent(input: {
  isError: boolean
  content: SessionEvent<'tool/result'>['data']['message']['content']
  meta?: SessionEvent<'tool/result'>['data']['meta']
}): SessionEvent<'tool/result'> {
  return {
    type: 'tool/result',
    surfaceOp: 'append',
    seq: SessionSeq(0),
    time: 0,
    data: {
      turn: 1,
      step: 1,
      message: {
        id: MessageId('tool-message'),
        role: 'tool',
        toolCallId: ToolCallId('call-1'),
        isError: input.isError,
        source: { kind: 'tool', callId: ToolCallId('call-1') },
        content: input.content,
      },
      ...input.meta === undefined ? {} : { meta: input.meta },
    },
  }
}

describe('standard ACP update projection', () => {
  it('omits empty reasoning, unsupported assistant blocks, and absent usage', async () => {
    const ctx = { get: () => undefined } as unknown as Context
    const session = { requestContext: () => undefined } as unknown as Session
    const event = assistantEvent([
      { type: 'reasoning', text: '' },
      { type: 'tool-call', id: ToolCallId('call-hidden'), name: 'hidden', arguments: '{}' },
    ])

    await expect(assistantUpdates(ctx, session, event)).resolves.toEqual([])
  })

  it('requires both measured usage and context capacity', async () => {
    const meter = { measure: vi.fn(() => ({ totalTokens: 7 })) }
    const withMeter = { get: (name: string) => name === 'tokenMeter' ? meter : undefined } as unknown as Context
    const withoutMeter = { get: () => undefined } as unknown as Context
    const withCapacity = { requestContext: () => ({ contextWindow: 100 }) } as unknown as Session
    const withoutCapacity = { requestContext: () => undefined } as unknown as Session
    const event = assistantEvent([{ type: 'text', text: 'done' }], { inputTokens: 1, outputTokens: 1 })

    expect((await assistantUpdates(withMeter, withoutCapacity, event)).map(update => update.sessionUpdate))
      .toEqual(['agent_message_chunk'])
    expect((await assistantUpdates(withoutMeter, withCapacity, event)).map(update => update.sessionUpdate))
      .toEqual(['agent_message_chunk'])
    expect(meter.measure).not.toHaveBeenCalled()
  })

  it('preserves malformed tool input and projects a failed result without hidden content', async () => {
    const call = toolCallUpdate({
      type: 'tool/call',
      seq: SessionSeq(0),
      time: 0,
      data: { turn: 1, step: 1, callId: ToolCallId('call-bad'), name: 'broken', arguments: '{' },
    })
    const result = await toolResultUpdate({ get: () => undefined } as unknown as Context, {
      type: 'tool/result',
      surfaceOp: 'append',
      seq: SessionSeq(0),
      time: 0,
      data: {
        turn: 1,
        step: 1,
        message: {
          id: MessageId('tool-message'),
          role: 'tool',
          toolCallId: ToolCallId('call-bad'),
          isError: true,
          source: { kind: 'tool', callId: ToolCallId('call-bad') },
          content: [{ type: 'reasoning', text: 'hidden' }],
        },
      },
    })

    expect(call).toMatchObject({ rawInput: '{', kind: 'other' })
    expect(result).toEqual({
      sessionUpdate: 'tool_call_update',
      toolCallId: 'call-bad',
      status: 'failed',
    })
  })

  it('tags built-in tools with their ACP kind, location, and call-time diff', () => {
    const bash = toolCallUpdate(toolCallEvent('bash', { command: 'pnpm test', description: 'Run tests' }))
    expect(bash).toMatchObject({ kind: 'execute', title: 'bash' })
    expect('content' in bash && bash.content).toBeUndefined()

    const read = toolCallUpdate(toolCallEvent('read', { file_path: '/tmp/a.md', offset: 10 }))
    expect(read).toMatchObject({ kind: 'read', locations: [{ path: '/tmp/a.md' }] })

    const edit = toolCallUpdate(toolCallEvent('edit', {
      file_path: '/tmp/a.md', old_string: 'old', new_string: 'new',
    }))
    expect(edit).toMatchObject({
      kind: 'edit',
      locations: [{ path: '/tmp/a.md' }],
      content: [{ type: 'diff', path: '/tmp/a.md', oldText: 'old', newText: 'new' }],
    })

    const write = toolCallUpdate(toolCallEvent('write', { file_path: '/tmp/b.md', content: 'body' }))
    expect(write).toMatchObject({
      kind: 'edit',
      content: [{ type: 'diff', path: '/tmp/b.md', oldText: null, newText: 'body' }],
    })

    const grep = toolCallUpdate(toolCallEvent('grep', { pattern: 'x', path: '/tmp' }))
    expect(grep).toMatchObject({ kind: 'search', locations: [{ path: '/tmp' }] })

    const custom = toolCallUpdate(toolCallEvent('mcp__x__y', { command: 'z' }))
    expect(custom).toMatchObject({ kind: 'other' })
    expect('locations' in custom && custom.locations).toBeUndefined()
  })

  it('projects persisted presentation diffs and the failure reason', async () => {
    const ctx = { get: () => undefined } as unknown as Context
    const diffed = await toolResultUpdate(ctx, toolResultEvent({
      isError: false,
      content: [{ type: 'text', text: 'The file a.md has been updated successfully.' }],
      meta: { diffs: [{ path: '/tmp/a.md', oldText: 'old', newText: 'new' }] },
    }))
    expect(diffed).toEqual({
      sessionUpdate: 'tool_call_update',
      toolCallId: 'call-1',
      status: 'completed',
      content: [{ type: 'diff', path: '/tmp/a.md', oldText: 'old', newText: 'new' }],
    })

    const created = await toolResultUpdate(ctx, toolResultEvent({
      isError: false,
      content: [{ type: 'text', text: 'Created file' }],
      meta: { operation: 'create', diffs: [] },
    }), [{ type: 'diff', path: '/tmp/b.md', oldText: null, newText: 'body' }])
    expect(created).toMatchObject({
      status: 'completed',
      content: [{ type: 'diff', path: '/tmp/b.md', oldText: null, newText: 'body' }],
    })

    const failed = await toolResultUpdate(ctx, toolResultEvent({
      isError: true,
      content: [{ type: 'text', text: 'old_string was not found' }],
    }))
    expect(failed).toEqual({
      sessionUpdate: 'tool_call_update',
      toolCallId: 'call-1',
      status: 'failed',
      content: [{ type: 'content', content: { type: 'text', text: 'old_string was not found' } }],
      rawOutput: { message: 'old_string was not found' },
    })
  })

  it('projects human user text and skips plugin injects', async () => {
    const ctx = { get: () => undefined } as unknown as Context
    const human: SessionEvent<'user/message'> = {
      type: 'user/message',
      surfaceOp: 'append',
      seq: SessionSeq(0),
      time: 0,
      data: {
        id: MessageId('user-1'),
        role: 'user',
        source: { kind: 'user' },
        content: [
          { type: 'text', text: 'hello' },
          { type: 'text', text: '' },
        ],
      },
    }
    const injected: SessionEvent<'user/message'> = {
      type: 'user/message',
      surfaceOp: 'append',
      seq: SessionSeq(1),
      time: 0,
      data: {
        id: MessageId('user-plugin'),
        role: 'user',
        source: { kind: 'plugin', plugin: 'dsh-skill' },
        content: [{ type: 'text', text: 'AGENTS.md' }],
      },
    }

    await expect(userMessageUpdates(ctx, human)).resolves.toEqual([{
      sessionUpdate: 'user_message_chunk',
      messageId: 'user-1',
      content: { type: 'text', text: 'hello' },
    }])
    await expect(userMessageUpdates(ctx, injected)).resolves.toEqual([])
  })
})
