/**
 * Provider-subagent projection for ACP clients.
 *
 * The bridge publishes only its root Agent as an ACP session, but in-process
 * delegation composes child sessions beneath that root and their committed
 * events reach the same `session/event` firehose. This module folds descendant
 * events into the provider-owned subagent vocabulary an ACP client can render
 * beside its own managed sessions: one descriptor per child, a read-only
 * timeline, and a lifecycle status.
 *
 * The projection is deliberately additive on the wire: it travels in an ACP
 * extension notification, so a client that does not know the method ignores
 * the whole notification and the standard surface stays unchanged.
 *
 * @module
 */

import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { Session, SessionEvent, SessionId } from '@deepseek-ai/dsh-session'

/**
 * ACP extension-notification method carrying this bridge's provider-subagent
 * input events. One notification carries an ordered batch for one root session.
 */
export const SUBAGENT_UPDATE_METHOD = '_dsh/sdk/subagent/update'

/** Lifecycle status understood by the client's provider-subagent surface. */
export type ProviderSubagentStatus = 'running' | 'completed' | 'failed' | 'canceled'

/** One read-only timeline item for a provider-owned child. */
export type ProviderSubagentTimelineItem =
  | { type: 'user_message'; text: string }
  | { type: 'assistant_message'; text: string; messageId?: string }
  | { type: 'reasoning'; text: string }
  | {
    type: 'tool_call'
    callId: string
    name: string
    detail: { type: 'plain_text'; label?: string; text?: string }
    status: 'running' | 'completed' | 'failed'
    error: unknown
    metadata: Record<string, unknown>
  }

/**
 * One descriptor, timeline, or removal instruction for a provider-owned child.
 * An `upsert` omitting `status` preserves the status the client already holds.
 */
export type ProviderSubagentEvent =
  | {
    type: 'upsert'
    id: string
    title?: string
    description?: string
    status?: ProviderSubagentStatus
    parentSubagentId?: string
    toolCallId?: string
    cwd?: string
  }
  | { type: 'timeline'; id: string; item: ProviderSubagentTimelineItem }
  | { type: 'remove'; id: string }

/** Per-child facts later events must recall to stay well formed. */
interface ChildState {
  /** Whether a display title was already derived from the delegated prompt. */
  titled: boolean
  /** Direct provider-subagent parent for a nested child; absent for a direct child. */
  parentSubagentId: string | undefined
  /** Call id to tool name, because a tool result restates the call without its name. */
  toolNames: Map<string, string>
}

/** Bound one derived label so a delegated prompt cannot dominate the track row. */
const MAX_TITLE_CHARS = 120

/** Join the plain-text blocks one content array carries. */
function textOf(content: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of content) {
    if (block.type === 'text' && block.text.length > 0) parts.push(block.text)
  }
  return parts.join('\n')
}

/** Collapse text to one bounded display line. */
function labelOf(text: string): string {
  const line = text.trim().split('\n', 1)[0] ?? ''
  return line.length > MAX_TITLE_CHARS ? `${line.slice(0, MAX_TITLE_CHARS - 1)}…` : line
}

/** Map one durable turn ending onto the client's coarser lifecycle status. */
function turnEndStatus(kind: string): ProviderSubagentStatus {
  if (kind === 'error') return 'failed'
  if (kind === 'aborted' || kind === 'interrupted') return 'canceled'
  return 'completed'
}

/** One completed or failed tool result as a timeline item. */
function toolResultItem(
  callId: string,
  name: string,
  text: string,
  failure: string | undefined,
): ProviderSubagentTimelineItem {
  return {
    type: 'tool_call',
    callId,
    name,
    detail: { type: 'plain_text', label: name, ...text.length === 0 ? {} : { text } },
    status: failure === undefined ? 'completed' : 'failed',
    error: failure ?? null,
    metadata: {},
  }
}

/**
 * Fold descendant session events into provider-subagent input events.
 *
 * One tracker belongs to one root ACP session. It keeps only the per-child
 * facts that later events must recall, so the projection stays proportional to
 * the live children rather than to their transcript volume.
 */
export class AcpSubagentTracker {
  private readonly children = new Map<SessionId, ChildState>()

  /**
   * Project one descendant event.
   * @param session - the descendant session that owns the event.
   * @param event - the committed event observed on the session firehose.
   * @returns ordered input events for the client; empty when nothing is reportable.
   */
  observe(session: Session, event: SessionEvent): ProviderSubagentEvent[] {
    const id = session.header.id
    const events: ProviderSubagentEvent[] = []
    let state = this.children.get(id)
    if (state === undefined) {
      const depth = session.header.delegationDepth ?? 0
      const parentSubagentId = depth > 1 ? session.header.parentSession : undefined
      state = { titled: false, parentSubagentId, toolNames: new Map() }
      this.children.set(id, state)
      events.push({
        type: 'upsert',
        id,
        status: 'running',
        ...parentSubagentId === undefined ? {} : { parentSubagentId },
        ...session.header.cwd === undefined ? {} : { cwd: session.header.cwd },
      })
    }

    switch (event.type) {
      case 'turn/start':
        events.push({ type: 'upsert', id, status: 'running' })
        break
      case 'turn/end':
        events.push({ type: 'upsert', id, status: turnEndStatus(event.data.reason.kind) })
        break
      case 'user/message': {
        if (event.data.source.kind !== 'user') break
        const text = textOf(event.data.content)
        if (text.length === 0) break
        if (!state.titled) {
          state.titled = true
          const label = labelOf(text)
          if (label.length > 0) events.push({ type: 'upsert', id, title: label, description: label })
        }
        events.push({ type: 'timeline', id, item: { type: 'user_message', text } })
        break
      }
      case 'assistant/message': {
        const messageId = event.data.message.id
        for (const block of event.data.message.content) {
          if (block.type === 'reasoning' && block.text.length > 0) {
            events.push({ type: 'timeline', id, item: { type: 'reasoning', text: block.text } })
          } else if (block.type === 'text' && block.text.length > 0) {
            events.push({
              type: 'timeline',
              id,
              item: { type: 'assistant_message', text: block.text, messageId },
            })
          }
        }
        break
      }
      case 'tool/call':
        state.toolNames.set(event.data.callId, event.data.name)
        events.push({
          type: 'timeline',
          id,
          item: {
            type: 'tool_call',
            callId: event.data.callId,
            name: event.data.name,
            detail: { type: 'plain_text', label: event.data.name, text: event.data.arguments },
            status: 'running',
            error: null,
            metadata: {},
          },
        })
        break
      case 'tool/result': {
        const result = event.data.message.content[0]
        events.push({
          type: 'timeline',
          id,
          item: toolResultItem(
            result.toolCallId,
            state.toolNames.get(result.toolCallId) ?? 'tool',
            textOf(result.content),
            result.isError === true ? (event.data.error?.code ?? 'tool failed') : undefined,
          ),
        })
        break
      }
      default:
        break
    }
    return events
  }
}
