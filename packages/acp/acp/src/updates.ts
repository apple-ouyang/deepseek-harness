/** Standard ACP updates derived from committed DSH session events. */

import type { Context } from '@deepseek-ai/cordis'
import type { SessionUpdate, ToolCallContent, ToolCallLocation, ToolKind } from '@agentclientprotocol/sdk'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-token-meter'
import { assistantBlockToAcp } from './content.ts'

/**
 * Convert one committed assistant message and its context usage in block order.
 * @param ctx - bridge context carrying attachment and token-meter services.
 * @param session - durable session used for context pressure.
 * @param event - committed assistant message event.
 * @returns ordered standard thought, message, and optional usage updates.
 */
export async function assistantUpdates(
  ctx: Context,
  session: Session,
  event: SessionEvent<'assistant/message'>,
): Promise<SessionUpdate[]> {
  const updates: SessionUpdate[] = []
  for (const block of event.data.message.content) {
    if (block.type === 'reasoning') {
      if (block.text.length > 0) {
        updates.push({
          sessionUpdate: 'agent_thought_chunk',
          messageId: event.data.message.id,
          content: { type: 'text', text: block.text },
        })
      }
      continue
    }
    const content = await assistantBlockToAcp(ctx, block)
    if (content !== undefined) {
      updates.push({
        sessionUpdate: 'agent_message_chunk',
        messageId: event.data.message.id,
        content,
      })
    }
  }
  const usage = usageUpdate(ctx, session, event)
  if (usage !== undefined) updates.push(usage)
  return updates
}

/**
 * ACP kind hints for the built-in tool vocabulary. Plugin and unknown tools
 * stay `other`; the durable log only carries the tool name, so the adapter
 * cannot recover a richer semantic for names it does not know.
 */
const acpToolKind: Readonly<Record<string, ToolKind>> = {
  bash: 'execute',
  pwsh: 'execute',
  read: 'read',
  read_image: 'read',
  edit: 'edit',
  write: 'edit',
  grep: 'search',
  glob: 'search',
  web_search: 'search',
  web_fetch: 'fetch',
}

/**
 * Surface the single file a tool call touches so clients can track it.
 * @param args - parsed tool arguments.
 * @returns one location for `file_path`/`path` arguments, or none.
 */
function acpToolLocations(args: unknown): ToolCallLocation[] | undefined {
  if (typeof args !== 'object' || args === null) return undefined
  const record = args as Record<string, unknown>
  const path = record.file_path ?? record.path
  return typeof path === 'string' && path.length > 0 ? [{ path }] : undefined
}

/**
 * Preview a file rewrite as an ACP diff before the tool result commits.
 * `edit` knows both literals; `write` is a full replacement or creation.
 * @param name - durable tool name.
 * @param args - parsed tool arguments.
 * @returns one diff block, or none for non-file tools.
 */
function acpToolDiffs(name: string, args: unknown): ToolCallContent[] | undefined {
  if (typeof args !== 'object' || args === null) return undefined
  const record = args as Record<string, unknown>
  const path = record.file_path
  if (typeof path !== 'string' || path.length === 0) return undefined
  if (name === 'edit' && typeof record.new_string === 'string') {
    return [{
      type: 'diff',
      path,
      oldText: typeof record.old_string === 'string' ? record.old_string : null,
      newText: record.new_string,
    }]
  }
  if (name === 'write' && typeof record.content === 'string') {
    return [{ type: 'diff', path, oldText: null, newText: record.content }]
  }
  return undefined
}

/**
 * Read the tool-owned presentation diffs a `tool/result` persists in `meta`.
 * Tools attach `meta.diffs` (e.g. `dsh-tool-fs` hunks) for exactly this
 * purpose; absent or malformed entries fall back to the plain text result.
 * @param meta - opaque tool-private payload from the durable result event.
 * @returns ACP diff content blocks, or undefined when the result has none.
 */
function acpResultDiffs(meta: unknown): ToolCallContent[] | undefined {
  if (typeof meta !== 'object' || meta === null) return undefined
  const diffs = (meta as { diffs?: unknown }).diffs
  if (!Array.isArray(diffs) || diffs.length === 0) return undefined
  const content: ToolCallContent[] = []
  for (const diff of diffs) {
    if (typeof diff !== 'object' || diff === null) continue
    const { path, oldText, newText } = diff as { path?: unknown; oldText?: unknown; newText?: unknown }
    if (typeof path !== 'string' || typeof newText !== 'string') continue
    content.push({
      type: 'diff',
      path,
      oldText: typeof oldText === 'string' ? oldText : null,
      newText,
    })
  }
  return content.length > 0 ? content : undefined
}

/**
 * Collect the model-facing text of a failed result for `rawOutput.message`,
 * the field ACP clients surface as the tool error.
 * @param message - committed tool-result message.
 * @returns joined text blocks, or undefined when none exist.
 */
function acpErrorMessage(message: SessionEvent<'tool/result'>['data']['message']): string | undefined {
  const parts: string[] = []
  for (const block of message.content) {
    if (block.type === 'text' && block.text.length > 0) parts.push(block.text)
  }
  return parts.length > 0 ? parts.join('\n') : undefined
}

/**
 * Start one generic ACP tool lifecycle from the durable call fact.
 * @param event - committed DSH tool-call event.
 * @returns the standard tool-call update with kind, location, and diff hints.
 */
export function toolCallUpdate(event: SessionEvent<'tool/call'>): SessionUpdate {
  const rawInput = parseToolArguments(event.data.arguments)
  const locations = acpToolLocations(rawInput)
  const content = acpToolDiffs(event.data.name, rawInput)
  return {
    sessionUpdate: 'tool_call',
    toolCallId: event.data.callId,
    title: event.data.name,
    kind: acpToolKind[event.data.name] ?? 'other',
    status: 'in_progress',
    rawInput,
    ...(locations === undefined ? {} : { locations }),
    ...(content === undefined ? {} : { content }),
  }
}

/**
 * Finish one generic ACP tool lifecycle from its committed model-facing result.
 * @param ctx - bridge context carrying the attachment store.
 * @param event - committed DSH tool-result event.
 * @param fallbackContent - diff blocks projected at call time, reused when the
 *   tool's result `meta` carries none (e.g. `write` on a new file). Mirrors the
 *   tool's own `presentResult` fallback onto the call arguments.
 * @returns the standard completed or failed tool-call update.
 */
export async function toolResultUpdate(
  ctx: Context,
  event: SessionEvent<'tool/result'>,
  fallbackContent?: ToolCallContent[],
): Promise<SessionUpdate> {
  const message = event.data.message
  // Presentation diffs replace the whole result body: the accompanying text is
  // only a confirmation sentence that would misrender as a unified diff.
  const diffs = acpResultDiffs(event.data.meta) ?? fallbackContent
  const content: ToolCallContent[] = diffs ?? []
  if (diffs === undefined) {
    for (const block of message.content) {
      const converted = await assistantBlockToAcp(ctx, block)
      if (converted !== undefined) content.push({ type: 'content' as const, content: converted })
    }
  }
  const errorMessage = message.isError === true ? acpErrorMessage(message) : undefined
  return {
    sessionUpdate: 'tool_call_update',
    toolCallId: message.toolCallId,
    status: message.isError === true ? 'failed' : 'completed',
    ...(content.length > 0 ? { content } : {}),
    ...(errorMessage === undefined ? {} : { rawOutput: { message: errorMessage } }),
  }
}

/** Report current context occupancy only when DSH has both usage and capacity facts. */
function usageUpdate(
  ctx: Context,
  session: Session,
  event: SessionEvent<'assistant/message'>,
): SessionUpdate | undefined {
  if (event.data.usage === undefined) return undefined
  const size = session.requestContext()?.contextWindow
  const meter = ctx.get('tokenMeter')
  if (size === undefined || meter === undefined) return undefined
  return {
    sessionUpdate: 'usage_update',
    used: meter.measure(session).totalTokens,
    size,
  }
}

/** Preserve malformed model output as opaque input instead of dropping the call update. */
function parseToolArguments(value: string): unknown {
  try {
    return JSON.parse(value) as unknown
  } catch (_invalidModelJson) {
    return value
  }
}

/**
 * Replay a committed user prompt as ACP `user_message_chunk` updates.
 * Plugin-injected context stays off the client transcript.
 */
export async function userMessageUpdates(
  ctx: Context,
  event: SessionEvent<'user/message'>,
): Promise<SessionUpdate[]> {
  const message = event.data
  if (message.source.kind !== 'user') return []
  const updates: SessionUpdate[] = []
  for (const block of message.content) {
    if (block.type === 'text' && block.text.length > 0) {
      updates.push({
        sessionUpdate: 'user_message_chunk',
        messageId: message.id,
        content: { type: 'text', text: block.text },
      })
      continue
    }
    if (block.type === 'image') {
      try {
        const content = await assistantBlockToAcp(ctx, block)
        if (content !== undefined) {
          updates.push({
            sessionUpdate: 'user_message_chunk',
            messageId: message.id,
            content,
          })
        }
      } catch (_image) {
        /* skip unrestorable attachments rather than failing the whole replay */
      }
    }
  }
  return updates
}
