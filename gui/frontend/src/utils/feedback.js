const MAX_MESSAGE_LENGTH = 1200;
const SAFE_CONTEXT_KEYS = [
  'url', 'path', 'userAgent', 'viewport', 'roomId', 'playerId', 'message', 'stack',
];

export const trimFeedbackText = (value) =>
  String(value || '').trim().slice(0, MAX_MESSAGE_LENGTH);

const safeContext = (context = {}) =>
  Object.fromEntries(
    SAFE_CONTEXT_KEYS
      .filter((key) => Object.prototype.hasOwnProperty.call(context, key))
      .map((key) => [key, context[key]]),
  );

export const buildFeedbackPayload = ({
  kind = 'suggestion',
  message = '',
  page = 'unknown',
  roomId = '',
  playerId = null,
  participantId = '',
  context = {},
} = {}) => {
  const payload = {
    kind,
    message: trimFeedbackText(message),
    page,
    roomId,
    playerId,
    context: safeContext(context),
  };
  if (participantId) {
    payload.participantId = participantId;
  }
  return payload;
};
