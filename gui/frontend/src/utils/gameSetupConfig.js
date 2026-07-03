export const EMPTY_LLM_CONFIG = {
  apiKey: '',
  baseUrl: '',
  model: '',
};

const trim = (value) => (typeof value === 'string' ? value.trim() : '');

export const needsLlmConfig = (playerConfigs) =>
  playerConfigs.some((player) => player.type === 'ai' && player.agent === 'llm');

export const hasCompleteLlmConfig = (llmConfig) =>
  Boolean(trim(llmConfig?.apiKey) && trim(llmConfig?.baseUrl) && trim(llmConfig?.model));

export const isUnsafeLlmOrigin = (locationLike) => {
  const location = locationLike || globalThis.location;
  if (!location) {
    return false;
  }
  const hostname = location.hostname || '';
  const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1';
  return location.protocol !== 'https:' && !isLocalhost;
};

export const buildRoomConfig = (playerConfigs, debugEnabled, llmConfig = {}) => {
  const agentTypes = {};
  const humanPlayerIds = [];

  playerConfigs.forEach((player, index) => {
    if (player.type === 'ai') {
      agentTypes[index.toString()] = player.agent;
    } else {
      humanPlayerIds.push(index);
    }
  });

  const roomConfig = {
    agentTypes,
    human_player_ids: humanPlayerIds,
    debug_enabled: Boolean(debugEnabled),
  };

  if (needsLlmConfig(playerConfigs)) {
    roomConfig.llmConfig = {
      apiKey: trim(llmConfig.apiKey),
      baseUrl: trim(llmConfig.baseUrl),
      model: trim(llmConfig.model),
    };
  }

  return roomConfig;
};
