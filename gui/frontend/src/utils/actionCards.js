export const actionCards = (action) => (
  Array.isArray(action?.[2]) ? action[2] : []
);
