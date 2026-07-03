export const sortedUnique = (indexes) => (
  [...new Set(indexes)].sort((a, b) => a - b)
);

export const dragModeForIndex = (selectedIndexes, index) => (
  selectedIndexes.includes(index) ? 'deselect' : 'select'
);

export const applyCardSelection = (selectedIndexes, index, mode) => {
  if (mode === 'select') {
    return sortedUnique([...selectedIndexes, index]);
  }
  if (mode === 'deselect') {
    return selectedIndexes.filter((i) => i !== index);
  }
  return sortedUnique(selectedIndexes);
};
