export const handCardStyle = ({ index, step, selected }) => ({
  left: `${index * step}px`,
  top: selected ? '-22px' : '0px',
  zIndex: index,
});
