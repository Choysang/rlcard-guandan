const HTTP_PROTOCOL_RE = /^https?:\/\//i;

export const normalizeServerUrl = (rawUrl, pageOrigin) => {
  const value = rawUrl.trim();
  if (!value) return pageOrigin;

  const hasExplicitProtocol = HTTP_PROTOCOL_RE.test(value);
  const url = new URL(hasExplicitProtocol ? value : `http://${value}`);
  const pageUrl = new URL(pageOrigin);

  if (
    url.hostname === pageUrl.hostname
    && url.port === '5000'
    && pageUrl.protocol === 'https:'
    && !pageUrl.port
  ) {
    return pageUrl.origin;
  }

  if (url.hostname === pageUrl.hostname && !url.port) {
    return pageUrl.origin;
  }

  if (!hasExplicitProtocol && !url.port) {
    url.port = '5000';
  }

  return url.origin;
};
