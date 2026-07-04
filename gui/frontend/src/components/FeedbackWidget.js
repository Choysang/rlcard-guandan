import { useState } from 'react';
import guandanService from '../services/GuandanService';
import { buildFeedbackPayload, trimFeedbackText } from '../utils/feedback';
import './FeedbackWidget.css';

const KIND_OPTIONS = [
  { value: 'suggestion', label: '建议' },
  { value: 'bug', label: '问题' },
  { value: 'client_error', label: '错误' },
];

const FeedbackWidget = ({
  page,
  roomId = '',
  playerId = null,
  participantId = '',
  className = '',
}) => {
  const [open, setOpen] = useState(false);
  const [kind, setKind] = useState('suggestion');
  const [message, setMessage] = useState('');
  const [status, setStatus] = useState('idle');

  const canSubmit = trimFeedbackText(message).length > 0 && status !== 'sending';

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setStatus('sending');
    const payload = buildFeedbackPayload({
      kind,
      message,
      page,
      roomId,
      playerId,
      participantId,
      context: {
        url: window.location.href,
        path: window.location.pathname,
        userAgent: navigator.userAgent,
        viewport: `${window.innerWidth}x${window.innerHeight}`,
        roomId,
        playerId,
      },
    });

    try {
      const response = await fetch(`${guandanService.getServerUrl()}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error('feedback failed');
      setMessage('');
      setStatus('sent');
    } catch {
      setStatus('failed');
    }
  };

  return (
    <div className={`feedback-widget ${className}`}>
      <button
        type="button"
        className="feedback-trigger"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        反馈
      </button>

      {open && (
        <div className="feedback-panel">
          <div className="feedback-kind-row" aria-label="反馈类型">
            {KIND_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={kind === option.value ? 'active' : ''}
                onClick={() => setKind(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <textarea
            value={message}
            maxLength="1200"
            onChange={(event) => {
              setMessage(event.target.value);
              setStatus('idle');
            }}
            placeholder="描述你遇到的问题或建议"
          />
          <div className="feedback-actions">
            <span className={`feedback-state ${status}`}>
              {status === 'sent' ? '已提交' : status === 'failed' ? '提交失败' : ''}
            </span>
            <button type="button" onClick={handleSubmit} disabled={!canSubmit}>
              {status === 'sending' ? '提交中' : '提交反馈'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeedbackWidget;
