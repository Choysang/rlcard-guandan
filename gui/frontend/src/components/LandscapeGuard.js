import './LandscapeGuard.css';

const LandscapeGuard = ({ children }) => (
  <>
    <div className="landscape-guard">
      <div className="rotate-device-icon">↻</div>
      <div className="rotate-title">请横屏游戏</div>
      <div className="rotate-copy">横屏可以看清手牌和每家出牌。</div>
    </div>
    <div className="landscape-content">{children}</div>
  </>
);

export default LandscapeGuard;
