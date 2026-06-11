"""Prompt templates for the LLM Guandan agent.

``COMPACT_PROMPT`` is the minimal state dump used in the original
experiments. ``GUIDED_PROMPT`` additionally restates the core rules and
gives playing advice. Both ask the model to answer with a single legal
action index.
"""

from guandan_rlcard.constants import CARD_RANK

COMPACT_PROMPT = '''\
你是一个掼蛋玩家，现在正在打掼蛋比赛。所有现有情况表述如下，请从合法动作中选取一个最优动作作为当前动作打出，注意你只需要回答该动作的序号（从0开始）。

```现有状态信息：
所有玩家id为 [0, 1, 2, 3]，你的id是{id}，你的队友的id为 {teammate_id}，你的队伍是 {teamid}。
当前队伍0的等级为 {team0_rank} ，队伍1的等级为 {team1_rank}。
现在打的是队伍 {play_team} 的等级，级牌为 '{cur_rank}'
```

```你的当前手牌：
{current_hand}
```

```每个玩家剩余手牌数（按玩家id顺序）：
{num_cards_left}
```

```最近的动作轨迹（按时间先后顺序从前到后排列，[玩家id，动作]）：
{trace}
```

```当前你的合法动作：
{legal_actions}
```

```合法动作序号上限：
{num_legal_actions}
```

在游戏过程中，你需要遵循以下不可违背的原则：
- 最终回答只是一个表示动作序号的数字，不要回答任何其他信息。
- 你回答的数字是合法动作列表中的索引，请不要超过合法动作序号上限。
'''

GUIDED_PROMPT = '''\
**任务**：你正在玩掼蛋。掼蛋是四人成对竞技的纸牌游戏，使用两副扑克（108张），你的目标是与队友（id相差2）合作率先出完手牌。

#### 核心规则提示：
1. 牌型：单张、对子、三张、三带二、三连对（如334455）、钢板（两个连续三张，如333444）、顺子（恰好5张，如34567）、炸弹（4-10张同点数）、同花顺（5张同花连续）、天王炸（四张王，最大）。
2. 牌点大小：大王 > 小王 > 级牌 > A > K > ... > 3 > 2；级牌是当前打的等级对应的牌。红桃级牌是"逢人配"，可以当作除王以外的任意牌使用。
3. 顺子/三连对/钢板内部按自然点数比较（A可作1），级牌在其中只按自然点数。
4. 炸弹可压任意普通牌型；张数多的炸弹更大；同花顺大于5张炸弹、小于6张炸弹；天王炸最大。
5. 跟牌须出同类型更大的牌或用炸弹压制，无法压制或不想出时选择PASS。

**游戏状态**
- 你的id：{id}，队友id：{teammate_id}，你的队伍：{teamid}
- 队伍0等级：{team0_rank}，队伍1等级：{team1_rank}，当前打队伍 {play_team} 的等级，级牌为 '{cur_rank}'
- 你的手牌：{current_hand}
- 各玩家剩余手牌数（按id顺序）：{num_cards_left}
- 最近动作轨迹（[玩家id, 动作]）：{trace}

**当前轮到你出牌**
- 合法动作列表：{legal_actions}
- 合法动作序号上限：{num_legal_actions}

**建议**
- 与队友配合：队友占优时避免压队友。
- 对手只剩少量牌时优先压制。
- 保留炸弹与逢人配到关键时刻。

**回答要求**
只回答所选动作的序号（从0开始的数字），不要输出其他内容，序号不得超过上限。
'''


def render_prompt(state, template=COMPACT_PROMPT, trace_window=20):
    """Fill a prompt template from an environment state dict."""
    player_id = state['self']
    rank_list = state['rank_list']
    return template.format(
        id=player_id,
        teammate_id=(player_id + 2) % 4,
        teamid=state['teamid'],
        team0_rank=CARD_RANK[rank_list[0]],
        team1_rank=CARD_RANK[rank_list[1]],
        play_team=state['play_team'],
        cur_rank=CARD_RANK[rank_list[state['play_team']]],
        current_hand=state['current_hand'],
        num_cards_left=state['num_cards_left'],
        trace=state['trace'][-trace_window:],
        legal_actions=state['actions'],
        num_legal_actions=len(state['actions']) - 1,
    )
