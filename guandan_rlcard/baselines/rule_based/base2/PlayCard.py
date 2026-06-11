def _debug_print(*args, **kwargs):
    # Debug output from the original research code, silenced for
    # the open-source release.
    pass


from .CreateActionList import CreateActionList
from .CountValue import CountValue
from .config import CompareRank
from . import config
import json
import time


class PlayCard():

    def actBack(self, handCards, curRank):
        bestPlay = []
        maxValue = -100
        for rank in config.cardRanks:
            if (rank!=curRank and rank<='9' and rank>='2'):
                for card in handCards:
                    if (card[1]==rank):
                        action = [card]
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        if (restValue>maxValue):
                            maxValue = restValue
                            bestPlay = {"action": action, "type": "back", "rank": rank}
                        #_debug_print(card, restValue)
                        break
        return bestPlay

    def GetAdditionalActionList(self, typeList, curRank, fullActionList):
        additionalActionList=[]
        dict = {}
        for action in fullActionList:
            if (action[0] in typeList and ((action[0], action[1]) not in dict.keys())):
                for card in action[2]:
                    if card == 'H'+curRank:
                        additionalActionList.append(action)
                        dict[(action[0], action[1])] = 1
                        break
        return additionalActionList

    def FreePlay(self, strategy, handCards, curRank, fullActionList = None):
        # _debug_print("Free play handCards:", handCards, "   restHandsCount:", strategy.restHandsCount)
        handValue, handActions = CountValue().HandCardsValue(handCards, 0, curRank)
        #_debug_print(handActions)
        #strategy.SetBeginning(0)
        strategy.SetRole(handValue, handActions, curRank)
        strategy.makeReviseValues()
        #_debug_print(strategy.recordPlayerActions)
        additionalActionList = self.GetAdditionalActionList(["ThreePair", "Straight"], curRank, fullActionList)
        #_debug_print("additionalActionList", additionalActionList)
        #beginning
        bestPlay = {}
        if (len(handCards)>=15 or strategy.roundStage != 'ending'):
            minValue = 100
            for action in handActions:
                actionValue = CountValue().ActionValue(action, action['type'], action['rank'], curRank) - strategy.freeActionRV[action['type']] \
                        - strategy.freeActionRV[(action['type'],action['rank'])]
                #_debug_print(action, actionValue)
                if actionValue < minValue:
                    minValue = actionValue
                    bestPlay = action
        #_debug_print(strategy.freeActionRV[('Pair','Q')])
        else:
            maxValue = -100
            actionList = CreateActionList().CreateList(handCards)
            for i in range(0, len(config.cardTypes)):
                type = config.cardTypes[i]
                #if (type == 'StraightFlush'): continue
                for rank1 in actionList[type]:
                    for card in actionList[type][rank1]:
                        color = None
                        rank = rank1  # to distinguish StraightFlush from others
                        if (type == 'StraightFlush'):
                            rank = rank1[1]
                            color = rank1[0]
                        #_debug_print("Free play trying type, rank, card:", type, rank, card)
                        action = CreateActionList().GetAction(type, rank, card, handCards, color)
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        thisHandValue = CountValue().ActionValue(action, type, rank, curRank)
                        thisHandValue += strategy.freeActionRV[type]
                        if (type, rank) in strategy.freeActionRV.keys():
                            thisHandValue += strategy.freeActionRV[(type, rank)]
                        # _debug_print(strategy.actionValueRevise)
                        # _debug_print(rank, card, thisHandValue, restValue)
                        if (thisHandValue < 0): thisHandValue = 0
                        if (thisHandValue + restValue > maxValue or (thisHandValue + restValue == maxValue and \
                            (bestPlay == [] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                            maxValue = thisHandValue + restValue
                            bestPlay = {"action": action, "type": type, "rank": rank}
                            # _debug_print(bestPlay, maxValue)

            #try additional list
            for action in additionalActionList:
                type = action[0]
                rank = action[1]
                card = rank
                if type == 'Bomb':
                    card = len(action[2])

                restCards = CreateActionList().GetRestCards(action[2], handCards)
                restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                restValue += strategy.handRV[type]
                thisHandValue = CountValue().ActionValue(action[2], type, rank, curRank)
                thisHandValue += strategy.freeActionRV[type]
                if (type, rank) in strategy.freeActionRV.keys():
                    thisHandValue += strategy.freeActionRV[(type, rank)]
                # _debug_print(strategy.actionValueRevise)
                # _debug_print(rank, card, thisHandValue, restValue)
                if (thisHandValue < 0): thisHandValue = 0
                if (thisHandValue + restValue > maxValue or (thisHandValue + restValue == maxValue and
                                (bestPlay == [] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                    maxValue = thisHandValue + restValue
                    bestPlay = {"action": action[2], "type": type, "rank": rank}
                    # _debug_print('Using additional action list')
                    # _debug_print(bestPlay, maxValue)

        #_debug_print("bestplay:",bestPlay, "handValue", handValue)
        return bestPlay

    def RestrictedPlay(self, strategy, handCards, formerAction, curRank, fullActionList = None):
        # _debug_print("Restricted Play handCards:", handCards,"   restHandsCount:", strategy.restHandsCount)
        actionList = CreateActionList().CreateList(handCards)

        additionalActionList = self.GetAdditionalActionList(["Bomb", "StraightFlush", "ThreePair", "Straight"], curRank,
                                                            fullActionList)
        #_debug_print("additionalActionList:", additionalActionList)

        bestPlay = []
        maxValue, restActions = CountValue().HandCardsValue(handCards, 0, curRank)
        strategy.SetRole(maxValue, restActions, curRank)
        strategy.makeReviseValues()
        maxValue += strategy.restrictedActionRV["PASS"]

        #_debug_print(maxValue)
        toc = time.time()
        #_debug_print(toc - tic)

        for i in range(0, len(config.cardTypes)):
            type = config.cardTypes[i]
            #_debug_print(type, formerAction["type"])
            #if (type == 'StraightFlush'): continue
            if (type != 'Bomb' and type != 'StraightFlush' and type != formerAction["type"]): continue
            for rank1 in actionList[type]:
                for card in actionList[type][rank1]:
                    color = None
                    rank = rank1  # to distinguish StraightFlush from others
                    if (type == 'StraightFlush'):
                        rank = rank1[1]
                        color = rank1[0]
                    #_debug_print("Restricted play trying rank, card:", type, rank, card)
                    if (CompareRank().Larger(type, rank, card, formerAction, curRank)):
                        action = CreateActionList().GetAction(type, rank, card, handCards, color)
                        restCards = CreateActionList().GetRestCards(action, handCards)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        #restValue += strategy.handRV[type]
                        thisHandValue = CountValue().ActionValue(action, type, rank, curRank)
                        thisHandValue += strategy.restrictedActionRV[type]
                        if (type, rank) in strategy.restrictedActionRV.keys():
                            thisHandValue += strategy.restrictedActionRV[(type, rank)]
                        #_debug_print(strategy.actionValueRevise)
                        #_debug_print(rank, card, thisHandValue, restValue)
                        if (thisHandValue < 0): thisHandValue = 0
                        if (thisHandValue + restValue > maxValue or (thisHandValue + restValue == maxValue and \
                        (bestPlay==[] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                            maxValue = thisHandValue + restValue
                            bestPlay = {"action": action, "type": type, "rank": rank}
                            # _debug_print(maxValue, bestPlay)

        #try additional list
        for action in additionalActionList:
            type = action[0]
            rank = action[1]
            card = rank
            if type == 'Bomb':
                card = len(action[2])
            if (CompareRank().Larger(type, rank, card, formerAction, curRank)):
                restCards = CreateActionList().GetRestCards(action[2], handCards)
                restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                #restValue += strategy.handRV[type]
                thisHandValue = CountValue().ActionValue(action[2], type, rank, curRank)
                thisHandValue += strategy.restrictedActionRV[type]
                if (type, rank) in strategy.restrictedActionRV.keys():
                    thisHandValue += strategy.restrictedActionRV[(type, rank)]
                # _debug_print(strategy.actionValueRevise)
                # _debug_print(rank, card, thisHandValue, restValue)
                if (thisHandValue < 0): thisHandValue = 0
                if (thisHandValue + restValue > maxValue or (thisHandValue + restValue == maxValue and
                                                (bestPlay == [] or CompareRank().Smaller(type, rank, card, bestPlay, curRank)))):
                    maxValue = thisHandValue + restValue
                    bestPlay = {"action": action[2], "type": type, "rank": rank}
                    # _debug_print('Using additional action list')
                    # _debug_print(bestPlay, maxValue)

        if (bestPlay==[]):
            bestPlay = {'action': 'PASS', 'type': 'PASS', 'rank': 'PASS'}
        #_debug_print("bestplay:", bestPlay, "maxvalue", maxValue)
        return bestPlay

    def Play(self, handCards, curRank):
        self.FreePlay(handCards, curRank)
