#pragma once

#include <unordered_map>
#include <set>
#include <memory>
#include <MetaNN/evaluate/eval_dispatcher.h>

namespace MetaNN
{
    template <typename TDevice>
    class EvalPlan
    {
        using DataPtr = const void*;
    public:
        static EvalPlan& Inst()
        {
            static EvalPlan inst;
            return inst;
        }

        template <typename TDispatcher>
        void Register(std::unique_ptr<BaseEvalItem<TDevice>> item)
        {
            assert(item);
            DataPtr outPtr = item->OutputPtr();
            if (IsAlreayRegisted(outPtr)) return;
            
            const auto itemID = item->ID();
            auto dispIt = m_itemDispatcher.find(itemID);
            if (dispIt == m_itemDispatcher.end())
            {
                m_itemDispatcher.insert({itemID, std::make_unique<TDispatcher>(itemID)});
            }

            const auto& inPtrs = item->InputPtrs();
            
            size_t inAct = 0;
            for (auto* const in : inPtrs)
            {
                if (m_nodes.find(in) != m_nodes.end())
                {
                    m_nodeAimPos[in].insert(outPtr);
                    ++inAct;
                }
            }
            m_nodeInActNum[outPtr] = inAct;
            m_nodes.emplace(outPtr, std::move(item));
            if (inAct == 0)
            {
                assert(m_procNodes.find(outPtr) == m_procNodes.end());
                m_procNodes.insert(outPtr);
            }
        }
        
        bool IsAlreayRegisted(DataPtr ptr) const
        {
            return m_nodes.find(ptr) != m_nodes.end();
        }
        
        void Eval()
        {
            if (m_procNodes.empty())
            {
                assert(m_nodes.empty());
                assert(m_nodeInActNum.empty());
                assert(m_nodeAimPos.empty());
                return;
            }

            AddToDispatcher(m_procNodes);
            
            while (!m_procNodes.empty())
            {
                size_t maxGroupSize = 0;
                auto itemDispIt = m_itemDispatcher.end();
                
                for (auto curDispIt = m_itemDispatcher.begin();
                     curDispIt != m_itemDispatcher.end(); ++curDispIt)
                {
                    size_t maxGroupOfThisDisp = curDispIt->second->MaxEvalGroupSize();
                    if (maxGroupOfThisDisp > maxGroupSize)
                    {
                        maxGroupSize = maxGroupOfThisDisp;
                        itemDispIt = curDispIt;
                    }
                }
                
                assert(maxGroupSize > 0);
                auto nextGroup = itemDispIt->second->PickNextGroup();
                nextGroup->Eval();
                auto resSet = nextGroup->ResultPointers();
                
                std::set<DataPtr> newProcNodes;
                for (DataPtr p : resSet)
                {
                    assert(m_procNodes.find(p) != m_procNodes.end());
                    assert(m_nodes.find(p) != m_nodes.end());
                    assert(m_nodeInActNum.find(p) != m_nodeInActNum.end());
                    assert(m_nodeInActNum[p] == 0);
                    
                    m_procNodes.erase(p);
                    m_nodes.erase(p);
                    m_nodeInActNum.erase(p);
                    
                    auto aimNodeIt = m_nodeAimPos.find(p);
                    if (aimNodeIt == m_nodeAimPos.end()) continue;
                    for (DataPtr aimNode : aimNodeIt->second)
                    {
                        auto actNumIt = m_nodeInActNum.find(aimNode);
                        assert(actNumIt != m_nodeInActNum.end());
                        assert(actNumIt->second > 0);
                        --actNumIt->second;
                        if (actNumIt->second == 0)
                        {
                            newProcNodes.insert(actNumIt->first);
                        }
                    }
                    m_nodeAimPos.erase(p);
                }
                AddToDispatcher(newProcNodes);
                m_procNodes.insert(newProcNodes.begin(), newProcNodes.end());
            }

            assert(m_nodeInActNum.empty());
            assert(m_nodeAimPos.empty());
            assert(m_nodes.empty());
        }

    private:
        EvalPlan() = default;

        void AddToDispatcher(const std::set<DataPtr>& procNodes)
        {
            for (DataPtr curNodePtr : procNodes)
            {
                auto it = m_nodes.find(curNodePtr);
                assert(it != m_nodes.end());
                
                std::unique_ptr<BaseEvalItem<TDevice>> curNode = std::move(it->second);
                
                const auto itemID = curNode->ID();
                auto dispIt = m_itemDispatcher.find(itemID);
                assert(dispIt != m_itemDispatcher.end());
                dispIt->second->Add(std::move(curNode));
            }
        }
                             
    private:
        std::unordered_map<DataPtr, size_t> m_nodeInActNum;
        std::unordered_map<DataPtr, std::set<DataPtr>> m_nodeAimPos;
        std::unordered_map<DataPtr, std::unique_ptr<BaseEvalItem<TDevice>>> m_nodes;
        std::unordered_map<size_t, std::unique_ptr<BaseEvalItemDispatcher<TDevice>>> m_itemDispatcher;
        std::set<DataPtr> m_procNodes;
    };
    
    template <typename TData>
    auto Evaluate(const TData& data)
    {
        using DeviceType = typename TData::DeviceType;
        auto evalHandle = data.EvalRegister();
        EvalPlan<DeviceType>::Inst().Eval();
        return evalHandle.Data();
    }
}