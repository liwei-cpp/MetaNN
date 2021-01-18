#pragma once

#include <unordered_map>
#include <set>
#include <memory>
#include <MetaNN/evaluate/eval_dispatcher.h>

namespace MetaNN
{
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
        void Register(std::unique_ptr<BaseEvalItem> item)
        {
            assert(item);
            DataPtr outPtr = item->OutputPtr();
            if (IsAlreadyRegisted(outPtr)) return;
            
            const auto itemID = item->ID();
            auto dispIt = m_itemDispatcher.find(itemID);
            if (dispIt == m_itemDispatcher.end())
            {
                m_itemDispatcher.insert({itemID, std::make_unique<TDispatcher>(itemID)});
            }

            const auto& inPtrs = item->InputPtrs();
            
            size_t inArc = 0;
            for (auto* const in : inPtrs)
            {
                if (m_nodes.find(in) != m_nodes.end())
                {
                    m_nodeAimPos[in].insert(outPtr);
                    ++inArc;
                }
            }
            m_nodeInArcNum[outPtr] = inArc;
            m_nodes.emplace(outPtr, std::move(item));
            if (inArc == 0)
            {
                assert(m_procNodes.find(outPtr) == m_procNodes.end());
                m_procNodes.insert(outPtr);
            }
        }
        
        bool IsAlreadyRegisted(DataPtr ptr) const
        {
            return m_nodes.find(ptr) != m_nodes.end();
        }
        
        void Eval()
        {
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
                    assert(m_nodeInArcNum.find(p) != m_nodeInArcNum.end());
                    assert(m_nodeInArcNum[p] == 0);
                    
                    m_procNodes.erase(p);
                    m_nodes.erase(p);
                    m_nodeInArcNum.erase(p);
                    
                    auto aimNodeIt = m_nodeAimPos.find(p);
                    if (aimNodeIt == m_nodeAimPos.end()) continue;
                    for (DataPtr aimNode : aimNodeIt->second)
                    {
                        auto arcNumIt = m_nodeInArcNum.find(aimNode);
                        assert(arcNumIt != m_nodeInArcNum.end());
                        assert(arcNumIt->second > 0);
                        --arcNumIt->second;
                        if (arcNumIt->second == 0)
                        {
                            newProcNodes.insert(arcNumIt->first);
                        }
                    }
                    m_nodeAimPos.erase(p);
                }
                AddToDispatcher(newProcNodes);
                m_procNodes.insert(newProcNodes.begin(), newProcNodes.end());
            }

            assert(m_nodeInArcNum.empty());
            assert(m_nodeAimPos.empty());
            assert(m_nodes.empty());
        }

    private:
        EvalPlan() = default;
        EvalPlan(const EvalPlan&) = delete;
        EvalPlan& operator= (const EvalPlan&) = delete;

        void AddToDispatcher(const std::set<DataPtr>& procNodes)
        {
            for (DataPtr curNodePtr : procNodes)
            {
                auto it = m_nodes.find(curNodePtr);
                assert(it != m_nodes.end());
                
                std::unique_ptr<BaseEvalItem> curNode = std::move(it->second);
                
                const auto itemID = curNode->ID();
                auto dispIt = m_itemDispatcher.find(itemID);
                assert(dispIt != m_itemDispatcher.end());
                dispIt->second->Add(std::move(curNode));
            }
        }
                             
    private:
        std::unordered_map<DataPtr, size_t> m_nodeInArcNum;
        std::unordered_map<DataPtr, std::set<DataPtr>> m_nodeAimPos;
        std::unordered_map<DataPtr, std::unique_ptr<BaseEvalItem>> m_nodes;
        std::unordered_map<size_t, std::unique_ptr<BaseEvalItemDispatcher>> m_itemDispatcher;
        std::set<DataPtr> m_procNodes;
    };
    
    template <typename TData>
    auto Evaluate(const TData& data)
    {
        auto evalHandle = data.EvalRegister();
        EvalPlan::Inst().Eval();
        return evalHandle.Data();
    }
}