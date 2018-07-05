#pragma once

#include <MetaNN/evaluate/cpu/trival_eval_pool.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <MetaNN/evaluate/facilities/eval_pool.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <vector>
#include <cassert>
#include <list>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <typeindex>
#include <algorithm>

namespace MetaNN
{
namespace NSEvalPlan
{
inline size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap,
                           const std::vector<const void*>& paramPtr)
{
    int res = -1;
        
    for (auto p : paramPtr)
    {
        auto it = depMap.find(p);
        if (it != depMap.end()) res = std::max(res, (int)(it->second));
    }

    return (size_t)res;
}
}

template <typename TDevice>
using EvalCluster = std::unordered_map<std::type_index, std::shared_ptr<BaseEvalGroup<TDevice>>>;

template <typename TDevice>
class EvalLayer
{
public:
    size_t Size() const
    {
        return m_evalSeq.size();
    }

    EvalCluster<TDevice>& operator[] (size_t i)
    {
        return m_evalSeq[i];
    }

    bool Empty() const
    {
        return m_evalSeq.empty();
    }

    void Clear()
    {
        m_evalSeq.clear();
        m_operands.clear();
        m_outputs.clear();
    }

    template <typename TEvalGroup, typename TEvalUnit>
    void EvalRegister(TEvalUnit&& evalReq, const void* resPtr,
                      const std::vector<const void*>& paramPtr)
    {
        if (!resPtr) return;
        if (m_outputs.find(resPtr) != m_outputs.end()) return;

        size_t depth = NSEvalPlan::OperandDepth(m_outputs, paramPtr) + 1;

        if (m_evalSeq.size() <= (size_t)depth)
        {
            m_evalSeq.resize(depth + 1);
        }
        EvalCluster<TDevice>& ec = m_evalSeq[depth];

        const auto typeIndex = std::type_index(typeid(TEvalGroup));
        auto it = ec.find(typeIndex);

        if (it == ec.end())
        {
            it = ec.insert({typeIndex, std::make_shared<TEvalGroup>()}).first;
        }
        it->second->Merge(std::forward<TEvalUnit>(evalReq));

        m_outputs.insert({resPtr, depth});
    }

private:
    std::vector<EvalCluster<TDevice>> m_evalSeq;
    std::unordered_set<const void*> m_operands;
    std::unordered_map<const void*, size_t> m_outputs;
};

template <typename TDevice>
class EvalPlan
{
private:
    static EvalPoolEnum& GlobalEvalPool()
    {
        static EvalPoolEnum inst = EvalPoolEnum::Trival;
        return inst;
    }
    
    static EvalPoolEnum& ThreadEvalPool()
    {
        static thread_local EvalPoolEnum inst = GlobalEvalPool();
        return inst;
    }
    
    static EvalPlan& ThreadInst()
    {
        static thread_local EvalPlan inst;
        return inst;
    }

public:
    static void SetEvalPool(EvalPoolEnum epType)
    {
        GlobalEvalPool() = epType;
    }

    template <typename TEvalGroup, typename TEvalUnit>
    static void Register(TEvalUnit&& evalReq, const void* outputPtr,
                         const std::vector<const void*>& paramPtr)
    {
        ThreadInst().template EvalRegister<TEvalGroup>(std::forward<TEvalUnit>(evalReq), outputPtr, paramPtr);
    }

    static void Eval()
    {
        EvalPlan& plan = ThreadInst();
        if ((ThreadEvalPool() != GlobalEvalPool()) || (!plan.m_evalPool))
        {
            switch(GlobalEvalPool())
            {
            case EvalPoolEnum::Trival:
                plan.m_evalPool = &(TrivalEvalPool<TDevice>::Instance());
                break;
            default:
                assert(false);
            }
            ThreadEvalPool() = GlobalEvalPool();
        }
        if (!plan.m_evalPool)
        {
            throw std::runtime_error("No Evaluation Pool is available.");
        }
        
        plan.DoLayerEval();
    }

private:
    EvalPlan()
        : m_evalPool(nullptr)
    {
        m_evalLayers.resize(1);
    }

    template <typename TEvalGroup, typename TEvalUnit>
    void EvalRegister(TEvalUnit&& evalReq, const void* outputPtr,
                      const std::vector<const void*>& paramPtr)
    {
        auto& curLayer = m_evalLayers.back();
        curLayer.template EvalRegister<TEvalGroup>(std::forward<TEvalUnit>(evalReq),
                                                   outputPtr, paramPtr);
    }

    void DoLayerEval()
    {
        EvalLayer<TDevice>& curLayer = m_evalLayers.back();
        if (curLayer.Empty()) return;

        m_evalLayers.push_back(EvalLayer<TDevice>{});
        size_t seqLen = curLayer.Size();
        for (size_t i = 0; i < seqLen; ++i)
        {
            EvalCluster<TDevice>& ec = curLayer[i];
            for (auto& eg : ec)
            {
                while(auto unit = eg.second->GetEvalUnit())
                {
                    m_evalPool->Process(unit);
                }
            }
            m_evalPool->Barrier();
            if (!m_evalLayers.back().Empty())
            {
                DoLayerEval();
            }
        }
        m_evalLayers.pop_back();
        curLayer.Clear();
    }

private:
    std::list<EvalLayer<TDevice>> m_evalLayers;
    BaseEvalPool<TDevice>* m_evalPool;
};

template <typename TData>
auto Evaluate(const TData& data)
{
    using DeviceType = typename TData::DeviceType;
    auto evalHandle = data.EvalRegister();
    EvalPlan<DeviceType>::Eval();
    return evalHandle.Data();
}
}