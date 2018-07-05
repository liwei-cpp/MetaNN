#pragma once

#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <memory>

namespace MetaNN
{
enum class EvalPoolEnum
{
    Trival
};

template <typename TDevice>
class BaseEvalPool
{
public:
    virtual ~BaseEvalPool() = default;
    virtual void Process(std::shared_ptr<BaseEvalUnit<TDevice>>&) = 0;
    virtual void Barrier() = 0;
};

template <typename TDevice>
class TrivalEvalPool;
}