#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/evaluate/facilities/eval_pool.h>
namespace MetaNN
{
template <>
class TrivalEvalPool<DeviceTags::CPU> : public BaseEvalPool<DeviceTags::CPU>
{
public:
    static TrivalEvalPool& Instance()
    {
        static TrivalEvalPool inst;
        return inst;
    }
private:
    TrivalEvalPool() = default;
public:
    void Process(std::shared_ptr<BaseEvalUnit<DeviceTags::CPU>>& eu) override
    {
        eu->Eval();
    }

    void Barrier() override {}
};
}