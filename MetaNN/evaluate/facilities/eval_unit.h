#pragma once

#include <unordered_map>
#include <MetaNN/data/facilities/tags.h>
#include <vector>

namespace MetaNN
{
template <typename TDevice>
class BaseEvalUnit
{
public:
    using DeviceType = TDevice;
    virtual ~BaseEvalUnit() = default;

    virtual void Eval() = 0;
};
}