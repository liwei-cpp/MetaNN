#pragma once

namespace MetaNN
{
template <typename TOpTag, typename TCate>
class OperAuxParams
{
public:
    bool operator == (const OperAuxParams&) const
    {
        return true;
    }
};
}