#pragma once

#include <MetaNN/data/linear_table/linear_table.h>
#include <MetaNN/data/linear_table/3d_array.h>
#include <MetaNN/data/linear_table/matrix.h>
#include <MetaNN/data/linear_table/scalar.h>

namespace MetaNN
{
template<typename TElem, typename TDevice, typename TCategory>
class Sequence : public LinearTable<TElem, TDevice, TCategory>
{
    using TBase = LinearTable<TElem, TDevice, TCategory>;
    
public:
    using TBase::TBase;
    
    bool operator == (const Sequence& val) const
    {
        const TBase& base = static_cast<const TBase&>(val);
        return TBase::operator== (base);
    }
    
    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }
    
    auto Length() const
    {
        return TBase::Count();
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
};

template <typename TElem, typename TDevice, typename TCategory>
struct LowerAccessImpl<Sequence<TElem, TDevice, TCategory>>
    : LowerAccessImpl<LinearTable<TElem, TDevice, TCategory>>
{
    using TBase = LowerAccessImpl<LinearTable<TElem, TDevice, TCategory>>;
    
public:
    using TBase::TBase;
};

template <typename TElement, typename TDevice, typename TCategory>
struct DataCategory_<Sequence<TElement, TDevice, TCategory>>
{
    using type = CategoryTags::Sequence<TCategory>;
};
}