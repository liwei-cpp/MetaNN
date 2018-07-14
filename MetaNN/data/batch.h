#pragma once

#include <MetaNN/data/linear_table/linear_table.h>
#include <MetaNN/data/linear_table/3d_array.h>
#include <MetaNN/data/linear_table/matrix.h>
#include <MetaNN/data/linear_table/scalar.h>

namespace MetaNN
{
template<typename TElem, typename TDevice, typename TCategory>
class Batch : public LinearTable<TElem, TDevice, TCategory>
{
    using TBase = LinearTable<TElem, TDevice, TCategory>;
    
public:
    using TBase::TBase;
    
    Batch(TBase val)
        : TBase(std::move(val)) {}
    
    bool operator == (const Batch& val) const
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
    
    auto BatchNum() const
    {
        return TBase::Count();
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
};

template <typename TElem, typename TDevice, typename TCategory>
struct LowerAccessImpl<Batch<TElem, TDevice, TCategory>>
    : LowerAccessImpl<LinearTable<TElem, TDevice, TCategory>>
{
    using TBase = LowerAccessImpl<LinearTable<TElem, TDevice, TCategory>>;
    
public:
    using TBase::TBase;
};

template <typename TElement, typename TDevice, typename TCategory>
struct DataCategory_<Batch<TElement, TDevice, TCategory>>
{
    using type = CategoryTags::Batch<TCategory>;
};

template <typename TElem, typename TDevice>
auto SubMatrix(const Batch<TElem, TDevice, CategoryTags::Matrix>& input,
               size_t p_rowB, size_t p_rowE, size_t p_colB, size_t p_colE)
{
    auto res = input.SubMatrix(p_rowB, p_rowE, p_colB, p_colE);
    return Batch<TElem, TDevice, CategoryTags::Matrix>(std::move(res));
}
}