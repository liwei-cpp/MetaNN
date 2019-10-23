#pragma once

#include <stdexcept>
#include <MetaNN/data/cardinal/matrix/matrix.h>

namespace MetaNN
{
template<typename TElem, typename TDevice>
class Vector : public Matrix<TElem, TDevice>
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value,
                  "TElem is not an available type");
public:
    using TBase = Matrix<TElem, TDevice>;
    friend struct LowerAccessImpl<Vector>;

public:
    explicit Vector(Shape<CategoryTags::Matrix> p_shape = Shape<CategoryTags::Matrix>())
        : TBase(std::move(p_shape))
    {
        if (TBase::Shape().RowNum() > 1)
        {
            throw std::runtime_error("a vector should contains 1 row.");
        }
    }
    
    explicit Vector(size_t colNum)
        : TBase(1, colNum)
    {}
    
    void SetValue(size_t p_colId, typename TBase::ElementType val)
    {
        TBase::SetValue(0, p_colId, val);
    }

    const auto operator () (size_t p_colId) const
    {
        return TBase::operator() (0, p_colId);
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(TBase(TBase::m_mem, 1, TBase::Shape().ColNum()));
    }
};

template<typename TElem, typename TDevice>
struct LowerAccessImpl<Vector<TElem, TDevice>>
{
    LowerAccessImpl(Vector<TElem, TDevice> p)
        : m_data(std::move(p))
    {}

    auto MutableRawMemory()
    {
        return m_data.m_mem.RawMemory();
    }

    const auto RawMemory() const
    {
        return m_data.m_mem.RawMemory();
    }

private:
    Vector<TElem, TDevice> m_data;
};
}