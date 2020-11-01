#pragma once

#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/facilities/traits.h>
#include <type_traits>
#include <stdexcept>

namespace MetaNN
{
    namespace NSTensor
    {
        template <typename TShape, typename TCurIndex, typename TNextParam, typename... TRemainParam>
        auto OffsetAndVal(const TShape& shape, size_t& gap, TCurIndex curIdx, TNextParam nextParam, TRemainParam... remPara)
        {
            constexpr size_t uDimNum = TShape::DimNum;
            constexpr size_t indexPos = uDimNum - sizeof...(TRemainParam) - 1;
            if (static_cast<size_t>(curIdx) >= shape[indexPos])
                throw std::runtime_error("Invalid dimension index.");

            if constexpr (sizeof...(TRemainParam) == 0)
            {
                return std::pair(static_cast<size_t>(curIdx), nextParam);
            }
            else
            {
                size_t curGap = 1;
                auto [pos, val] = OffsetAndVal(shape, curGap, nextParam, remPara...);
                gap = curGap * shape[indexPos + 1];
                pos += static_cast<size_t>(curIdx) * gap;
                return std::pair(pos, val);
            }
        }
    }

    template <typename TElem, typename TDevice, size_t uDim>
    class Tensor
    {
        static_assert(std::is_same_v<RemConstRef<TElem>, TElem>);
        static_assert(uDim > 0);

    public:
        using CategoryTag = CategoryTags::Tensor<uDim>;
        using ElementType = TElem;
        using DeviceType = TDevice;

        friend struct LowerAccessImpl<Tensor>;

    public:
        template <typename... TShapeParameter>
        explicit Tensor(TShapeParameter... shapes)
            : m_shape(shapes...)
            , m_mem(m_shape.Count())
        {}
        
        explicit Tensor(ContinuousMemory<ElementType, DeviceType> p_mem,
                        MetaNN::Shape<uDim> p_shape)
            : m_shape(std::move(p_shape))
            , m_mem(std::move(p_mem))
        {}

        const auto& Shape() const noexcept
        {
            return m_shape;
        }
    
        bool operator== (const Tensor& val) const
        {
            return (m_shape == val.m_shape) &&
                   (m_mem == val.m_mem);
        }
        
        bool AvailableForWrite() const { return !m_mem.IsShared(); }

        template <typename... TPosValParams>
        void SetValue(TPosValParams... posValParams)
        {
            static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                          "Only CPU supports this method.");
            static_assert(sizeof...(TPosValParams) == uDim + 1);

            assert(AvailableForWrite());
            size_t gap = 1;
            auto [pos, val] = NSTensor::OffsetAndVal(m_shape, gap, posValParams...);
            (m_mem.RawMemory())[pos] = static_cast<ElementType>(val);
        }
        
        template <typename... TPosParams>
        ElementType operator()(TPosParams... posParams) const
        {
            static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                          "Only CPU supports this method.");
            auto pos = m_shape.IndexToOffset(posParams...);
            return (m_mem.RawMemory())[pos];
        }
        
        const auto operator [] (size_t id) const
        {
            if constexpr (uDim > 1)
            {
                using AimType = Tensor<ElementType, DeviceType, uDim - 1>;
                MetaNN::Shape<uDim - 1> aimShape;
                size_t count = 1;
                for (size_t i = 1; i < uDim; ++i)
                {
                    const size_t curDim = m_shape[i];
                    aimShape[i - 1] = curDim;
                    count *= curDim;
                }

                const size_t pos = id * count;
                if (pos >= m_shape.Count())
                {
                    throw std::runtime_error("ID out of bound.");
                }
                return AimType(m_mem.Shift(pos), aimShape);
            }
            else
            {
                using AimType = Tensor<ElementType, DeviceType, 0>;
                if (id >= m_shape[0])
                {
                    throw std::runtime_error("ID out of bound.");
                }
                return AimType(m_mem.Shift(id));
            }
        }
        
        auto EvalRegister() const
        {
            return MakeConstEvalHandle(*this);
        }
    private:
        MetaNN::Shape<uDim> m_shape;
        ContinuousMemory<ElementType, DeviceType> m_mem;
    };

    template <typename TElem, typename TDevice>
    class Tensor<TElem, TDevice, 0>
    {
        static_assert(std::is_same<RemConstRef<TElem>, TElem>::value);

    public:
        using CategoryTag = CategoryTags::Tensor<0>;
        using ElementType = TElem;
        using DeviceType = TDevice;

        friend struct LowerAccessImpl<Tensor>;

    public:        
        explicit Tensor(ElementType elem = ElementType())
            : m_mem(1)
        {
            SetValue(elem);
        }
        
        explicit Tensor(MetaNN::Shape<0>)
            : Tensor() {}

        explicit Tensor(ContinuousMemory<ElementType, DeviceType> p_mem)
            : m_mem(std::move(p_mem))
        {
            assert(m_mem.Size() >= 1);
        }

        const auto& Shape() const noexcept
        {
            const static MetaNN::Shape<0> shape;
            return shape;
        }

        bool AvailableForWrite() const
        {
            return !m_mem.IsShared();
        }

        void SetValue(ElementType val)
        {
            assert(AvailableForWrite());
            (m_mem.RawMemory())[0] = val;
        }

        auto Value() const noexcept
        {
            return (m_mem.RawMemory())[0];
        }
    
        bool operator== (const Tensor& val) const noexcept
        {
            return (Value() == val.Value());
        }

        auto EvalRegister() const
        {
            return MakeConstEvalHandle(*this);
        }

    private:
        ContinuousMemory<ElementType, DeviceType> m_mem;
    };

    template <typename TElement, typename TDevice, size_t uDIm>
    struct LowerAccessImpl<Tensor<TElement, TDevice, uDIm>>
    {
        LowerAccessImpl(Tensor<TElement, TDevice, uDIm> p)
            : m_data(std::move(p))
        {}

        TElement* MutableRawMemory()
        {
            return m_data.m_mem.RawMemory();
        }

        const TElement* RawMemory() const
        {
            return m_data.m_mem.RawMemory();
        }
        
        auto SharedMemory()
        {
            return m_data.m_mem;
        }

    private:
        Tensor<TElement, TDevice, uDIm> m_data;
    };

    template <typename TElem, typename TDevice>
    using Scalar = Tensor<TElem, TDevice, 0>;

    template <typename TElem, typename TDevice>
    using Vector = Tensor<TElem, TDevice, 1>;

    template <typename TElem, typename TDevice>
    using Matrix = Tensor<TElem, TDevice, 2>;

    template <typename TElem, typename TDevice>
    using ThreeDArray = Tensor<TElem, TDevice, 3>;
}