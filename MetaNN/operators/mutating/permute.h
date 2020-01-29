#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Permute;
}

namespace MetaNN
{
    namespace OperPermute::NSCaseGen
    {
        template <typename TInputHandle, typename TOutputHandle>
        class EvalItem : public BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>
        {
            using BaseType = BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>;
        public:
            using CategoryTag = CategoryTagFromHandle<TOutputHandle>;
            
            template <typename TAuxParams>
            EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle, Shape<CategoryTag::DimNum> shape, const TAuxParams& param)
                : BaseType(std::type_index(typeid(EvalItem)),
                           {oriHandle.DataPtr()}, outputHandle.DataPtr())
                , m_inputHandle(std::move(oriHandle))
                , m_outputHandle(std::move(outputHandle))
                , m_dims(param.Dims())
                , m_outShape(std::move(shape))
            {}
        
            const TInputHandle m_inputHandle;
            TOutputHandle m_outputHandle;
            std::array<size_t, CategoryTag::DimNum> m_dims;
            Shape<CategoryTag::DimNum> m_outShape;
        };
        
        template <typename TInputHandle, typename TOutputHandle>
        class EvalGroup : public TrivalEvalGroup<EvalItem<TInputHandle, TOutputHandle>>
        {
            using EvalItemType = EvalItem<TInputHandle, TOutputHandle>;
            constexpr static size_t uDim = EvalItemType::CategoryTag::DimNum;

        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                const auto& in = evalItem.m_inputHandle.Data();
                const auto& oriShape = in.Shape();
                
                const auto& dims = evalItem.m_dims;
                
                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;
                ResType out(evalItem.m_outShape);

                const size_t count = oriShape.Count();
                assert(count == evalItem.m_outShape.Count());
                
                if (count == 0)
                {
                    evalItem.m_outputHandle.SetData(std::move(out));
                    return;
                }

                std::array<size_t, uDim> aimGaps;
                auto rit1 = evalItem.m_outShape.rbegin();
                auto rit2 = aimGaps.rbegin();
                *rit2 = 1;
                while (true)
                {
                    auto rit2_n = rit2;
                    ++rit2_n;
                    if (rit2_n == aimGaps.rend()) break;
                    *rit2_n = (*rit1) * (*rit2);
                    ++rit1;
                    ++rit2;
                }


                auto low_in = LowerAccess(in);
                const ElementType* mem_in = low_in.RawMemory();

                auto low_out = LowerAccess(out);
                ElementType* mem_out = low_out.MutableRawMemory();

                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
                
                mem_out[0] = mem_in[0];
                
                std::array<size_t, uDim> oriPos{};
                std::array<size_t, uDim> aimPos;
                for (size_t i = 1; i < count; ++i)
                {
                    IncOriPos(oriShape, oriPos);
                    CalAimPos(oriPos, dims, aimPos);
                    size_t index = Pos2Idx(aimPos, aimGaps);
                    mem_out[index] = mem_in[i];
                }
                evalItem.m_outputHandle.SetData(std::move(out));
            }
        private:
            void IncOriPos(const Shape<uDim>& oriShape, std::array<size_t, uDim>& oriPos) const
            {
                auto sit = oriShape.rbegin();
                auto pit = oriPos.rbegin();
                ++(*pit);
                while (*pit == *sit)
                {
                    *pit = 0;
                    ++pit;
                    ++(*pit);
                    ++sit;
                }
            }
            
            void CalAimPos(const std::array<size_t, uDim>& oriPos, const std::array<size_t, uDim>& dims,
                           std::array<size_t, uDim>& aimPos) const
            {
                for (size_t i = 0; i < uDim; ++i)
                {
                    aimPos[i] = oriPos[dims[i]];
                }
            }
            
            size_t Pos2Idx(const std::array<size_t, uDim>& aimPos, const std::array<size_t, uDim>& aimGaps) const
            {
                size_t res = 0;
                for (size_t i = 0; i < uDim; ++i)
                {
                    res += aimPos[i] * aimGaps[i];
                }
                return res;
            }
        };
    }
    
    template <>
    struct OperSeq_<OpTags::Permute>
    {
        using type = OperCalAlgoChain<TailCalculator<OperPermute::NSCaseGen::EvalItem, OperPermute::NSCaseGen::EvalGroup>>;
    };

    template <typename TP>
    constexpr bool IsValidOper<OpTags::Permute, TP> = (IsValidCategoryTag<DataCategory<TP>>) &&
                                                      (DataCategory<TP>::DimNum > 1);

    template <typename TCate>
    class OperAuxParams<OpTags::Permute, TCate>
    {
        constexpr static size_t DimNum = TCate::DimNum;
    public:
        OperAuxParams(std::array<size_t, DimNum> dims)
            : m_dims(dims) {}

        const std::array<size_t, DimNum>& Dims() const { return m_dims; }
        bool operator== (const OperAuxParams& param) const
        {
            return m_dims == param.m_dims;
        }
    private:
        std::array<size_t, DimNum> m_dims;
    };
    
    template <typename TCate>
    class OperShapeInfo<OpTags::Permute, TCate>
    {
        constexpr static size_t uDim = TCate::DimNum;
    public:
        template <typename TOperand>
        OperShapeInfo(const OperAuxParams<OpTags::Permute, TCate>& params,
                      const TOperand& operand)
        {
            auto prevShape = operand.Shape();
            const std::array<size_t, uDim>& dimInfo = params.Dims();
            
            for (size_t i = 0; i < uDim; ++i)
            {
                m_shape[i] = prevShape[dimInfo[i]];
            }
        }
    
        const auto& Shape() const
        {
            return m_shape;
        }
    
    private:
        MetaNN::Shape<uDim> m_shape;
    };

    namespace OperPermute
    {
        template <size_t uDim>
        bool ValidDims(std::array<size_t, uDim> dims)
        {
            bool isPlainDim = true;
            for (size_t i = 0; i < uDim; ++i)
            {
                if (dims[i] != i)
                {
                    isPlainDim = false;
                    break;
                }
            }
            if (isPlainDim) return false;
            
            std::sort(dims.begin(), dims.end());
            for (size_t i = 0; i < uDim; ++i)
            {
                if (dims[i] != i) return false;
            }
            return true;
        }
    }

    template <typename TP,
              typename = std::enable_if_t<IsValidOper<OpTags::Permute, TP>>>
    auto Permute(TP&& oper, std::array<size_t, DataCategory<TP>::DimNum> dims)
    {
        assert(OperPermute::ValidDims(dims));
        
        using rawOp = RemConstRef<TP>;
        using ResType = Operator<OpTags::Permute, rawOp>;
        return ResType(OperAuxParams<OpTags::Permute, OperCateCal<OpTags::Permute, rawOp>>(std::move(dims)),
                       std::forward<TP>(oper));
    }
}