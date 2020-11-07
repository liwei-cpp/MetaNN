#pragma once
#include <MetaNN/facilities/_.h>
#include <MetaNN/operation/facilities/_.h>
#include <MetaNN/policies/_.h>
#include <MetaNN/facilities/cont_metafuns/value_sequential.h>
#include <numeric>

namespace MetaNN::OpTags
{
    struct Reshape;
}

namespace MetaNN
{
    constexpr static size_t InferredDimSize = (size_t)-1;

    namespace OperReshape::NSCaseGen
    {
        template <typename TInputHandle, typename TOutputHandle>
        class EvalItem : public BaseEvalItem
        {
        public:
            using CategoryTag = CategoryTagFromHandle<TOutputHandle>;

            EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle, Shape<CategoryTag::DimNum> shape)
                : BaseEvalItem(TypeID<EvalItem>(),
                               {oriHandle.DataPtr()}, outputHandle.DataPtr())
                , m_inputHandle(std::move(oriHandle))
                , m_outputHandle(std::move(outputHandle))
                , m_outShape(std::move(shape))
            {}
        
            const TInputHandle m_inputHandle;
            TOutputHandle m_outputHandle;
            Shape<CategoryTag::DimNum> m_outShape;
        };

        template <typename TInputHandle, typename TOutputHandle>
        class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle>>
        {
            using EvalItemType = EvalItem<TInputHandle, TOutputHandle>;

        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                const auto& in = evalItem.m_inputHandle.Data();
                assert(in.Shape().Count() == evalItem.m_outShape.Count());
                auto low_in = LowerAccess(in);
                
                using ResType = typename TOutputHandle::DataType;
                if constexpr (IsScalar<ResType>)
                {
                    ResType out(low_in.SharedMemory());
                    evalItem.m_outputHandle.SetData(std::move(out));
                }
                else
                {
                    ResType out(low_in.SharedMemory(), evalItem.m_outShape);
                    evalItem.m_outputHandle.SetData(std::move(out));
                }
            }
        };
    }
    
    template <>
    struct OperSeq_<OpTags::Reshape>
    {
        using type = OperCalAlgoChain<TailCalculator<OperReshape::NSCaseGen::EvalItem,
                                                     OperReshape::NSCaseGen::EvalGroup,
                                                     PolicyContainer<PPassShape>>>;
    };
    
    template <typename TCate, typename TPolicies>
    class OperShapeInfo<OpTags::Reshape, TCate, TPolicies>
    {
        constexpr static size_t uDim = TCate::DimNum;
    public:
        template <typename TOperAuxParams, typename TOperand>
        OperShapeInfo(const TOperAuxParams& params, const TOperand&)
            : m_shape(params.m_shape)
        {}
    
        const auto& Shape() const
        {
            return m_shape;
        }
    
    private:
        MetaNN::Shape<uDim> m_shape;
    };
    
    template <typename TElem, typename TCate>
    struct OperAuxParams<OpTags::Reshape, TElem, TCate>
    {
        template <typename T>
        OperAuxParams(MetaNN::Shape<TCate::DimNum> new_shape, const T& old_shape)
            : m_shape(new_shape)
        {
            const size_t ori_shape_count = old_shape.Count();
            if constexpr(TCate::DimNum == 0)
            {
                if (ori_shape_count != 1)
                {
                    throw std::runtime_error("Reshape error: cannot reshape to a scalar with more than 1 items.");
                }
            }
            else
            {
                size_t inferred_pos = (size_t)-1;
                size_t current_count = 1;
                for (size_t i = 0; i < m_shape.size(); ++i)
                {
                    if (m_shape[i] == InferredDimSize)
                    {
                        if (inferred_pos == (size_t)-1)
                        {
                            inferred_pos = i;
                        }
                        else
                        {
                            throw std::runtime_error("Reshape error: too many inferrence positions.");
                        }
                    }
                    else
                        current_count *= m_shape[i];
                }
                if (inferred_pos == (size_t)-1)
                {
                    if (current_count != ori_shape_count)
                    {
                        throw std::runtime_error("Reshape error, element count mismatch.");
                    }
                }
                else
                {
                    if (ori_shape_count % current_count != 0)
                    {
                        throw std::runtime_error("Reshape error, element count mismatch.");
                    }
                    m_shape[inferred_pos] = ori_shape_count / current_count;
                }
            }
        }
        
        MetaNN::Shape<TCate::DimNum> m_shape;
    };
    
    template <typename TPolicy, typename TOperand>
    struct OperCategory_<OpTags::Reshape, TPolicy, TOperand>
    {
        constexpr static size_t uDim = PolicySelect<DimPolicy, TPolicy>::DimCount;
        using type = CategoryTags::Tensor<uDim>;
    };
    
    template <typename TP, size_t uDim,
              std::enable_if_t<IsValidOper<OpTags::Reshape, TP>>* = nullptr>
    auto Reshape(TP&& oper, MetaNN::Shape<uDim> newShape)
    {
        using rawOp = RemConstRef<TP>;
        using ResType = Operation<OpTags::Reshape, OperandContainer<rawOp>,
                                  PolicyContainer<PDimCountIs<uDim>>>;

        OperAuxParams<OpTags::Reshape, typename ResType::ElementType, CategoryTags::Tensor<uDim>> aux(std::move(newShape), oper.Shape());
        return ResType(std::move(aux), std::forward<TP>(oper));
    }
}