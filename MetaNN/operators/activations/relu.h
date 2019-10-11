#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct ReLU;
    struct ReLUGrad;
}

namespace MetaNN
{
namespace OperReLU::NSCaseGen
{
template <typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams&)
        : m_inputHandle(std::move(oriHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in = m_inputHandle.Data();
        using ResType = typename TOutputHandle::DataType;
        using ElementType = typename ResType::ElementType;
        ResType out(in.Shape());

        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        const ElementType zero{};
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = (mem_in[i] > zero) ? mem_in[i] : zero;
        }
        m_outputHandle.SetData(std::move(out));
    }
    
private:
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::ReLU>
{
    using type = OperSeqContainer<TailCalculator<OperReLU::NSCaseGen::EvalUnit>>;
};

template <typename TP,
          typename = std::enable_if_t<IsValidOper<OpTags::ReLU, TP>>>
auto ReLU(TP&& p_m)
{
    using rawM = RemConstRef<TP>;
    using ResType = Operator<OpTags::ReLU, rawM>;
    return ResType(std::forward<TP>(p_m));
}
}

/// Gradient
namespace MetaNN
{
namespace OperReLUGrad::NSCaseGen
{
template <typename TGradHandle, typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TGradHandle gradHandle, TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams&)
        : m_gradHandle(std::move(gradHandle))
        , m_inputHandle(std::move(oriHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& grad = m_gradHandle.Data();
        const auto& in = m_inputHandle.Data();
        assert(grad.Shape() == in.Shape());
        
        using ResType = typename TOutputHandle::DataType;
        using ElementType = typename ResType::ElementType;
        ResType out(in.Shape());

        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_grad = LowerAccess(grad);
        ElementType* mem_grad = low_grad.MutableRawMemory();
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        ElementType zero{};
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = (mem_in[i] > zero) ? mem_grad[i] : zero;
        }
        m_outputHandle.SetData(std::move(out));
    }
    
private:
    const TGradHandle m_gradHandle;
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::ReLUGrad>
{
    using type = OperSeqContainer<TailCalculator<OperReLUGrad::NSCaseGen::EvalUnit>>;
};

template <typename TGrad, typename TInput,
          typename = std::enable_if_t<IsValidOper<OpTags::ReLUGrad, TGrad, TInput>>>
auto ReLUGrad(TGrad&& p_grad, TInput&& p_input)
{
    using ResType = Operator<OpTags::ReLUGrad, RemConstRef<TGrad>, RemConstRef<TInput>>;
    return ResType(std::forward<TGrad>(p_grad), std::forward<TInput>(p_input));
}
}