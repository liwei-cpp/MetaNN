#pragma once
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
namespace NSSoftmaxLayer
{
template <bool isFeedback>
struct FeedbackOut_
{
    template <typename T, typename DataType>
    static auto RecordData(const T& p_in, DataType& data)
    {
        auto res = MakeDynamic(p_in);
        data.push(res);
        return res;
    }

    template <typename TGrad, typename DataType>
    static auto FeedBack(DataType& data, const TGrad& grad)
    {
        if (data.empty())
        {
            throw std::runtime_error("Cannot do FeedBackward for Softmax Layer");
        }

        auto tmp = VecSoftmaxDerivative(grad.template Get<LayerIO>(), data.top());
        auto res = LayerIO::Create().template Set<LayerIO>(tmp);
        data.pop();
        return res;
    }
};

template <>
struct FeedbackOut_<false>
{
    template <typename T, typename DataType>
    static auto RecordData(T&& val, DataType&)
    {
        return std::forward<T>(val);
    }

    template <typename TGrad, typename DataType>
    static auto FeedBack(const DataType&, const TGrad&)
    {
        return LayerIO::Create();
    }
};
}

template <typename TPolicies>
class SoftmaxLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = false;
    using InputType = LayerIO;
    using OutputType = LayerIO;

private:
    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device;

private:
    using FeedbackOut_ = NSSoftmaxLayer::FeedbackOut_<IsFeedbackOutput>;
    
public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val = p_in.template Get<LayerIO>();

        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same<rawType, NullParameter>::value, "parameter is invalid");

        auto tmp = VecSoftmax(val);
        auto tmp2 = FeedbackOut_::RecordData(tmp, m_data);
        return LayerIO::Create().template Set<LayerIO>(tmp2);
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        return FeedbackOut_::FeedBack(m_data, p_grad);
    }

    void NeutralInvariant()
    {
        if constexpr(IsFeedbackOutput)
        {
            if (!m_data.empty())
            {
                throw std::runtime_error("NeutralInvariant Fail!");
            }
        }
    }
    
private:
    using DataType = LayerTraits::LayerInternalBuf<IsFeedbackOutput,
                                                   PolicySelect<InputPolicy, CurLayerPolicy>::BatchMode,
                                                   typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element,
                                                   typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device,
                                                   CategoryTags::Matrix, CategoryTags::BatchMatrix>;
    DataType m_data;
};
}
