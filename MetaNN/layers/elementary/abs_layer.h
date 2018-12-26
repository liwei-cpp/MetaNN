#pragma once
/*
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
template <typename TPolicies>
class AbsLayer
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

public:
    template <typename TIn>
    auto FeedForward(TIn&& p_in)
    {
        decltype(auto) val = std::forward<TIn>(p_in).template Get<LayerIO>();

        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same<rawType, NullParameter>::value, "parameter is invalid");

        if constexpr (IsFeedbackOutput)
        {
            m_data.push(MakeDynamic(val));
        }
        return LayerIO::Create().template Set<LayerIO>(Abs(std::forward<decltype(val)>(val)));
    }

    template <typename TGrad>
    auto FeedBackward(TGrad&& p_grad)
    {
        if constexpr (IsFeedbackOutput)
        {
            if (m_data.empty())
            {
                throw std::runtime_error("Cannot feed back in SigmoidLayer");
            }
            decltype(auto) tmp = std::forward<TGrad>(p_grad).template Get<LayerIO>();
            auto& tmp2 = m_data.top();
            auto res = LayerIO::Create().template Set<LayerIO>(Sign(tmp2) * std::forward<decltype(tmp)>(tmp));
            m_data.pop();
            return res;
        }
        else
        {
            return LayerIO::Create();
        }
    }

    void NeutralInvariant() const
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
                                                   ElementType, DeviceType,
                                                   CategoryTags::Matrix, CategoryTags::BatchMatrix>;
    DataType m_data;
};
}*/

namespace MetaNN
{
	template <typename TInputMap, typename TPolicies>
	class AbsLayer
	{
		static_assert(IsPolicyContainer<TPolicies>);
		using CurLayerPolicy = PlainPolicy<TPolicies>;

	public:
		static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
		static constexpr bool IsUpdate = false;
		using InputContType = LayerIO;
		using OutputContType = LayerIO;

	private:
		using AimInputType = typename TInputMap::template Find<LayerIO>;
	public:
		template <typename TIn>
		auto FeedForward(TIn&& p_in)
		{
			auto valOri = std::forward<TIn>(p_in).template Get<LayerIO>();
			static_assert(!std::is_same_v<decltype(valOri), NullParameter>);

			auto val = DynamicTransWithFlag<IsDynamic<AimInputType>>(std::move(valOri));
			static_assert(std::is_same_v<decltype(val), AimInputType>);
			
			if constexpr (IsFeedbackOutput)
			{
				m_data.push(val);
			}
			return LayerIO::Create().template Set<LayerIO>(Abs(std::move(val)));
		}
		
		template <typename TGrad>
		auto FeedBackward(TGrad&& p_grad)
		{
			if constexpr (IsFeedbackOutput)
			{
				if (m_data.empty())
				{
					throw std::runtime_error("Cannot feed back in SigmoidLayer");
				}
				auto grad = std::forward<TGrad>(p_grad).template Get<LayerIO>();
				auto& tmp2 = m_data.top();
				auto res = LayerIO::Create().template Set<LayerIO>(Sign(tmp2) * std::move(tmp));
				m_data.pop();
				return res;
			}
			else
			{
				return LayerIO::Create();
			}
		}
		
		void NeutralInvariant() const
		{
			if constexpr(IsFeedbackOutput)
			{
				if (!m_data.empty())
				{
					throw std::runtime_error("NeutralInvariant Fail!");
				}
			}
		}
	public:
		using InputTypeMap = TInputMap;
		using OutputTypeMap = LayerIOMapTrasfer<AbsLayer, InputTypeMap>;
	private:
		std::stack<AimInputType> m_data;
	};
}