#pragma once
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/model_rel/param_initializer/facilities/traits.h>

namespace MetaNN
{
namespace NSWeightLayer
{
template <typename TWeight, typename TIn>
auto EvalHelper(const TWeight& p_weight, const TIn& p_in)
{
    return Dot(p_in, p_weight);
}
}

template <typename TPolicies>
class WeightLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsUpdate;
    using InputType = LayerIO;
    using OutputType = LayerIO;

private:
    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device;

public:
    WeightLayer(std::string p_name, size_t p_inLen, size_t p_outLen)
        : m_name(std::move(p_name))
        , m_inputLen(p_inLen)
        , m_outputLen(p_outLen)
    {
        if ((m_inputLen == 0) || (m_outputLen == 0))
        {
            throw std::runtime_error("Invalidate matrix size for weight layer");
        }
    }

public:
    template <typename TInitializer, typename TBuffer, 
              typename TInitPolicies = typename TInitializer::PolicyCont>
    void Init(TInitializer& initializer, TBuffer& loadBuffer, std::ostream* log = nullptr)
    {
        if (auto cit = loadBuffer.find(m_name); cit != loadBuffer.end())
        {
            const Matrix<ElementType, DeviceType>& m = cit->second;
            if ((m.RowNum() != m_inputLen) || (m.ColNum() != m_outputLen))
            {
                throw std::runtime_error("Load matrix error in WeightLayer");
            }
            m_weight = m;
            if (log)
            {
                std::string logInfo = "Load from load buffer: " + m_name + '\n';
                (*log) << logInfo;
            }
            return;
        }
        else if (initializer.IsMatrixExist(m_name))
        {
            m_weight = Matrix<ElementType, DeviceType>(m_inputLen, m_outputLen);            
            initializer.GetMatrix(m_name, m_weight);
            loadBuffer[m_name] = m_weight;
            if (log)
            {
                std::string logInfo = "Copy from initializer: " + m_name + '\n';
                (*log) << logInfo;
            }
            return;
        }
        else
        {
            m_weight = Matrix<ElementType, DeviceType>(m_inputLen, m_outputLen);
            using CurInitializer = PickInitializer<TInitPolicies, InitPolicy::WeightTypeCate>;
            if constexpr (!std::is_same<CurInitializer, void>::value)
            {
                auto& cur_init = initializer.template GetFiller<CurInitializer>();
                cur_init.Fill(m_weight, m_inputLen, m_outputLen);
                loadBuffer[m_name] = m_weight;
                if (log)
                {
                    std::string logInfo = "Random init from initializer: " + m_name + '\n';
                    (*log) << logInfo;
                }
            }
            else
            {
                throw std::runtime_error("Cannot get initializer for InitPolicy::WeightTypeCate");
            }
        }
    }

    template <typename TSave>
    void SaveWeights(TSave& saver) const
    {
        typename TSave::const_iterator cit = saver.find(m_name);
        if ((cit != saver.end()) && (cit->second != m_weight))
        {
            throw std::runtime_error("Duplicate save for matrix: " + m_name);
        }
        saver[m_name] = m_weight;
    }

    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val = p_in.template Get<LayerIO>();

        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same<rawType, NullParameter>::value, "parameter is invalid");

        if constexpr (IsUpdate)
        {
            m_updateInfo.push(MakeDynamic(val));
        }
        
        auto res = NSWeightLayer::EvalHelper(m_weight, val);
        return LayerIO::Create().template Set<LayerIO>(std::move(res));
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        if constexpr (IsUpdate)
        {
            auto tmp = p_grad.template Get<LayerIO>();
            if (m_updateInfo.empty())
            {
                throw std::runtime_error("Cannot do FeedBackward for Weight Layer");
            }

            auto tw = Transpose(m_updateInfo.top());
            m_updateInfo.pop();
            auto res = NSWeightLayer::EvalHelper(tmp, tw);
            m_gradInfo.push(MakeDynamic(res));
        }
        
        if constexpr (IsFeedbackOutput)
        {
            auto tmp = p_grad.template Get<LayerIO>();
            auto tw = Transpose(m_weight);
            auto res = NSWeightLayer::EvalHelper(tw, tmp);
            return LayerIO::Create().template Set<LayerIO>(std::move(res));
        }
        else
        {
            return LayerIO::Create();
        }
    }

    template <typename TGradCollector>
    void GradCollect(TGradCollector& col)
    {
        if constexpr (IsUpdate)
        {
            LayerTraits::MatrixGradCollect(m_weight, m_gradInfo, col);
        }
    }

    void NeutralInvariant() const
    {
        if constexpr(IsUpdate)
        {
            if ((!m_updateInfo.empty()) || (!m_gradInfo.empty()))
            {
                throw std::runtime_error("NeutralInvariant Fail!");
            }
        }
    }

private:
    const std::string m_name;
    const size_t m_inputLen;
    const size_t m_outputLen;

    Matrix<ElementType, DeviceType> m_weight;
    
    using DataType = LayerTraits::LayerInternalBuf<IsUpdate,
                                                   PolicySelect<InputPolicy, CurLayerPolicy>::BatchMode,
                                                   ElementType, DeviceType,
                                                   CategoryTags::Matrix, CategoryTags::BatchMatrix>;
    DataType m_updateInfo;
    DataType m_gradInfo;
};
}
