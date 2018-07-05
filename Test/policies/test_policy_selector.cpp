#include "test_policy_selector.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
struct AccPolicy
{
    using MajorClass = AccPolicy;
    
    struct AccuTypeCate
    {
        struct Add;
        struct Mul;
    };
    using Accu = AccuTypeCate::Add;

    struct IsAveValueCate;
    static constexpr bool IsAve = false;

    struct ValueTypeCate;
    using Value = float;
};

#include <MetaNN/policies/policy_macro_begin.h>
TypePolicyObj (PAddAccu,     AccPolicy, Accu,  Add);
TypePolicyObj (PMulAccu,     AccPolicy, Accu,  Mul);
ValuePolicyObj(PAve,         AccPolicy, IsAve, true);
ValuePolicyObj(PNoAve,       AccPolicy, IsAve, false);
TypePolicyTemplate (PValueTypeIs,  AccPolicy, Value);
ValuePolicyTemplate(PAvePolicyIs, AccPolicy, IsAve);
#include <MetaNN/policies/policy_macro_end.h>

template <typename...TPolicies>
struct Accumulator
{
    using TPoliCont = PolicyContainer<TPolicies...>;
    using TPolicyRes = PolicySelect<AccPolicy, TPoliCont>;

    using ValueType = typename TPolicyRes::Value;
    static constexpr bool is_ave = TPolicyRes::IsAve;
    using AccuType = typename TPolicyRes::Accu;

public:
    template <typename TIn>
    static auto Eval(const TIn& in)
    {
        if constexpr(std::is_same<AccuType, AccPolicy::AccuTypeCate::Add>::value)
        {
            ValueType count = 0;
            ValueType res = 0;
            for (const auto& x : in)
            {
                res += x;
                count += 1;
            }

            if constexpr (is_ave)
                return res / count;
            else
                return res;
        }
        else if constexpr (std::is_same<AccuType, AccPolicy::AccuTypeCate::Mul>::value)
        {
            ValueType res = 1;
            ValueType count = 0;
            for (const auto& x : in)
            {
                res *= x;
                count += 1;
            }
            if constexpr (is_ave)
                return pow(res, 1.0 / count);
            else
                return res;
        }
        else
        {
            static_assert(DependencyFalse<AccuType>);
        }
    }
};
}

void test_policy_selector()
{
    cout << "Test policy selector...\t";

    const int a[] = {1, 2, 3, 4, 5};
    assert(fabs(Accumulator<>::Eval(a) - 15) < 0.0001);
    assert(fabs(Accumulator<PMulAccu>::Eval(a) - 120) < 0.0001);
    assert(fabs(Accumulator<PMulAccu, PAve>::Eval(a) - pow(120.0, 0.2)) < 0.0001);
    assert(fabs(Accumulator<PAve, PMulAccu>::Eval(a) - pow(120.0, 0.2)) < 0.0001);
    assert(fabs(Accumulator<PAvePolicyIs<true>, PMulAccu>::Eval(a) - pow(120.0, 0.2)) < 0.0001);
    cout << "done" << endl;
}