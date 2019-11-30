#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    double weight_input[] =
        {-0.09661144, -0.33417121, +0.29103363, +0.50999618, -0.23182100, +0.24369442,
         +0.66899598, +0.36953580, -0.51747060, -0.26284057, +0.28918779, +0.55565894,
         +0.66740763, +0.21685427, -0.09278971, -0.41988072, +0.70715761, -0.09363341,
         +0.34229791, -0.22610682, +0.58880997, +0.60940993, +0.13732076, +0.65681124};

    double weight_update[] =
        {-0.66653228, -0.47682026, -0.67518288, -0.40545496, -0.36147988, +0.01941973,
         +0.69609451, +0.25287008, -0.74357712, +0.72910321, +0.06159645, +0.25983655,
         +0.04027188, +0.60454583, -0.06552815, +0.62309813, -0.19332850, +0.04896450,
         -0.64127958, -0.23409408, -0.67684901, +0.54364264, +0.40317559, -0.71374387};

    double weight_reset[] =
        {-0.53187048, +0.77293050, +0.17450088, -0.00994116, +0.04143310, -0.33568737,
         -0.15345633, -0.43647453, -0.31598645, +0.73241997, +0.41814876, -0.22857052,
         -0.57352620, +0.02003479, +0.21309507, -0.32143161, -0.15456516, +0.47672486,
         -0.60603082, +0.52535045, +0.03762549, +0.42038560, +0.60655487, +0.64915311};

    double trans_input[] =
        {+0.70399410, +0.13278353, -0.19926924, +0.04356903, +0.35045522, +0.52788538,
         -0.63065779, -0.45202187, +0.07422507, +0.36386842, +0.18230617, +0.46815771,
         +0.52402252, -0.47640398, +0.11233097, -0.27676845, -0.65701407, +0.67802221,
         -0.60481840, -0.15317315, -0.06706792, +0.69611615, +0.35044616, +0.34480113,
         -0.70122135, +0.58410794, +0.26500583, +0.10885358, +0.47127050, +0.57044607,
         +0.59831005, +0.45211726, -0.56619442, +0.53402656, +0.60157329, +0.68390757};

    double trans_reset[] =
        {-0.14363223, +0.63675302, +0.16913497, -0.44189492, -0.56120068, +0.26084465,
         +0.44514757, +0.59415108, -0.30962875, -0.31645194, -0.52880931, -0.16519666,
         +0.26052433, -0.49828398, +0.40446800, +0.07982349, -0.00644308, +0.35322934,
         +0.58120221, +0.53890330, -0.27229568, -0.11808503, +0.36836761, -0.18573779,
         -0.02476197, +0.19951814, -0.07490581, -0.46724612, +0.68554229, -0.29110119,
         -0.40188420, -0.09623235, -0.38734370, +0.57530755, +0.61518854, -0.37863937};

    double trans_update[] =
        {+0.11948478, -0.47503161, -0.45790726, +0.19722390, -0.60824418, -0.63365418,
         -0.36145496, +0.34658331, -0.36760733, +0.36732060, -0.41768077, -0.48393381,
         -0.49159506, -0.60170573, +0.42115003, -0.57490540, -0.05455971, +0.70709771,
         +0.32830757, +0.63654321, +0.32902294, -0.51632595, +0.45209199, -0.41814247,
         -0.52965844, -0.63281918, +0.22141445, +0.02858138, +0.10368818, +0.55148059,
         +0.41502982, +0.03049517, +0.66101068, -0.59646982, +0.36144584, -0.52966756};

    // input sequence, one line per vector, sequence length = 3
    double x_np_1[] = {+0.06803755, -0.02112342, +0.05661985, +0.05968801,
                       -0.04444506, +0.01079400, -0.00452059, +0.02577419,
                       +0.02714235, +0.04345939, -0.07167949, +0.02139378};
    
    double fres_1[] = {+0.57356697, +0.67493051, +0.29173079, +0.91266167, +0.84974384, +0.98380929,
                       +0.08489671, +0.46299744, +0.03604220, +0.81269711, +0.74108052, +0.94202799,
                       -0.25375497, +0.34222481, -0.00472301, +0.74003857, +0.61487955, +0.86550480};

    // input sequence, 2 lines per matrix, sequence length = 3
    double x_np_2[] = {+0.06803755, -0.02112342, +0.05661985, +0.05968801,
                       +0.08232947, -0.06048973, -0.03295545, +0.05364592,
                       -0.04444506, +0.01079400, -0.00452059, +0.02577419,
                       -0.02704310, +0.00268018, +0.09044594, +0.08323903,
                       +0.02714235, +0.04345939, -0.07167949, +0.02139378,
                       -0.09673989, -0.05142265, -0.07255369, +0.06083535};

    double fres_2[] = {+0.57356697, +0.67493051, +0.29173079, +0.91266167, +0.84974384, +0.98380929,
                       +0.55533367, +0.67668849, +0.30710635, +0.92483586, +0.82781047, +0.98345935,
                       +0.08489671, +0.46299744, +0.03604220, +0.81269711, +0.74108052, +0.94202799,
                       +0.11790840, +0.45316586, +0.06124194, +0.82058430, +0.75081849, +0.94535780,
                       -0.25375497, +0.34222481, -0.00472301, +0.74003857, +0.61487955, +0.86550480,
                       -0.23477679, +0.34515449, +0.01637596, +0.74365497, +0.62452525, +0.86549950};

    double wi_grad_check[] = {-0.04264588, -0.01554555, -0.03181295, -0.01012502, -0.02110841, +0.00796517,
                              +0.02034394, +0.02356272, +0.01869094, +0.01973352, +0.03446945, +0.00592688,
                              -0.02382816, -0.01072955, -0.00267675, -0.01318462, +0.00054733, +0.00702775,
                              -0.02760490, -0.08487224, -0.02311299, -0.07875377, -0.11316016, -0.03615509};

    double wu_grad_check[] = {+0.03632787, +0.02715230, +0.01062525, +0.00892575, +0.01664936, -0.00250984,
                              -0.01864217, -0.01733737, -0.00582790, -0.00920409, -0.01183222, -0.00222107,
                              +0.01316092, +0.00946078, +0.00133530, +0.00743668, +0.00213212, -0.00179100,
                              +0.02557347, +0.03541989, +0.00772383, +0.02940711, +0.02389065, +0.0124517};

    double wr_grad_check[] = {-0.01041598, +0.00345089, +0.00921119, +0.00381854, -0.00276361, -0.00800987,
                              +0.00684211, -0.00091481, -0.00587776, +0.00235688, +0.00611044, +0.01163567,
                              -0.00513169, +0.00180021, +0.00335244, +0.00136408, -0.00014323, -0.00600464,
                              -0.01480510, -0.00092086, +0.01130431, -0.01644023, -0.02170095, -0.04062329};

    double check_up_grad[] = {+0.48058826, +0.54069245, +0.13667561, +0.38400191, +0.35677141, +0.12427084,
                              +0.45646650, +0.57106030, +0.13630769, +0.43481576, +0.38776311, +0.16771854,
                              +0.47375917, +0.49797082, +0.13775741, +0.32429212, +0.32719505, +0.09297007,
                              +0.43777096, +0.62169272, +0.13541570, +0.51348776, +0.43199876, +0.22456039,
                              +0.43995771, +0.60675037, +0.13567832, +0.49119174, +0.42036501, +0.21021718,
                              +0.43016893, +0.63637161, +0.13516138, +0.53686970, +0.44570836, +0.24260066};

    double check_reset_grad[] = {-0.22573137, +0.00850728, +0.17660162, -0.16386601, -0.25808012, -0.50331688,
                                 -0.23692220, -0.00422012, +0.18385042, -0.22045365, -0.31345046, -0.59030676,
                                 -0.20450531, +0.01547305, +0.16353773, -0.11952476, -0.21175016, -0.41683775,
                                 -0.25851166, -0.01967195, +0.19749409, -0.29627436, -0.38922000, -0.71662235,
                                 -0.25175691, -0.01601686, +0.19325995, -0.27675560, -0.36948425, -0.68223709,
                                 -0.26449898, -0.02474073, +0.20129845, -0.32011148, -0.41287124, -0.75509369};

    double check_input_grad[] = {-0.36018932, -0.65483344, -0.26400137, -0.59727877, -0.86820155, -0.21833292,
                                 -0.41349605, -0.95949709, -0.32700250, -0.87920940, -1.28881669, -0.36934814,
                                 -0.21169925, -0.32412082, -0.16103597, -0.29028437, -0.43350112, -0.09657308,
                                 -0.13432479, -0.58535790, -0.13005032, -0.54306334, -0.79876238, -0.27465478,
                                 -0.28591725, -0.97712284, -0.25488314, -0.90252972, -1.32828522, -0.43310648,
                                 -0.18858892, -0.58639282, -0.16131029, -0.54136020, -0.79330409, -0.25235096};
    void test_gru1()
    {
        cout << "Test GRU layer case 1 (test step, inference)...\t";
        using RootLayer = MakeInferLayer<GruStep>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root", 4, 6);

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/W",  FillMatrix<CheckElement>(4, 6, weight_input));
        initializer.SetParam("root/Wz", FillMatrix<CheckElement>(4, 6, weight_update));
        initializer.SetParam("root/Wr", FillMatrix<CheckElement>(4, 6, weight_reset));
        initializer.SetParam("root/U",  FillMatrix<CheckElement>(6, 6, trans_input));
        initializer.SetParam("root/Uz", FillMatrix<CheckElement>(6, 6, trans_update));
        initializer.SetParam("root/Ur", FillMatrix<CheckElement>(6, 6, trans_reset));
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
        
        auto input = LayerInputCont<RootLayer>()
                     .Set<LayerInput>(FillMatrix<CheckElement>(1, 4, x_np_1))
                     .Set<Previous<LayerOutput>>(TrivalMatrix(Scalar<CheckElement, CheckDevice>{1}, 1, 6));

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        
        assert(Compare(res, FillMatrix<CheckElement>(1, 6, fres_1), 0.001f));
        cout << "done" << endl;
    }
    
    void test_gru2()
    {
        cout << "Test GRU layer case 2 (test rnn, infer)...\t";
        using RootLayer = MakeInferLayer<RecurrentLayer, PActFuncIs<GruStep>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root", 4, 6);

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/kernel/W",  FillMatrix<CheckElement>(4, 6, weight_input));
        initializer.SetParam("root/kernel/Wz", FillMatrix<CheckElement>(4, 6, weight_update));
        initializer.SetParam("root/kernel/Wr", FillMatrix<CheckElement>(4, 6, weight_reset));
        initializer.SetParam("root/kernel/U",  FillMatrix<CheckElement>(6, 6, trans_input));
        initializer.SetParam("root/kernel/Uz", FillMatrix<CheckElement>(6, 6, trans_update));
        initializer.SetParam("root/kernel/Ur", FillMatrix<CheckElement>(6, 6, trans_reset));
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
        
        auto input = LayerInputCont<RootLayer>()
                     .Set<LayerInput>(FillMatrixSequence<CheckElement>(3, 1, 4, x_np_1))
                     .Set<Previous<LayerOutput>>(TrivalMatrix(Scalar<CheckElement, CheckDevice>{1}, 1, 6));

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        
        assert(Compare(res, FillMatrixSequence<CheckElement>(3, 1, 6, fres_1), 0.001f));
        cout << "done" << endl;
    }
    
    void test_gru3()
    {
        cout << "Test GRU layer case 3 (test rnn, infer-2)...\t";
        using RootLayer = MakeInferLayer<RecurrentLayer, PActFuncIs<GruStep>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root", 4, 6);

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/kernel/W",  FillMatrix<CheckElement>(4, 6, weight_input));
        initializer.SetParam("root/kernel/Wz", FillMatrix<CheckElement>(4, 6, weight_update));
        initializer.SetParam("root/kernel/Wr", FillMatrix<CheckElement>(4, 6, weight_reset));
        initializer.SetParam("root/kernel/U",  FillMatrix<CheckElement>(6, 6, trans_input));
        initializer.SetParam("root/kernel/Uz", FillMatrix<CheckElement>(6, 6, trans_update));
        initializer.SetParam("root/kernel/Ur", FillMatrix<CheckElement>(6, 6, trans_reset));
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
        
        auto input = LayerInputCont<RootLayer>()
                     .Set<LayerInput>(FillMatrixSequence<CheckElement>(3, 2, 4, x_np_2))
                     .Set<Previous<LayerOutput>>(TrivalMatrix(Scalar<CheckElement, CheckDevice>{1}, 2, 6));

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(Compare(res, FillMatrixSequence<CheckElement>(3, 2, 6, fres_2), 0.001f));

        cout << "done" << endl;
    }
    
    using InputMap = LayerIOMap<LayerKV<LayerInput, MatrixSequence<CheckElement, CheckDevice>>,
                                LayerKV<Previous<LayerOutput>, TrivalMatrix<Scalar<CheckElement, CheckDevice>>>
                               >;
    void test_gru4()
    {
        cout << "Test GRU layer case 4 (test rnn, train)...\t";
        using RootLayer = MakeTrainLayer<RecurrentLayer, InputMap, PActFuncIs<GruStep>, PUpdate>;
        static_assert(RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root", 4, 6);

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/kernel/W",  FillMatrix<CheckElement>(4, 6, weight_input));
        initializer.SetParam("root/kernel/Wz", FillMatrix<CheckElement>(4, 6, weight_update));
        initializer.SetParam("root/kernel/Wr", FillMatrix<CheckElement>(4, 6, weight_reset));
        initializer.SetParam("root/kernel/U",  FillMatrix<CheckElement>(6, 6, trans_input));
        initializer.SetParam("root/kernel/Uz", FillMatrix<CheckElement>(6, 6, trans_update));
        initializer.SetParam("root/kernel/Ur", FillMatrix<CheckElement>(6, 6, trans_reset));
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
        
        auto input = LayerInputCont<RootLayer>()
                     .Set<LayerInput>(FillMatrixSequence<CheckElement>(3, 2, 4, x_np_2))
                     .Set<Previous<LayerOutput>>(TrivalMatrix(Scalar<CheckElement, CheckDevice>{1}, 2, 6));

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        
        assert(Compare(res, FillMatrixSequence<CheckElement>(3, 2, 6, fres_2), 0.001f));
        
        auto grad = LayerOutputCont<RootLayer>()
                    .Set<LayerOutput>(res * -1);
        layer.FeedBackward(grad);
        
        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        
        const auto& w_grad = Evaluate(grad_collector.GetGradInfo<CategoryTags::Matrix>("root/kernel/W").Grad());
        assert(Compare(w_grad, FillMatrix<CheckElement>(4, 6, wi_grad_check), 0.001f));
        
        const auto& wz_grad = Evaluate(grad_collector.GetGradInfo<CategoryTags::Matrix>("root/kernel/Wz").Grad());
        assert(Compare(wz_grad, FillMatrix<CheckElement>(4, 6, wu_grad_check), 0.001f));
        
        const auto& wr_grad = Evaluate(grad_collector.GetGradInfo<CategoryTags::Matrix>("root/kernel/Wr").Grad());
        assert(Compare(wr_grad, FillMatrix<CheckElement>(4, 6, wr_grad_check), 0.001f));

        const auto& u_grad = Evaluate(grad_collector.GetGradInfo<CategoryTags::Matrix>("root/kernel/U").Grad());
        assert(Compare(u_grad, FillMatrix<CheckElement>(6, 6, check_input_grad), 0.001f));
        
        const auto& uz_grad = Evaluate(grad_collector.GetGradInfo<CategoryTags::Matrix>("root/kernel/Uz").Grad());
        assert(Compare(uz_grad, FillMatrix<CheckElement>(6, 6, check_up_grad), 0.001f));
        
        const auto& ur_grad = Evaluate(grad_collector.GetGradInfo<CategoryTags::Matrix>("root/kernel/Ur").Grad());
        assert(Compare(ur_grad, FillMatrix<CheckElement>(6, 6, check_reset_grad), 0.001f));

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
}

namespace Test::Layer::Recurrent
{
    void test_gru()
    {
        test_gru1();
        test_gru2();
        test_gru3();
        test_gru4();
    }
}