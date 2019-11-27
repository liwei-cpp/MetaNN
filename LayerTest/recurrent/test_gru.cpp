#include <MetaNN/meta_nn2.h>
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
    double x_np[] = {+0.06803755, -0.02112342, +0.05661985, +0.05968801,
                     -0.04444506, +0.01079400, -0.00452059, +0.02577419,
                     +0.02714235, +0.04345939, -0.07167949, +0.02139378};
    
    double fres_0[] = {+0.57356697, +0.67493051, +0.29173079, +0.91266167, +0.84974384, +0.98380929,
                       +0.08489671, +0.46299744, +0.03604220, +0.81269711, +0.74108052, +0.94202799,
                       -0.25375497, +0.34222481, -0.00472301, +0.74003857, +0.61487955, +0.86550480};
    
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
                     .Set<LayerInput>(FillMatrix<CheckElement>(1, 4, x_np))
                     .Set<Previous<LayerOutput>>(TrivalMatrix(Scalar<CheckElement, CheckDevice>{1}, 1, 6));

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        
        assert(Compare(res, FillMatrix<CheckElement>(1, 6, fres_0), 0.001f));
        cout << "done" << endl;
    }
    
    void test_gru2()
    {
        cout << "Test GRU layer case 2 (test rnn, train)...\t";
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
                     .Set<LayerInput>(FillMatrixSequence<CheckElement>(3, 1, 4, x_np))
                     .Set<Previous<LayerOutput>>(TrivalMatrix(Scalar<CheckElement, CheckDevice>{1}, 1, 6));

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        
        assert(Compare(res, FillMatrixSequence<CheckElement>(3, 1, 6, fres_0), 0.001f));
        cout << "done" << endl;
    }
}

namespace Test::Layer::Recurrent
{
    void test_gru()
    {
        test_gru1();
        test_gru2();
    }
}