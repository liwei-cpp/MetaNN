# MetaNN
MetaNN 是一个深度学习框架的骨架，旨在探索 C++ 模板元编程在深度学习中的应用。它不支持多机并发训练。目前来说，它只支持 CPU ，包含了一些简单的构造。但 MetaNN 具有足够的扩展性，能够相对容易地支持诸如 GPU 或 FPGA 这样的设备。就目前来说，它所包含的计算逻辑只是示例性的，其速度并不快，但其框架非常灵活，基于这套框架进行的程序优化可以在保证易用性的同时，极大地提升深度学习系统的性能。

本文会简述 MetaNN 的主体设计思想，核心组件以及优化的可能性。

## 代码组织与运行环境
MetaNN 是一个 C\+\+ 模板库，与其它模板库类似，它的核心逻辑都包含在头文件中（以 .h 结尾）。除了这些头文件外，MetaNN 还包含了一个示例工程： GeneralTest ，它会调用 MetaNN 中包含的逻辑实现具体的计算。 GeneralTest 主要是为了测试而引入的，但读者也可以阅读这其中的代码来了解 MetaNN 的调用方式。

### 运行环境
MetaNN 的头文件以及 GeneralTest 均使用 CodeLite 进行开发。 GeneralTest 在 GCC 7 上编译并运行。 不建议使用较低版本的 GCC 编译本工程。对于该分支而言，我没有试过其它的编译器（另一个 book 分支支持 clang 等编译环境）。MetaNN 使用了 C\+\+ 17 中的特性。理论上只要完美支持 C\+\+17 的编译器，均可以编译并运行 GeneralTest ，但我并不能保证程序的每一行代码都完全符合 C\+\+ 17 这一标准，同时如果编译器存在 Bug ，则无法保证能够编译。读者可以选择自己的编译器进行试验。这里推荐的运行环境是 Ubuntu & GCC 7。

MetaNN 是自成体系的。除了对 C\+\+17 的依赖之外，作者有意没有依赖任何第三方库。这样不仅方便运行环境的搭建，也可以使程序的阅读者不会因对某个库的不熟悉而影响对程序的理解（当然，要想理解这个库，需要对 C\+\+17 以及元编程有一定的了解）。

### 代码组织
整个框架核心逻辑被包含在 MetaNN 目录中。与 MetaNN 目录同级的是 GeneralTest 目录，其中包含了相应的测试程序。测试目录（GeneralTest）与 逻辑目录（MetaNN）的组织形式基本相同，因此这里仅以 MetaNN 目录为例说明代码的组织形式。

 * **_root**: 只包含了一个文件 meta_nn.h ，引用这个文件即可使用 MetaNN 所提供的全部功能
 * **data**: 存放数据的目录，其中包含了深度学习框架可能会使用到的各种数据类型
 * **data_copy**: 保存了用于复制数据的逻辑。目前该目录中只包含了一个函数，用于将一个 CPU 矩阵赋予另一个 CPU 矩阵。后续会根据需要引入更多的数据复制方法
 * **evaluate**: 保存了 MetaNN 中的求值逻辑
 * **facilities**: 包含了一些辅助（元）函数与结构
 * **layers**: “层”的实现逻辑，层是整个深度学习框架的核心
 * **model_res**: 目标是存放一些模型相关的逻辑，如参数初始化，梯度收集等。目前这个目录中只包含了一个梯度收集器，后续会根据需要引入更多的功能
 * **operators**: 包含了层所使用的操作算子
 * **policies**: 包含了 Policy 的相关实现。 Policy 可以被视为一种编译期的分支机制，在 MetaNN 中主要用于层的策略定制。我们会在后文讨论层的设计时说明 Policy 所起的作用

接下来，文本将从底向上依次讨论框架中的若干组件。阐述其设计理念并给出一些使用示例。

## 数据类型
### 设计理念
与很多深度学习框架不同， MetaNN 并没有引入诸如 Tensor 或 Blob 这样的通用构造。 MetaNN 在数据类型上禀承了如下的设计理念：
 * **概念清晰**：作者认为，像 Tensor 这样的构造其概念过于宽泛了——它可以用于表示矩阵或矩阵的集合，也可以表示三维或更高维的信息。表示的信息不同，其使用方式也会有所区别。 这种设计方式在使用之初具有一定的优势，但如果编写相对复杂的程序，则容易产生误用，同时使用单一或少量的概念来描述数据，无法涵盖很多特殊的情况。 MetaNN 采用了多种方式来表示数据，对所表示的数据进行分类，并支持天然的扩展。它其中并不包含诸如 Tensor 这样的数据类型，而是包含了像```Batch<Matrix<>>```这样的模板，用于表示批处理的结果。
 * **简化接口**：同样出于概念清晰的考虑， MetaNN 中的数据类型并不包含很多框架中所提供的诸如```reshape```这样的接口。因为这些接口并非与深度学习理论中的概念直接相关，仅仅是为了提升系统性能而引入的小技巧。这种技巧的存在同样会使得程序变得复杂难懂。 MetaNN 采用了其它的方式来实现效果类似的优化，比如引入内存池来减少内存的分配与释放。
 * **支持数据扩展与分类**： MetaNN 的数据类型支持若干维度的扩展。除了前文所述的可以引入新的数据类型外，还可以对已有数据类型（实际上是数据模板）所包含的元素类型（如```float```或者```double```或者其它的定点类型）进行扩展；对数据类型所支持的计算设备（如```CPU```或```GPU```）进行扩展。 MetaNN 中的数据类型按照其类别进行划分，目前包含了矩阵类别、矩阵列表类别与标量类别（注意并非类型），可以在此基础上扩展出新的数据类别。

### 基本用法
可以用如下的方式声明一个包含了 10 行，20 列的矩阵对象：
```cpp
Matrix<int, DeviceTags::CPU> matrix(10, 20);
```
从声明中不难看出，该对象中的数据元素类型为```int```，所支持的计算设备为```CPU```。类似地，还可以用如下的方式声明其它的矩阵类型对象：
```cpp
// 列向量，包含100个元素，第37个元素为1,其余为0
OneHotColVector<int, DeviceTags::CPU> m1(100, 37);

// 一个 10*20 的矩阵，其中元素值均为100
TrivalMatrix<int, DeviceTags::CPU> m2(10, 20, 100);

// 一个 10*20 的矩阵，其中元素值均为0
ZeroMatrix<int, DeviceTags::CPU> m3(10, 20)
```
以上所声明的对象均为 Matrix 类别。也可以声明 BatchMatrix 类别的矩阵列表：
```cpp
// 矩阵列表对象，包含了 7 个 10*20 的矩阵
Batch<Matrix<float, DeviceTags::CPU>> bm(10, 20， 7);
```
我们还可以声明如下类型的矩阵列表：
```cpp
// 矩阵列表对象，其中的每个元素都是一个 TrivalMatrix
Batch<TrivalMatrix<int, DeviceTags::CPU>> ...
```

### 类别划分
上述声明的类型不同，可以应用于不同的场景。每种类型都采用了不同的方式存储其内部数据，并提供不同的访问接口。MetaNN 将这些类型划分成不同的类别，比如：```Matrix<>```，```OneHotColVector<>```与```ZeroMatrix<>```等都属于矩阵；而```Batch<Matrix<>>```则属于矩阵列表。可以用如下的方式获取特定数据类型所对应的类别：
```cpp
IsMatrix<Matrix<int, DeviceTags::CPU>>;             // true
IsMatrix<Batch<Matrix<int, DeviceTags::CPU>>>;      // false
IsBatchMatrix<Batch<Matrix<int, DeviceTags::CPU>>>;	// true
IsBatchMatrix<Matrix<int, DeviceTags::CPU>>;	    // false
```
或者使用如下的方式：
```cpp
// CategoryTags::Matrix
DataCategory<Matrix<int, DeviceTags::CPU>>;

// CategoryTags::BatchMatrix
DataCategory<Batch<Matrix<int, DeviceTags::CPU>>>;
```

除了自身特有的接口外，具有相同类别的数据类型提供一组通用的接口。比如，矩阵类别的对象需要提供接口返回行数与列数，并提供求值接口以转换为```Matrix<>```类型的对象（我们会在后面讨论求值的过程）。

### 引入新的类型
我们可以很简单地为 MetaNN 引入新的类型：
```cpp
template <typename TElem, typename TDevice>
class MyMatrix;

MyMatrix<float, DeviceTags::CPU> ...
```
在此基础上，通过简单的模板特化，就可以将类型纳入到 MetaNN 的类型体系之中：
```cpp
template <typename TElem, typename TDevice>
constexpr bool IsMatrix<MyMatrix<float, DeviceTags::CPU>> = true;
```
自定义的类型与 MetaNN 已经定义的类型地位相同。


### 动态类型
除了上述类型外， MetaNN 还引入了一个特殊的类型模板： ```DynamicData<>```。它用于对不同的数据类别进行封装，掩盖其底层的具体类型信息，只暴露出该数据类型最核心的特性与接口。比如：
```cpp
DynamicData<float, DeviceTags::CPU, CategoryTags::Matrix>
```
它可以被视为一个容器，其中可以放置元素类型为```float```，设备类型为```CPU```的矩阵：
```cpp
vector<DynamicData<float, DeviceTags::CPU, CategoryTags::Matrix>> vec;

OneHotColVector<float, DeviceTags::CPU> m1(100, 37);
vec.push_back(MakeDynamic(m1));

ZeroMatrix<float, DeviceTags::CPU> m2(10, 20)
vec.push_back(MakeDynamic(m2));
```
```MakeDynamic```是 MetaNN 提供的一个函数，给定一个具体的数据类型，可以转换为相应的```Dynamic```版本。

同样的，我们也可以将矩阵列表置于```DynamicData<... CategoryTags::BatchMatrix>```这个容器中。

```DynamicData``` 用于保存中间变量，它掩盖了其底层所包含的具体的数据类型。便于存储类型不同（类型未知）但类别相同（类别已知）的数据。但 ```DynamicData``` 的引入导致了具体信息类型的丢失，对泛型编程会产生一些不好的影响。因此，```DynamicData``` 只会在必要时才被使用。

### 判等
MetaNN 的每个数据类型都提供了 ```operator ==``` 与 ```operator !=```来与其它数据进行比较。但这里的比较仅供后续求值优化所使用。我们并不希望提供一个元素级的比较——即比较矩阵中的每个元素：一方面，浮点类型的数据无法精确比较；另一方面，这样的比较很耗时。MetaNN 中所提供的是一种快速的比较方案：
 * 如果 A 复制自 B，那么 A 与 B 相等
 * 如果 A 复制自 B，同时 C 与 D 是 A 与 B 求值的结果，那么 C 与 D 相等

除此之外， MetaNN 并不对相等性进行任何保证。

## 操作
MetaNN 包含了多个操作符。每个操作符都可以接收一到多个数据对象，返回操作结果。

### 设计理念
MetaNN 使用表达式模板作为操作的结果。表达式模板对象的内部包含了表示该操作的操作数对象，它并不进行实际的计算。因此基于操作来构造操作结果模板的复杂度非常低。帮达式模板实际的求值过程被推迟到显式调用求值时进行——这就为求值过程的优化提供了空间。

一方面，表达式模板类型可以被视为操作结果；另一方面，也可以将其看成一种特殊数据。与 MetaNN 中基本的数据类型相似，操作产生的表达式模板对象也会被划分为矩阵、矩阵列表等类别，并提供相应分类所需要支持的接口。这样，我们就可以将某个操作的结果用做另一个操作的操作数了。

MetaNN 是富类型的，并非任意类型都可以做为某个操作的操作数。 MetaNN 设计了相关的机制来防止操作数的误用（当然，我们可以通过扩展的方式使得某种操作支持新的操作数或操作数的组合）。

### 基本用法
可以通过如下的方式来调用 MetaNN 中的操作并生成相应的表达式模板：
```cpp
auto rm1 = Matrix<int, DeviceTags::CPU>(4, 5);
auto rm2 = Matrix<int, DeviceTags::CPU>(4, 5);
auto add = rm1 + rm2;
```
```add``` 就是由两个操作数所形成的表达式模板。其实际类型为：
```cpp
BinaryOp<BinaryOpTags::Add,
         Matrix<int, DeviceTags::CPU>,
         Matrix<int, DeviceTags::CPU>>
```
通常来说，我们在实际的程序设计过程中不需要关心 ```add``` 的实际类型（只需要像上例中那样使用 ```auto``` 让编译器自动推导，或者使用前文所提供的动态类型对其进行封装即可）。但在这里，还是有必要了解一下 ```add``` 对象对应的类型所包含的信息的。

```add``` 对象的实际类型是 ```BinaryOp``` 模板实例化的结果。这表明这个类型是一个“二元操作符”。二元操作符的具体内容则由模板参数 ```BinaryOpTags::Add``` 给出。```BinaryOp``` 的第二个与第三个模板参数分别表示了两个操作数的类型。最后两个模板参数则表示了模板表达式所使用的元素与设备类型。

模板表达式本身也可以视为一种数据，因此，我们可以这样写：
```cpp
auto rm1 = Matrix<int, DeviceTags::CPU>(4, 5);
auto rm2 = Matrix<int, DeviceTags::CPU>(4, 5);
auto add = rm1 + rm2;
auto rm3 = ZeroMatrix<int, DeviceTags::CPU>(4, 5);
auto sub = add - rm3;
```
此时， ```sub``` 所对应的类型为：
```cpp
BinaryOp<BinaryOpTags::Substract,
         BinaryOp<BinaryOpTags::Add,
                  Matrix<int, DeviceTags::CPU>,
                  Matrix<int, DeviceTags::CPU>>,
         ZeroMatrix<int, DeviceTags::CPU>>
```

本质上，模板表达式形成了一种树的结构。而模板表达式的求值过程，也可以被视为一种树的遍历过程。这个过程可以被优化。我们将在后续讨论求值时讨论相关的优化方法。

### 模板表达式的分类
模板表达式也可以被视为数据， MetaNN 自动实现了模板表达式的分类。比如：
```cpp
IsMatrix<BinaryOp<BinaryOpTags::Substract,
                  BinaryOp<BinaryOpTags::Add,
                           Matrix<int, DeviceTags::CPU>,
                           Matrix<int, DeviceTags::CPU>>,
                  ZeroMatrix<int, DeviceTags::CPU>>;    // true
```
这是因为两个矩阵相加的结果还是矩阵，而矩阵与矩阵相减的结果还是矩阵。

### 各式各样的操作
需要说明的是，并非所有的操作都像加减法这样“一目了然”——其操作数与操作结果的类别相一致。比如， MetaNN 中定义了 Collapse 操作，用于将矩阵列表中的对应元素相加，得到一个矩阵。此时，操作的输入输出类型就不相同了。 MetaNN 的操作各式各样，也可以相对容易地扩展，可以在扩展的同时指定操作数与操作结果的类别，以及一些其它的信息。

### 操作数的支持
MetaNN 是富类型的，一个使用 MetaNN 的程序可能会在操作中使用多种类型。对于特定的操作，并非所有的操作数类型都是合法的。 MetaNN 引入了特定的元函数来描述某个操作可以接收的合法的操作数。同样以加法为例，相应的元函数为：
```cpp
template <typename TP1, typename TP2>
struct OperAdd_
{
// valid check
private:
    using rawM1 = RemConstRef<TP1>;
    using rawM2 = RemConstRef<TP2>;

public:
    static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>) ||
                                  (IsMatrix<rawM1> && IsScalar<rawM2>) ||
                                  (IsScalar<rawM1> && IsMatrix<rawM2>);
    // ...
}
```
只有 ```OperAdd_<TP1, TP2>::valid``` 为真时，类型为 ```TP1``` 与 ```TP2``` 的两个操作数才能相加。从上述声明中不难看出，目前加法只支持矩阵与矩阵相加，矩阵与标量相加，以及标量与矩阵相加：
```cpp
float a, b;
// 这不会触发 MetaNN 中的操作逻辑
float c = a + b;

Matrix<int, DeviceTags::CPU> rm1(2, 3);
Batch<Matrix<int, DeviceTags::CPU>> rm2(2, 3);
// 触发 MetaNN 中的操作逻辑，由于不支持矩阵与矩阵列表相加，因此编译出错
auto rm3 = rm1 + rm2;
```

当然，操作所支持的操作数组合也是可以扩展的。比如，我们可以为加法扩展出支持矩阵列表的版本。但要在修改 ```OperAdd_::valid``` 变量的同时为这种类型的加法行为给出明确的定义。

## 基本层
层是 MetaNN 中相对高级的组件。当前很多深度学习框架都抛弃了层的概念，转而将操作与层的概念合并起来。 MetaNN 认为区分层与操作，可以更好地描述不同层面上的抽象。本节讨论基本层。

### 设计理念
层是 MetaNN 对外的接口。基本层封装了操作，调用具体的操作实现数据的正向、反向传播。每个基本层都能实现特定的功能，比如将两个矩阵相加，或者求矩阵的点乘。虽然基本层的功能相对单一，但其行为细节是可以配制的。比如，我们可以指定某个层是否要在训练的过程中更新其中所包含的参数。层应该了解其具体的行为细节，并根据其进行相应的优化。

通常情况下，层的具体行为极少在程序的执行过程中发生改变。因此，可以在编译期指定相应的信息，以确保尽早地利用这些信息进行最大限度的优化。

每个层都应当支持正向、反向传播。除此之外，层可能需要根据其自身特性提供若干额外的接口函数。比如，如果层中包含了参数矩阵，那么它就需要提供接口来对其所包含的矩阵进行初始化、读取或加载矩阵中的内容等。

这些接口大部分都以模板方法的形式提供。这就确保了可以采用不同类型的参数调用层所提供的方法——这一点对正向、反向传播尤其重要——它与表达式模板一起构成了性能优化的前提。我们会在后面看到这一点。

层在设计与使用上有很多独特之处，可以说，它是 MetaNN 框架中极富特色的一个组成部分。 MetaNN 中的很多子模块的引入都与层相关。本节将讨论基本层及其涉及到的子模块，下一节将在此基础上讨论组合层的一些特色之处。

### Policy 与层的声明
#### Policy 简述
我们希望能对层的具体行为进行控制，进一步，希望在编译期就指定层的具体行为——从而尽量早地获取信息，提升系统优化的空间。 MetaNN 使用 Policy 机制来实现这一点。

Policy 是一种编译期构造，它指定了函数或方法在使用时的具体行为。一个典型的 Policy 为：使用乘法而非加法进行累积——累积函数可以根据其中的信息改变其缺省的行为。

MetaNN 中所使用的 Policy 是被系统化地组织起来的。其中的每个 Policy 都包含了一个主体类别，次要类别，以及相关的缺省值。主体类别用于 Policy 的分类，次要类别用于 Policy 对象的互斥（我们很快就会讨论到 Policy 对象）：
```cpp
struct FeedbackPolicy {
    struct IsUpdateValue;
    struct IsFeedbackOutputValue;

    static constexpr bool IsUpdate = false;
    static constexpr bool IsFeedbackOutput = false;
};
```
这段代码定义了两个 Policy ： ```IsUpdate``` 与 ```IsFeedbackOutput``` 。其主体类别为 ```FeedbackPolicy``` ——这表示了两个 Policy 与反向传播相关。它们所属的次要类别分别为 ```IsUpdateValue``` 与 ```IsFeedbackOutputValue``` （次要类别使用 Policy 的名称作为前缀，这一点是专门设计的；而次要类别名称中的 ```value``` 则表示了该类别所对应的 Policy 是数值 Policy）。它们的缺省值均为 ```false``` 。

MetaNN 不仅支持数值 Policy ，还支持类型与枚举 Policy （由于 Policy 是编译期的构造，因此支持类型与枚举就并不奇怪了）：
```cpp
struct OperandPolicy {
    struct ElementTypeType;
    using ElementType = float;
};

struct SingleLayerPolicy {
    struct ActionTypeEnum {
        struct Sigmoid;
        struct Tanh;
    };
    using ActionType = ActionTypeEnum::Sigmoid;

    // ...
};
```
```ElementType``` 是类型 Policy ，其次要类型为 ```ElementTypeType``` ，缺省“值” 为 ```float``` 。 ```ActionType``` 是枚举 Policy ，其次要类型为 ```ActionTypeEnum``` ，缺省值为 ```ActionTypeEnum::Sigmoid``` 。

事实上， MetaNN 中的 Policy 还能是其它形式的，比如我们甚至可以定义取值为类模板的 Policy 。但数值、类型与枚举 Policy 在 MetaNN 中使用的频率最高， MetaNN 也为这三种 Policy 专门提供了宏以建立 Policy 对象。

#### Policy 对象
Policy 与 Policy 对象的关系很像类与其对象的关系。 Policy 对象可以被视做某个 Policy 实例化的结果。Policy 对象包含了相应 Policy 的主体与次要类别信息，但其取值可能不是 Policy 的缺省值：
```cpp
ValuePolicyObj(PUpdate,   FeedbackPolicy, IsUpdate, true);
ValuePolicyObj(PNoUpdate, FeedbackPolicy, IsUpdate, false);
```
上述代码定义了两个 Policy 对象：```PUpdate``` 与 ```PNoUpdate``` 。它们均对应于之前提及的 ```IsUpdate``` Policy。但其取值一个为 ```true``` ，一个为 ```false``` 。类似的，还可以定义取值为类型或枚举的 Policy 对象：
```cpp
TypePolicyObj(PDoubleElement,  OperandPolicy, ElementType, double);
EnumPolicyObj(PTanhAction, SingleLayerPolicy, ActionType, Tanh);
```
接下来，让我们看一下如何使用 Policy 对象来指定层的行为细节。

#### 在层中使用 Policy 对象
MetaNN 中的大部分层都被声明为模板。接收一个模板参数，该模板参数是一个“容器”，可以存放零到多个 Policy 对象：
```cpp
template <typename TPolicies>
class WeightLayer;

// layer is an object with interfaces such as feed-back and feed-forward;
WeightLayer<PolicyContainer<PRowVecInput, PUpdate, PFeedbackOutput>> layer;
```
上述程序使用 Policy 对象的列表作为模板参数，实例化了 ```WeightLayer``` 模板。可以利用 MetaNN 所提供的元函数使得定义更加清晰易懂：
```cpp
using MyLayer = MakeLayer<WeightLayer,
                          PRowVecInput, PUpdate, PFeedbackOutput>;
MyLayer layer;
```

Policy 对象的顺序并不重要。以下两个层的功能是相同的（虽然它们的 C\+\+ 类型并不相同）：
```cpp
using MyLayer1 = MakeLayer<WeightLayer, PRowVecInput, PUpdate>;
using MyLayer2 = MakeLayer<WeightLayer, PUpdate, PRowVecInput>;
```

但并非所有的组合都是有效的，比如下面的声明就是非法的：
```cpp
using MyLayer1 = MakeLayer<WeightLayer, PNoUpdate, PUpdate>;
```
显然，这是因为一个层不能同时具有“更新内部参数”以及“不更新内部参数”两种状态。声明这样的层会导致编译器报错：在同一个层的声明中包含了多个属于相同次要类别的 Policy 对象。

对于特定的层来说，并非所有的 Policy 对象都会对其行为产生影响。比如，```AddLayer``` 并不包含内部参数，因此设置 ```PUpdate``` 并不会导致相应的参数更新。此时，相应 Policy 对象设置与否不会对层的行为细节产生影响。也即，下面的两个声明具有相同的行为：
```cpp
// MyLayer1 = AddLayer<PolicyContainer<PUpdate>>
using MyLayer1 = MakeLayer<AddLayer, PUpdate>;

// MyLayer2 = AddLayer<PolicyContainer<>>
using MyLayer2 = MakeLayer<AddLayer>;
```

另一方面，如果某个层需要根据特定的 Policy 来决定其行为，但该 Policy 所对应的对象在层的实例化时并未指定，那么该层将使用该 Policy 所对应的缺省值来确定其行为细节。因此，下面两个声明具有相同的行为细节：
```cpp
using MyLayer1 = MakeLayer<BiasLayer, PNoUpdate>;
using MyLayer2 = MakeLayer<BiasLayer>;
```

最后，让我们少许深入一下层的实现，看一下层是如何获取 Policy 所对应的值的：
```cpp
template <typename TPolicies>
class BiasLayer {
public:
    static constexpr bool IsFeedbackOutput
    	= PolicySelect<FeedbackPolicy, TPolicies>::IsFeedbackOutput;
```
实际的实现过程与这里所讨论的有少许出入（因为涉及到组合层，需要进一步处理）。但这里所展示的代码并不影响我们对逻辑的理解。 MetaNN 提供了元函数 ```PolicySelect``` 来选择 Policy 。它接收 ```PolicyContainer``` 列表以及要查询的 Policy 的主体、次要类别，返回相应的 Policy 值（当然，对于类型或枚举 Policy ，则返回相应的类型与枚举信息）。

层会在其内部根据 Policy 的取值对其行为进行调整，尽量减少不必要的计算。

### 正向、反向传播
除了构造函数外，每个层都至少需要提供两个接口，用于进行数据的正向、反向传播。正向、反向传播的输入与输出均是容器。正向传播的输入容器与反向传播的输出容器具有相同的“结构”，在 MetaNN 中称为层的输入容器；正向传播的输出容器与反向传播的输入容器具有相同的“结构”，在 MetaNN 中称为层的输出容器。本节将首先讨论容器的一些特性，之后对正向、反向传播函数的一些特点进行讨论。

#### 容器
以正向传播为例，通常来说，层会接收一到多个数据作为输出，计算后产生一到多个数据作为输出。一些深度学习框架都使用类似 ```std::vector``` 这样的线性表来输入与输出的数据结构——输入数据用一个 ```vector``` 表示，输出数据用另一个 ```vector``` 表示。这种方法有两个缺点：首先，无法通过 ```vertor``` 数据结构本身来区分传递给层的参数。比如，对于一个计算了两个矩阵相减的层，可以规定传入 ```vector``` 的第一个参数为减数，第二个为被减数。但这种规定并没有从程序的角度有所体现，可能会存在误用。

其次的一个问题是 ```vector``` 这样的列表只能存储类型相同的元素。层的输入与输出数据类型千变万化，使用 ```vector``` 这样的列表无疑限制了输入输出的数据类型。

MetaNN 定义了一种编译期容器解决上述问题。使用该容器的第一步是声明其结构：
```cpp
struct A; struct B; struct Weight;
struct FParams : public NamedParams<A, B, Weight> {};
```
这里定义了一个容器 ```NamedParams<A, B, Weight>``` 。其中 ```A, B, Weight``` 为关键字，每个关键字可以对应存储一个数据对象。为了简化后续的使用，这里利用派生为该容器引入了一个别名 ```FParams``` 。当然，也可以通过如下的方式引入别名：
```cpp
struct A; struct B; struct Weight;
using FParams = NamedParams<A, B, Weight>;
```
进一步，可以将上述代码进一步简化为：
```cpp
using FParams = NamedParams<struct A, struct B, struct Weight>;
```
使用派生的方法能够获得一个额外的好处。如果容器中只需要包含一个 Key ，那么我们可以让容器名与 Key 的名称相同：
```cpp
struct LayerIO : public NamedParams<LayerIO> {};
```

声明了容器的构造后，接下来就可以建立容器对象并向其中添加数据了：
```cpp
auto data = FParams::Create().Set<A>(true)
                             .Set<B>(std::string("abc"))
                             .Set<Weight>(15);
```
上述代码并不难理解，它建立了 ```FParams``` 容器的实例，并为每个 Key 指定了一个相应的 Value。

这里有几点需要说明：首先，不难看出，每个 Key 所对应的 Value 类型可以是不同的。这似乎有点像一些弱类型语言所提供的支持。但 MetaNN 还是依赖于 C\+\+ 这个强类型的语言进行的构造，考虑下面的代码：
```cpp
auto data1 = FParams::Create().Set<A>(true)
                              .Set<B>(std::string("abc"))
                              .Set<Weight>(15);

auto data2 = FParams::Create().Set<A>(12)
                              .Set<B>(3.5)
                              .Set<Weight>(std::vector<int>());

data1 = data2;    // error!
```
```data1``` 与 ```data2``` 的类型是不同的：它们所对应数值的类型有所区别。

其次，```FParams::Create().Set...``` 之后得到的结果类型并非 ```FParams``` ，而是由 ```FParams``` 所推导出的一个类型。用户不需要关注具体的类型，只需要使用 ```auto``` 来定义变量，由编译器自动推导即可。

第三，容器中元素的设置顺序并不需要与容器声明时 Key 的顺序相同，我们也可以这样写：
```cpp
auto data = FParams::Create().Set<B>(std::string("abc"))
                             .Set<A>(true)
                             .Set<Weight>(15);
```

可以通过如下的方式获取容器中的对象：
```cpp
auto a = data.template Get<A>();
```
注意编译器会进行自动推导，为 ```a``` 指定正确的数据类型：比如 ```A``` 对应的对象为 ```float``` 型的，那么在执行上述代码后， ```a``` 也就是 ```float``` 类型的变量了。

#### 容器与正向、反向传播
MetaNN 中的层接提供了正向、反向传播的接口，这些接口接收容器对象作为参数，返回容器对象作为结果。以 ```AddLayer``` 为例：
```cpp
struct LayerIO : public NamedParams<LayerIO> {};
struct AddLayerInput : public NamedParams<struct AddLayerIn1,
                                          struct AddLayerIn2> {};
...

template <typename TPolicies>
class AddLayer {
	// ...
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val1 = p_in.template Get<AddLayerIn1>();
        const auto& val2 = p_in.template Get<AddLayerIn2>();

        return OutputType::Create().template Set<LayerIO>(val1 + val2);
    }

    template <typename TGrad>
    auto FeedBackward(TGrad&& p_grad)
    {
    	const auto& res = p_grad.template Get<LayerIO>();
	    return AddLayerInput::Create().template Set<AddLayerIn1>(res)
    	                              .template Set<AddLayerIn2>(res);
    }
};
```
通过使用容器，我们避免了前文所述的使用 ```vector``` 这样的结构所带来的问题：首先，容器中的每个对象都使用 Key 来检索，这减少了使用错误的可能性。其次，容器中的每个 Key 所对应的对象可以是不同类型的——这就简化了代码编写。

#### 层的正向反向传播并不进行实际计算
考虑如下的代码：
```cpp
auto i1 = Matrix<float, DeviceTags::CPU>(2, 3);
auto i2 = Matrix<float, DeviceTags::CPU>(2, 3);

auto input = AddLayerInput::Create().Set<AddLayerIn1>(i1)
                                    .Set<AddLayerIn2>(i2);
auto out = layer.FeedForward(input).Get<LayerIO>();
```
此时 ```out``` 的类型为：
```cpp
BinaryOp<BinaryOpTags::Add,
         Matrix<float, DeviceTags::CPU>,
         Matrix<float, DeviceTags::CPU>>
```
它是一个模板表达式，而非实际的计算结果。正是由于我们将层的正向反向传播的接口设置成模板的，同时使用容器作为输入输出对象，从而实现了这个效果。对于很多深度学习框架来说，引入层就意味着将计算限制在了层的内部（层的输入与输出是计算好的结果），这就对系统的整体性能优化产生了不利的影响。但 MetaNN 使用了模板与编译期计算，有效地解决了这个问题！在 MetaNN 中，层只是用于代码与模板表达式的组织，最终的计算会留到最后一并进行。

### 层的其它接口
MetaNN 中的每个层都需要提供正向、反向传播用的接口。除此之外，它们还可能提供如下的接口：
 * ```Init```: 用于初始化层中所包含的参数矩阵。
 * ```LoadWeights```: 用于从外部读取参数矩阵。
 * ```SaveWeights```: 用于将参数矩阵保存到外部。
 * ```GradCollect```: 用于获取梯度信息。
 * ```NeutralInvariant```: 用于断言层处于中性状态。

#### 关于```NeutralInvariant```
这里解释一下最后一个接口。 MetaNN 的层负责保存正向、反向传播的一些中间结果，以供梯度计算等操作使用。在层对象被使用之前，其中不应当包含任何中间结果。我们将这种状态称为“中性”状态。层在进行完一次正向传播，一次反向传播，获取相应的梯度信息后，也不应该再拥有任何中间结果了。也即，此时层也应当处于中性状态。我们可以在一次正向传播，一次反向传播之后调用 ```NeutralInvariant``` 对此进行断言，从而确保程序的正确性。

#### 通用接口
除了正向、反向传播之外，其余的接口都是可选的：并非所有的层都会包含全部的接口。比如，有的层并没有内部参数，因此也就不需要 ```Init``` 等接口了。因此给定一个层对象 ```layer``` ，调用 ```layer.Init``` 可能会出错。为了解决这个问题， MetaNN 引入了如下的通用接口函数（最后两个函数是为了保持一致性而引入的）：
 * ```LayerInit```: 对 ```Init``` 的封装
 * ```LayerLoadWeights```: 对 ```LoadWeights``` 的封装
 * ```LayerSaveWeights```: 对 ```SaveWeights``` 的封装
 * ```LayerGradCollect```: 对 ```GradCollect``` 的封装
 * ```LayerNeutralInvariant```: 对 ```NeutralInvariant``` 的封装
 * ```LayerFeedForward```: 对 ```FeedForward``` 的封装
 * ```LayerFeedBackward```: 对 ```FeedBackward``` 的封装

比如，可以调用 ```LayerInit(layer, ...)``` 来实现初始化。 MetaNN 足够智能，如果 ```layer``` 对象没有包含 ```Init``` 接口，那么 ```LayerInit``` 什么也不干。否则就会调用 ```layer``` 对象的 ```Init``` 接口以实现初始化。

## 复合层
复合层实现了典型的组合模式。它对基本层或者其它复合层进行封装，使用相对基本的元素构建复杂的网络结构。

### 设计理念
与基本层相比，复合层由于其天然的特性，引入了两个新的概念：父子关系与拓扑结构。

复合层是基本层的组合，很自然地，复合层与其所包含的基本层就形成了一种父子关系。对于基本层，我们可以通过 Policy 来调整其行为细节。与之类似，我们也希望使用 Policy 来调整复合层的行为。但由于这种父子关系的存在，因此我们需要引入一种机制，可以方便地指定每个子层的行为细节。

通常来说，复合层中的基本层组成了一个有向无环图的结构。复合层需要引入一种机制来描述这种结构，在给定了结构的基础上，实现自动的正向与反向传播。

MetaNN 中实现了几个简单的复合层：主要是出于展示目的，可以通过其了解到复合层的实现方法。接下来，我们将以 ```LinearLayer``` 为例描述复合层的构造。```LinearLayer``` 接收一个输入向量，首先使用 ```WeightLayer``` 将其与某个矩阵点乘，之后使用 ```BiasLayer``` 将点乘结果与另一个向量相加并返回。

### 描述拓扑结构
建立复合层的第一步是要描述该复合层所包含的子层，以及这些子层的连接关系——即复合层的拓扑结构。首先，我们要为复合层中的每个子层给出唯一的名字。注意这个名字并不是层的类型名——一个复合层中可能包含多个具有相同类型的子层，它们也需要进行区分：
```cpp
template <>
struct Sublayerof<LinearLayer> {
    struct Weight;
    struct Bias;
};
```
```SubLayerOf``` 是 MetaNN 中的一个模板， MetaNN 使用它来描述复合层的子层。上述代码表明： ```LinearLayer``` 包含了两个子层，分别命名为 ```Weight``` 与 ```Bias``` 。

在此基础上，我们可以定义复合层的拓扑结构了：
```cpp
using WeightSublayer = Sublayerof<LinearLayer>::Weight;
using BiasSublayer = Sublayerof<LinearLayer>::Bias;

using Topology =
    ComposeTopology<SubLayer<WeightSublayer, WeightLayer>,
                    SubLayer<BiasSublayer, BiasLayer>,
                    InConnect<LayerIO, WeightSublayer, LayerIO>,
                    InternalConnect<WeightSublayer, LayerIO,
                                    BiasSublayer, LayerIO>,
                    OutConnect<BiasSublayer, LayerIO, LayerIO>>;
```
```ComposeTopology``` 是 MetaNN 中的一种构造。用于描述复合层的拓扑结构。它是一个模板，可以接收四种类型的模板参数：
 * ```SubLayer```: 将子层的名称与层的类型关联起来。
 * ```InConnect```: 将复合层的输入关联到某个子层的输入上。
 * ```OutConnect```: 将某个子层的输出关联到复合层的输出上。
 * ```InternalConnect```: 将某个子层的输出关联到另一个子层的输入上。

因此，上面 ```Topology``` 中语句的含意依次是：
 * ```LinearLayer``` 的子层 ```Sublayerof<LinearLayer>::Weight``` 是一个 ```WeightLayer```
 * ```LinearLayer``` 的子层 ```Sublayerof<LinearLayer>::Bias``` 是一个 ```BiasLayer```
 * ```LinearLayer``` 的输入容器中有一个 Key 为 ```LayerIO``` 的元素，这个元素应当与 ```Weight``` 子层输入容器的 ```LayerIO``` 相关联
 * ```Weight``` 子层的输出容器中有一个 Key 为 ```LayerIO``` 的元素，将这个元素关联到 ```Bias``` 子层输入容器的 ```LayerIO``` 上
 * ```Bias``` 子层的输出容器中 Key 为 ```LayerIO``` 的元素应当关联到 ```LinearLayer``` 输出容器中 Key 为 ```LayerIO``` 的元素中

结合 ```LinearLayer``` 所实现的功能，这个描述就很容易理解了。需要说明的是，描述中语句的顺序是任意的。比如：下面的描述与上面的描述等价：
```cpp
using Topology2 =
    ComposeTopology<OutConnect<BiasSublayer, LayerIO, LayerIO>,
                    SubLayer<WeightSublayer, WeightLayer>,
                    InConnect<LayerIO, WeightSublayer, LayerIO>,
                    InternalConnect<WeightSublayer, LayerIO,
                                    BiasSublayer, LayerIO>,
                    SubLayer<BiasSublayer, BiasLayer>>;
```
### 定义复合层
在有了拓扑结构后，复合层的定义就非常简单了。以下是 ```LinearLayer``` 的完整定义：
```cpp
template <typename TPolicies>
using Base = ComposeKernel<LayerIO, LayerIO, TPolicies, Topology>;

template <typename TPolicies>
class LinearLayer : public Base<TPolicies>
{
    using TBase = Base<TPolicies>;
    using TupleProcessor = typename TBase::TupleProcessor;

public:
    LinearLayer(const std::string& p_name,
                size_t p_inputLen, size_t p_outputLen)
        : TBase(TupleProcessor::Create()
                    .template Set<NSLinearLayer::WeightSublayer>(p_name + "-weight", p_inputLen, p_outputLen)
                    .template Set<NSLinearLayer::BiasSublayer>(p_name + "-bias", p_outputLen))
    { }
};
```
```ComposeKernel``` 是 MetaNN 中的一个模板。它实现了几乎所有复合层所需要的功能。这个模板接收4个参数，分别表示了复合层的输入容器，输出容器，Policy 容器以及拓扑结构。之所以还要在这里进行派生，是因为 ```ComposeKernel``` 中没有对构造函数进行定义（我们需要在复合层的构造函数中调用其子层的构造函数）。这是 ```LinearLayer``` 定义中唯一需要完成的工作。

复合层的定义并不麻烦，这是因为 ```ComposeKernel``` 完成了绝大多数的工作，其中就包括一个编译期实现的拓扑排序的逻辑。正是这个逻辑使得我们可以在仅仅定义了这些信息的基础上实现自动的正向、反向传播。

### 复合层的使用
复合层的使用与基本层非常相似，唯一的区别是 Policy 的指定。在 MetaNN 中，子层的 Policy 会“继承”自复合层的 Policy ，同时，子层也可以指定其自己的 Policy ，如果复合层与其子层的 Policy 出现冲突，那么以子层的为准：
```cpp
// use default policies
using RootLayer1 = MakeLayer<LinearLayer>;

// Both weight and bias sub-layers are updated
using RootLayer2 = MakeLayer<LinearLayer, PUpdate>;

// Bias layer will be updated, while Weight layer will not update
using RootLayer3 =
    MakeLayer<LinearLayer, PUpdate,
              SubPolicyContainer<Sublayerof<LinearLayer>::Weight,
                                 PNoUpdate>>;

// Similar to RootLayer3
using RootLayer4 =
    MakeLayer<LinearLayer,
              SubPolicyContainer<Sublayerof<LinearLayer>::Bias,
                                 PUpdate>>;
```
```SubPolicyContainer``` 是 MetaNN 专门引入的容器，用于描述子层与其父亲之间 Policy 的差异。

## 求值
MetaNN 的一个主体思想就是将求值的过程后移。层的正向、反向传播的结果被表示为表达式模板，并不进行实际的求值。通常来说，在进行完正向/反向传播后，我们会得到若干表达式模板所对应的对象，此时，我们可以进行统一的求值。我们将在本节看到：采用这种方式，使得我们可以在多个维度上对求值过程进行优化。

### 最简单的求值语句
最简单的求值语句调用方式如下：
```cpp
auto rm1 = Matrix<int, DeviceTags::CPU>(4, 5);
auto rm2 = Matrix<int, DeviceTags::CPU>(4, 5);
auto add = rm1 + rm2;

auto add_r = Evaluate(add);
```
```Evaluate``` 会接收一个表达式模板，返回该表达式模板所对应的值。

### ```Evaluate``` 的实现细节
```Evaluate``` 的内部定义如下：
```cpp
template <typename TData>
auto Evaluate(const TData& data)
{
    auto evalHandle = data.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    return evalHandle.Data();
}
```
```Evaluate``` 内部实际上是对框架一般性求值过程的一个封装。对于任何一个表达式模板（甚至大部分 MetaNN 中的数据类型），我们都可以调用 ```EvalRegister``` 函数来进行求值的注册，注册将返回一个句柄。之后，我们需要调用 EvalPlan::Eval() 对注册的函数进行求值。在求值之后，我们可以通过返回句柄的 ```Data``` 方法来获取计算结果。

相比 ```Evaluate``` 来说，这种一般性的求值过程可能更有用。考虑如下的代码：
```cpp
auto rm1 = Matrix<int, DeviceTags::CPU>(4, 5);
auto rm2 = Matrix<int, DeviceTags::CPU>(4, 5);
auto rm3 = Matrix<int, DeviceTags::CPU>(4, 5);
auto add1 = rm1 + rm2;
auto add2 = rm1 + rm3;

auto handle1 = add1.EvalRegister();
auto handle2 = add2.EvalRegister();
EvalPlan<DeviceTags::CPU>::Eval();
auto res1 = handle1.Data();
auto res2 = handle2.Data();
```
在调用 EvalPlan::Eval 之前，程序注册了两个求值过程，那么在 EvalPlan::Eval 具体求值时，系统就可以进行一定程度上的并行化，从而提升其性能。

而这就涉及到了求值的优化问题了。接下来，我们将讨论几种可能的优化(需要说明的是，现有的 MetaNN 并没有实现后续讨论的大部分优化技术。但 MetaNN 已经把框架搭好了，在此基础上实现这些技术是相对平凡的工作)：

### 求值优化之避免重复计算
进行注册的多个表达式模板之间可能存在某些关系。比如它们的一些子树的相同的。 MetaNN 引入了自动分析机制，可以确保在这种情况下，相同的子树只会被求值一次。同时如果某个子树之前已经完成了求值，则不会被再次求值。

### 求值优化之同类操作的合并
为了达到较高的计算速度，框架的底层需要调用一些高效计算的库。比如 Intel 的 MKL ，或者 CUDA 等等。这些库提供了一些函数，可以将多个计算合并进行。比如同时计算多组矩阵点乘。使用求值注册的方法，我们可以将多个此类任务合并到一起，在调用 EvalPlan::Eval 时一并完成，从而提升计算速度。

### 求值优化之不同操作的合并
大部分情况下，对表达式模板的求值可以看成树的广度优先遍历：要对某个结点求值，就需要先对其子结点求值。但在一些情况下，我们可以简化这个求值的过程。比如，如果当前结点是求对数，而其子结点是求指数，那么我们可以将求对数与求指数的操作抵消掉。我们还可以引入更复杂的机制，在一定程度上简化求值的过程。

事实上，这是一种非常典型的求值优化手段。 MetaNN 中为此专门引入了 ```OperSeq``` 模板以实现类似的功能。 ```OperSeq``` 实现了典型的职责链模式，可以处理表达式模板中“特殊”的情况。以 ```RowSoftmaxDerivative``` 为例，MetaNN 定义了：
```cpp
template <>
struct OperBuildInSeq_<BinaryOpTags::RowSoftmaxDerivative>
{
    using type = OperSeqContainer<CaseNLL::Calculator,
                                  CaseGeneral::Calculator>;
};
```
它就表示了：首先看看当前结构是否满足 ```CaseNLL``` ，如果满足，就按照 ```CaseNLL``` 所定义方式计算。否则再尝试一般方式。

## 总结
以上就是 MetaNN 的一个简单的介绍了。与其它的深度学习框架相比， MetaNN 更深层次地使用了模板元编程的技术。正如文章开头所述，它目前只是一个骨架，距实际的应用还有一定的距离。同时使用这套框架也需要一定的模板元编程基础，这可能对很多 C\+\+ 开发者来说都是一块不算熟悉的领域。但它也有它的优势。

框架采用了模板元编程的技术，将计算优化与模型结构彻底分开。这能带来很多好处。比如，我们可能并不需要框架对 mini-batch 的显式支持了。大部分情况下（Batch Normalization 除外），引入 Batch 是为了快速计算，但 Batch 的引入会导致系统变得更加复杂。比如在神经网络机器翻译中，为了使用 Batch ，通常来说我们需要对训练语料按照源语言长度进行排序，引入填充标签等，以保证 Batch 可以更高效地进行。我们甚至要处理高维数据：很多时候我们需要使用矩阵来表示中间结果，而要使用三维或者更高维的结构来表示 Batch 的中间结果——这大大增加了程序的复杂度，影响其理解。但在 MetaNN 中，大部分情况下，我们需要做的只是引入循环，在每个循环中只处理一个条目。循环结束后统一调用 ```EvalPlan::Eval``` 即可，```EvalPlan::Eval``` 会帮我们完成之前需要 Batch 才能进行的计算合并。程序的逻辑会更加清楚。

另一个明显的好处是用户在使用上会更加省心，不用考虑一些小技巧。一个非常典型的例子是：很多深度学习框架都包含了 ```Softmax``` 与 ```SoftmaxLoss``` 层或者类似的构造，这是因为后者需要解决 ```Softmax``` 可能会引起的数值稳定性问题。使用 MetaNN ，这个问题被掩盖在了求值优化之中。用户不需要再关注这些层行为的细节差异了。

除了深度学习框架之外， MetaNN 对于 C\+\+ 模板元编程也有其贡献。Policy 的概念并非 MetaNN 原创，但它对 Policy 的实现进行了扩展，使得在一个模板中 Policy 的个数不再有限制（事实上，Policy 的个数还会取决于编译器对变长模板的支持程度）。MetaNN 还实现了一个编译期容器，对数据分类、Policy 组织方面都有深入的探讨。它所实现的编译期拓扑排序也将元编程的技术从“小技巧”扩展成切实可用的算法层面。可以说，如果能将这套代码读懂，那么对 C\+\+ 元编程就会有相对深入的理解了。

最后要说的一点是：本套代码没有注释。这个文档是 MetaNN 的一个简介，但并没有深入到其中的细节之中，想通过这篇文档来理解程序的实现还是比较困难的。 MetaNN 包含了很多细节，之所以没有注释，是因为如果读者对 C\+\+ 模板元编程有所了解，那么我认为阅读这个代码中的逻辑对于这些读者来说是比较直白的。反过来，如果读者不了解 C\+\+ 模板元编程，或者了解较少，那么加了注释并不能起到多大作用。要完全描述出这个框架中的实现细节，需要一本书的篇幅——我已经完成了这么一本书的写作，该书讨论了 book 分支中的代码，将很快面世。
