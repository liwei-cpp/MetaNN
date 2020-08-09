[中文版](readme.md)
# MetaNN
MetaNN is a deep learning framework skeleton, which aims to explore the application of C++ template metaprogramming in deep learning. It does not support multi machine concurrent training. At present, it only supports CPU and contains some simple constructs. However, MetaNN has enough scalability to easily support devices such as GPU or FPGA. At present, the computational logic contained in it is mainly illustrative, and its speed is not fast, but its framework is very flexible. Program optimization based on this framework can greatly improve the performance of deep learning system while ensuring the ease of use.

This document will briefly describe the main design ideas, core components and optimization possibilities of MetaNN.

## Code Organization and Running Environment
MetaNN is a C++ template library. Like other template libraries, its core logic is contained in the header file (ending with .h). In addition to these header files, MetaNN also contains a number of test projects: Tests/DataOpTest, Tests/LayerTest and Tests/OtherTests. They are used to test the internal logic of MetaNN. At the same time, they are simple examples to show how the data and operations, layers, and other parts of MetaNN are used. They call the logic contained in the MetaNN to implement specific calculations. Readers can read the code and understand how MetaNN works from the perspective of users.

### Running Environment
Both the MetaNN library and the test projects are developed with CodeLite. The test program can be compiled and run in the following environment:
* g++ 7.4.0
* g++ 8.3.0
* g++ 9.1.0
* clang++ 8.0.0
* clang++ 9.0.0

It is not recommended to compile this project with lower version of compilers. MetaNN uses features from C++ 17. I can't guarantee that every line of code in the program fully conforms to the C++ 17 standard; furthermore, the undefined parts of the standard may cause different compiler behaviors; bugs in the compiler may also prevent the program from compiling smoothly. Readers can choose their own compiler to test. The recommended running environment is Ubuntu & G++.

It should be noted that since it is a troublesome job to test the behavior of each compiler, I will not test it after every modification. It will be tested uniformly after accumulating several modifications. For future testing cases, I'll mark a question mark for compilers that are not tested.

MetaNN is self-contained. Except to the dependence on C++ 17, the author intentionally does not rely on any third-party library. This not only facilitates the construction of the running environment, but also ensures that readers do not need the background knowledge of specific libraries when understaning the program. (of course, to understand this library, you need to have a certain understanding of C++ 17 and metaprogramming).

### Code Organization
The whole framework core logic is contained in the MetaNN directory. At the same level as the MetaNN directory is the Tests directory, which contains the corresponding test program. Here, we only take the MetaNN directory as an example to illustrate the organization of the code.

 * **_root**: Only one file meta_nn.h is included. Reference this file to use all the functions provided by MetaNN.
 * **data**: The directory where the data is stored, it contains the various data types that the deep learning framework may use.
 * **data_copy**: The logic used to replicate the data is preserved at here. At present, there is only one function in this directory, which is used to assign one CPU tensor to another. In the future, more data replication methods might be introduced as needed.
 * **evaluate**: The evaluation logic in MetaNN is saved at here.
 * **facilities**: It contains some auxiliary (meta) functions and structures.
 * **layers**: It contains the implemtation logic of "Layer", which is the core of the deep learning framework.
 * **model**: It stores some model related logic, such as parameter initialization, gradient collection, etc.
 * **operation**: It contains the implementation logic of the operation.
 * **policies**: It contains the implementation of policy. Policy can be regarded as a branch mechanism at compile time, which is mainly used for policy customization of layers in MetaNN. We will explain the role of policy when we discuss the design of layers later.
 
Next, the text discusses several components in the framework from the bottom up. The design concept is described and some examples are given.

## Data Types
### Design Concepts
* **Basic Data Representation**： Previous versions of MetaNN did not support Tensor, but used class templates such as Matrix and Scalar to represent data. The purpose of this is to introduce a relatively clear concept of the data structure. However, with the development, the author finds that the scalability is limited by using the original architecture. Therefore, the new version abandons the original design and adopts the expression of Tensor used in popular deep learning frameworks.

* **Support Data Expansion and Classification**： The data type of MetaNN supports the extension of several dimensions. In addition to the introduction of new data types mentioned above, it is also possible to extend the element types (such as ```float```, ```double```, or other fixed-point types) contained in existing data types (actually data templates), and extend the computing devices supported by data types (such as ```CPU```, or ```GPU```). The data types in MetaNN are classified according to their categories. We use ```CategoryTags::Tensor<Dim>``` to represent the data category. For example, ```CategoryTags::Tensor<2>``` denotes matrices; ```CategoryTags::Tensor<0>``` denotes scalars.

### Basic Usage
The most basic type of data represented in MetaNN is Tensor. For example, you can declare a Tensor in the following way:
```cpp
Tensor<int, DeviceTags::CPU, 2> matrix(10, 20);
```
This is equivalent to defining a matrix with 10 rows and 20 columns in each row. The data element type of the object is ```int```, and the computing device supported is ```CPU```.

The system introduces several aliases for ```Tensor```, such as:
```cpp
template <typename TElem, typename TDevice>
using Matrix = Tensor<TElem, TDevice, 2>;
```
Therefore, the following definition is equivalent to the above definition:
```cpp
Matrix<int, DeviceTags::CPU> matrix(10, 20);
```

MetaNN can also define other types of Tensor objects:
```cpp
// a vector, containing 100 elements, the 37th element is 0.3, and the rests are 0
BiasVector(100, 37, Scalar<int, DeviceTags::CPU>{0.3});

// A 10 * 20 matrix, where the element values are all 100
TrivalTensor(Scalar<int, DeviceTags::CPU>{100}, 10, 20);

// A 10 * 20 matrix, where the element values are all 0
ZeroTensor<int, DeviceTags::CPU, 2> m3(10, 20)
```
You can also declare tensors with higher dimensions:
```cpp
// Tensor object with dimension of 5 * 10 * 20 and elements of 0
ZeroTensor<CheckElement, CheckDevice, 3>(5, 10, 20);
```

### Category Classification
The above statements are of different types and can be applied to different scenarios. Each type stores its internal data in different ways and provides different access interfaces. MetaNN divides these types into different categories, and the corresponding categories of specific data types can be obtained in the following ways:
```cpp
IsMatrix<Matrix<int, DeviceTags::CPU>>;                    // true
IsMatrix<ZeroTensor<int, DeviceTags::CPU, 3>>;        // false
IsThreeDArray<ZeroTensor<int, DeviceTags::CPU, 3>>;	  // true
IsMatrix<ZeroTensor<int, DeviceTags::CPU, 3>>;	      // false
```
Or use the following methods:
```cpp
// CategoryTags::Tensor<2> (也即 CategoryTags::Matrix)
DataCategory<Matrix<int, DeviceTags::CPU>>;

// CategoryTags::Tensor<3>
DataCategory<ZeroTensor<int, DeviceTags::CPU, 3>>;
```

In addition to their own unique interfaces, data types with the same category provide a common set of interfaces. For example, all the data types provid Shape method. However, the types of returned objects of ```Shape()``` vary according to the data types: the objects of matrix category need to provide interfaces to return the number of rows and columns, while the ```Shape()``` corresponding to the three-dimensional tensor will return values of three dimensions. In addition, each data type has to provide an evaluation interface to convert to an object of the corresponding principal type (we'll discuss the evaluation process later).

### Introducing New Types
We can simply introduce a new type for MetaNN:
```cpp
template <typename TElem, typename TDevice>
class MyMatrix
{
public:
    using CategoryTag = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = TDevice;
}

MyMatrix<float, DeviceTags::CPU> ...
```
Note that the ```CategoryTag``` defined above will set the category of ```MyMatrix``` as a matrix. In this way, the following calls will return ```true```:
```cpp
IsMatrix<MyMatrix<int, DeviceTags::CPU>>;
```

Custom types have the same status as those already defined by MetaNN.

### Dynamic Type
In addition to the above types, MetaNN also introduced a special type template: ```DynamicData<>```. It is used to encapsulate different data types, conceal the underlying specific type information, and only expose the core features and interfaces of the data type. For example:
```cpp
DynamicData<float, DeviceTags::CPU, CategoryTags::Matrix>
```
It can be regarded as a container in which a matrix with element type ```float``` and device type ```CPU``` can be placed:
```cpp
vector<DynamicData<float, DeviceTags::CPU, CategoryTags::Matrix>> vec;

Matrix<float, DeviceTags::CPU> m1(100, 37);
vec.push_back(MakeDynamic(m1));

ZeroData<CategoryTags::Matrix, float, DeviceTags::CPU> m2(10, 20)
vec.push_back(MakeDynamic(m2));
```
```MakeDynamic``` is a function provided by MetaNN. Given a specific data type, it can be converted to the corresponding ```Dynamic``` version.

Similarly, we can also put a three-dimensional tensor in the container of ```DynamicData<... CategoryTags::Tensor<3>>```.

```DynamicData``` is used to save intermediate variables, which conceal the specific data types contained in the bottom layer. It is convenient to store data of different types (unknown type) but the same category (known category). However, the introduction of ```DynamicData``` leads to the loss of specific information types, which will have some adverse effects on compile-time calculations. Therefore, ```DynamicData``` will only be used when necessary.

### Equality Testing
Each data type of MetaNN provides ```operator ==``` and ```operator !=``` to compare with other data. But the comparison here is only for evaluation and optimization. We don't want to provide an element-level comparison, that is, to compare every element in a tensor: on the one hand, floating-point data cannot be accurately compared; on the other hand, such comparisons are time-consuming. What MetaNN provides is a quick comparison scheme:
 * If A is copied from B, then A and B are equal
 * If A is copied from B, and C and D are the results of the evaluation of A and B, then C and D are equal

In addition to the above items, the data is usually not equal (unless the operation cost of the underlying data is very low, MetaNN will consider using the underlying data to determine whether it is equal).

## Operation
MetaNN contains multiple operations. Each operation can receive one or more data objects and return one operation result.

### Design Concept
MetaNN uses expression templates as the result of operations. The expression template object contains the operand objects and represents the operation, and it does not perform actual calculations. Therefore, the complexity of constructing the operation result template based on the operation is very low. The actual evaluation process of the expression template is postponed until the evaluation is explicitly called, this provides room for optimization of the evaluation process.

On the one hand, the expression template type can be regarded as the result of an operation; on the other hand, it can also be regarded as a special kind of data. Similar to the basic data types in MetaNN, the expression template objects generated by the operation will also be divided into categories such as matrix and scalar, and provide the interfaces required for the corresponding category. In this way, we can use the result of one operation as the operand of another operation.

MetaNN is a rich type, not any type can be used as an operand of an operation. MetaNN has designed mechanisms to prevent misuse of operands (of course, we can make certain operations support new operands or combinations of operands in an extended manner).

### Basic usage
You can call the operations in MetaNN and generate the corresponding expression template in the following ways:
```cpp
auto rm1 = Matrix<int, DeviceTags::CPU>(4, 5);
auto rm2 = Matrix<int, DeviceTags::CPU>(4, 5);
auto add = rm1 + rm2;
```
```add``` is an expression template formed by two operands. The actual type is:
```cpp
Operation<OpTags::Add, 
          OperandContainer<Matrix<int, DeviceTags::CPU>, Matrix<int, DeviceTags::CPU>>
          PolicyContainer<>
         >
```
Generally speaking, we don't need to care about the actual type of ```add``` in the actual programming process (just use ```auto``` as in the above example to let the compiler automatically deduct it, or use ```DynamicData``` to encapsulate it). But here, it is still necessary to understand the information contained in the type corresponding to the ```add``` object.

The actual type of the ```add``` object is the result of the instantiation of the ```Operation``` template. This means that it is an "operation". The specific operation type is represented by ```OpTags::Add```. The second template parameter of ```Operation``` indicates types of operands. The third template parameter of ```Operation``` is used to save compile-time parameters.

The template expression itself can also be regarded as a kind of data, therefore, we can write:
```cpp
auto rm1 = Matrix<int, DeviceTags::CPU>(4, 5);
auto rm2 = Matrix<int, DeviceTags::CPU>(4, 5);
auto add = rm1 + rm2;
auto rm3 = ZeroTensor<CategoryTags::Matrix, DeviceTags::CPU, 2>(4, 5);
auto sub = add - rm3;
```
At this time, the type corresponding to ```sub``` is:
```cpp
Operation<OpTags::Substract,
          OperandContainer<
                           Operation<OpTags::Add,
                                     OperandContainer<
                                                      Matrix<int, DeviceTags::CPU>,
                                                      Matrix<int, DeviceTags::CPU>>,
                                     PolicyContainer<>>
                           ZeroData<CategoryTags::Matrix, DeviceTags::CPU>,
          PolicyContainer<>>
```

In essence, template expressions form a tree structure. The evaluation process of template expressions can also be regarded as a tree traversal process. This process can be optimized. We will discuss related optimization methods when we discuss evaluation later.

### Classification of Template Expressions
Template expressions can also be regarded as data. MetaNN automatically implements the classification of template expressions. such as:
```cpp
IsMatrix<Operation<OpTags::Substract,
          OperandContainer<
                           Operation<OpTags::Add,
                                     OperandContainer<
                                                      Matrix<int, DeviceTags::CPU>,
                                                      Matrix<int, DeviceTags::CPU>>,
                                     PolicyContainer<>>
                           ZeroData<CategoryTags::Matrix, DeviceTags::CPU>,
          PolicyContainer<>>>;    // true
```
This is because the result of adding two matrices is still a matrix, and the result of subtracting a matrix from a matrix is still a matrix.

### Various Operations
It should be noted that not all operations are as "clear at a glance" as addition and subtraction, whose operands are consistent with the category of the operation results. For example, MetaNN defines the ```ReduceSum``` operation, which can sum the data along with certain dimensions in a tensor to obtain a new tensor. At this time, the input and output category of operations are different. MetaNN has various operations and can be expanded relatively easily. You can specify the category of operands and operation results, as well as some other information while expanding.

### Operand Support
MetaNN is rich in types, and a program using MetaNN may use multiple types in one operation. For a particular operation, not all operand types are legal. MetaNN introduces a specific meta-function to describe the legal operands that an operation can receive. Take addition as an example, the corresponding meta function is:
```cpp
template <typename TOp1, typename TOp2>
constexpr bool IsValidOper<OpTags::Add, TOp1, TOp2>
    = IsValidCategoryTag<DataCategory<TOp1>> && IsValidCategoryTag<DataCategory<TOp2>>;
    
namespace OperAddWithNum
{
template <typename TOp1, typename TOp2>
constexpr bool Valid()
{
    if constexpr (IsValidCategoryTag<DataCategory<TOp1>> && !IsValidCategoryTag<DataCategory<TOp2>>)
    {
        return std::is_constructible_v<typename RemConstRef<TOp1>::ElementType, TOp2>;
    }
    else if constexpr (!IsValidCategoryTag<DataCategory<TOp1>> && IsValidCategoryTag<DataCategory<TOp2>>)
    {
        return std::is_constructible_v<typename RemConstRef<TOp2>::ElementType, TOp1>;
    }
    else
    {
        return false;
    }
}

template <typename TOp1, typename TOp2>
constexpr bool IsValidOper<OpTags::AddWithNum, TOp1, TOp2> = OperAddWithNum::Valid<TOp1, TOp2>();

template <typename TP1, typename TP2,
          std::enable_if_t<IsValidOper<OpTags::Add, TP1, TP2> ||
                           IsValidOper<OpTags::AddWithNum, TP1, TP2>>* = nullptr>
auto operator+ (TP1&& p_m1, TP2&& p_m2)
```
Only when ```IsValidOper<OpTags::Add, TP1, TP2>``` or ```IsValidOper<OpTags::AddWithNum, TP1, TP2>``` is true, two operands with type ```TP1``` and ```TP2``` can be added. In fact, this means only two tensors, or one tensor and one value can be added.
```cpp
float a, b;
// This will not trigger the operation logic in MetaNN
float c = a + b;

Matrix<int, DeviceTags::CPU> rm1(2, 3);
Tensor<int, DeviceTags::CPU, 3> rm2(5, 2, 3);
// Triggering the operation logic in MetaNN
// and the result of the addition is a 5*2*3 tensor.
auto rm3 = rm1 + rm2;
```

Of course, the operand combination supported by the operation can also be extended. The corresponding extension can be introduced through the specialization ```IsValidOper```, but at the same time, a clear definition should be given for the corresponding operation.

## Basic Layer
Layer is a relatively advanced component in MetaNN. Many current deep learning frameworks have weakened the concept of layers, and instead merged the concepts of operations and layers. MetaNN believes that distinguishing layers and operations can better describe abstractions at different levels. This section discusses the basic layers.

### Design Concept
Layer provides the external interface of MetaNN. The basic layer encapsulates operations, and specific methods to realize forward and backward data propagation. Each basic layer can implement a specific function, such as adding two matrices or get the dot product of two matrixs. Although the functions of basic layers are relatively simple, the details of their behaviors can be configured. For example, we can specify whether a layer should update the parameters contained in it during the training process. The layer should understand its specific behavior details and optimize accordingly.

Under normal circumstances, the specific behavior of a layer rarely changes during the execution of the program. Therefore, you can specify the corresponding information at compile time to ensure that the information is used as early as possible for maximum optimization.

Each layer should support forward and backward propagation. In addition, the layer may need to provide several additional interface functions according to its own characteristics. For example, if the layer contains a parameter matrix, then it needs to provide an interface to initialize the matrix it contains, read or load the content of the matrix, etc.

Most of these interfaces are provided in the form of template methods. This ensures that different types of parameters can be used to call the methods provided by the layer, which is particularly important for forward and backward propagation, and together with expression templates constitute the premise of performance optimization. We will see this later.

The layer has many unique features in design and use. It can be said that it is a very characteristic part of the MetaNN framework. The introduction of many sub-modules in MetaNN are related to layers. This section will discuss the basic layer and its related sub-modules. The next section will discuss some features of the combined layer on this basis.

### Layer Declaration
Generally speaking, the layer will be declared as a class template, such as:
```cpp
template <typename TInputs, typename TPolicies>
class AddLayer;
```
It contains two template parameters, the former is a input type map, and the latter is Policy. Next, we will discuss these two parts one by one.

### Input/output port collection and input type map
Each layer will specify the number and name of data it can receive, and the number and name of output results. For example, for the additive layer:
```cpp
template <typename TInputs, typename TPolicies>
class AddLayer
{
public:
    using InputPortSet = LayerPortSet<struct LeftOperand, struct RightOperand>;
    using OutputPortSet = LayerPortSet<struct LayerOutput>;
    // ...
};
```
This means that the input of the additive layer receives two parameters, named ```LeftOperand``` and ```RightOperand```; it will produce an output result, named ```LayerOutput```.

```InputPortSet``` and ```OutputPortSet``` are called the input/output port set of the layer. The names in the port set will be used in forward and backward propagation.

The input/output port set only specifies the number and name of the input and output parameters of the layer, and does not limit the specific input type of the layer. This is sufficient for the layer that is only used for inference. For the layer used for training, the specific type of each parameter needs to be given. This information is specified by the input type map (the first parameter of the layer template).

The input type map of the layer can be specified in the following way:
```cpp
using CommonInputMap = LayerInMap<LayerKV<LeftOperand, Matrix<CheckElement, CheckDevice>>,
                                  LayerKV<RightOperand, Matrix<CheckElement, CheckDevice>>
                                 >;
AddLayer<CommonInputMap, ...> layer;
```
Among them, ```CommonInputMap``` is an input type map, which indicates that the types of ```LeftOperand``` and ```RightOperand``` are both ```Matrix<CheckElement, CheckDevice>```.

For the layer only used for inference, the input map can be empty (use ```NullParameter``` to indicate). Therefore, the following code is equivalent to constructing a layer for inference only:
```cpp
AddLayer<NullParameter, ...> layer;
```

MetaNN also provides auxiliary functions ```MakeTrainLayer``` and ```MakeInferLayer``` to construct the layer for training and the layer for inference.

### Policy and Layer Declaration
#### Brief Introduction
We hope to be able to control the specific behavior of the layer, and further, hope to specify the specific behavior of the layer during the compile time, so as to obtain information as early as possible to improve the system optimization opportunity. MetaNN uses the Policy mechanism to achieve this.

Policy is a compile-time construct that specifies the specific behavior of a function or method when it is used. A typical Policy is: use multiplication instead of addition for accumulation, the accumulation function can change its default behavior based on the information in it.

The Policy used in MetaNN is systematically organized. Each Policy contains a major category, a minor category, and related default values. The major category is used for the classification of Policy, and the minor category is used for mutual exclusion of Policy objects (we will discuss Policy objects soon):
```cpp
struct GradPolicy
{
    using MajorClass = GradPolicy;

    struct IsUpdateValueCate;
    struct IsFeedbackOutputValueCate;

    static constexpr bool IsUpdate = false;
    static constexpr bool IsFeedbackOutput = false;
};
```
This code defines two Policies: ```IsUpdate``` and ```IsFeedbackOutput```. Their major category is ```GradPolicy```. This means that the two policies are related to backpropagation. The minor categories they belong to are ```IsUpdateValueCate``` and ```IsFeedbackOutputValueCate``` (the minor category uses the name of Policy as a prefix, which is specially designed; and the ```ValueCate``` in the minor category name means that the policy corresponding to the category is a numeric policy). Their default values are both ```false```.

MetaNN not only supports numeric Policy, but also supports type Policy (since Policy is a compile-time construction, it is not surprising that it supports types):
```cpp
struct ParamPolicy
{
    using MajorClass = ParamPolicy;

    struct ParamTypeTypeCate;
    using ParamType = NullParameter;

    struct InitializerTypeCate;
    using Initializer = NullParameter;

    struct ElementTypeTypeCate;
    using  ElementType = float;

    struct DeviceTypeTypeCate;
    using  DeviceType = DeviceTags::CPU;
};
```
```ElementType``` is a type Policy, its minor category is ```ElementTypeTypeCate```, and the default "value" is ```float```.

In fact, MetaNN's Policy can take other forms. For example, we can even define a Policy whose value is a class template. However, numeric, and type policies are used most frequently in MetaNN, and MetaNN also provides macros to create these Policy objects.

#### Policy Object
The relationship between Policy and Policy objects is much like the relationship between a class and its objects. The Policy object can be regarded as the result of a Policy instantiation. The Policy object contains the major and minor category information of the corresponding Policy, but its value may not be the default value of the Policy:
```cpp
ValuePolicyObj(PUpdate,   GradPolicy, IsUpdate, true);
ValuePolicyObj(PNoUpdate, GradPolicy, IsUpdate, false);
```
The above code defines two Policy objects: ```PUpdate``` and ```PNoUpdate```. They all correspond to the aforementioned ```IsUpdate``` Policy. But one of its values is ```true``` and the other is ```false```. Similarly, you can also define a Policy object whose value is of type:
```cpp
TypePolicyObj(PNormVarScale,    VarScaleFillerPolicy, Distribute, Norm);
TypePolicyObj(PUniformVarScale, VarScaleFillerPolicy, Distribute, Uniform);
```

In addition to ```ValuePolicyObj``` and ```TypePolicyObj```, MetaNN also provides ```ValuePolicyTemplate``` and ```TypePolicyTemplate``` to construct Policy objects, such as:
```cpp
ValuePolicyTemplate(PUpdateIs, GradPolicy, IsUpdate);
```
A policy object template ```PUpdateIs``` is constructed, and we can use ```PUpdateIs<true>``` or ```PUpdateIs<false>``` as Policy objects.

Next, let's look at how to use the Policy object to specify the behavior details of the layer.

#### Use Policy Objects in Layers
Most of the layers in MetaNN are declared as templates. Receive two template parameters, one of which is a "container" that can store zero to more Policy objects:
```cpp
template <typename TInputMap, typename TPolicies>
class AddLayer;

// layer is an object with interfaces such as feed-backward and feed-forward;
AddLayer<TInputMap, PolicyContainer<PUpdate, PFeedbackOutput>> layer;
```
The above program uses the list of Policy objects as template parameters and instantiates the ```AddLayer``` template. The meta function provided by MetaNN can be used to make the definition clearer and easier to understand:
```cpp
using MyLayer = MakeInferLayer<AddLayer, PUpdate, PFeedbackOutput>;
MyLayer layer;
```

The order of Policy objects is not important. The functions of the following two layers are the same (although their C++ types are different):
```cpp
using MyLayer1 = MakeInferLayer<AddLayer, PRowVecInput, PUpdate>;
using MyLayer2 = MakeInferLayer<AddLayer, PUpdate, PRowVecInput>;
```

Not all combinations are valid. For example, the following statement is illegal:
```cpp
using MyLayer1 = MakeInferLayer<AddLayer, PNoUpdate, PUpdate>;
```
Obviously, this is because a layer cannot have two states of "update internal parameters" and "not update internal parameters" at the same time. Declaring such a layer will cause the compiler to report an error: The declaration of the same layer contains multiple Policy objects belonging to the same minor category.

For a particular layer, not all Policy objects will affect its behavior. For example, ```AddLayer``` does not contain internal parameters, so setting ```PUpdate``` will not cause the corresponding parameters to be updated. At this time, whether the corresponding Policy object is set or not will not affect the behavior details of the layer. That is, the following two declarations have the same behavior:
```cpp
// MyLayer1 = AddLayer<NullParameter, PolicyContainer<PUpdate>>
using MyLayer1 = MakeInferLayer<AddLayer, PUpdate>;

// MyLayer2 = AddLayer<NullParameter, PolicyContainer<>>
using MyLayer2 = MakeInferLayer<AddLayer>;
```

On the other hand, if a layer needs to determine its behavior according to a specific Policy, but the object corresponding to the Policy is not specified when the layer is instantiated, then the layer will use the default value corresponding to the Policy to determine details of its behavior. Therefore, the following two declarations have the same behavioral details:
```cpp
using MyLayer1 = MakeInferLayer<AddLayer, PNoFeedbackOutput>;
using MyLayer2 = MakeInferLayer<AddLayer>;
```

Finally, let's dive a little deeper into the implementation of the layer and see how the layer obtains the value corresponding to a Policy:
```cpp
template <typename TInputMap, typename TPolicies>
class AddLayer {
public:
    static constexpr bool IsFeedbackOutput
    	= PolicySelect<FeedbackPolicy, TPolicies>::IsFeedbackOutput;
```
The actual implementation process is slightly different from what is discussed here (because it involves the composite layer, it needs further processing). But the code shown here does not affect our understanding of logic. MetaNN provides the meta function ```PolicySelect``` to select Policy. It receives the list of ```PolicyContainer``` and the major and minor categories of the Policy to be queried, and returns the corresponding Policy value (of course, for type Policy, it returns the corresponding type information).

The layer will adjust its behavior internally according to the value of the Policy to minimize unnecessary calculations.

### Forward and back propagation
In addition to the constructor, each layer needs to provide at least two interfaces for forward and backward propagation. The input and output of forward and backward propagation are containers. The input container of forward propagation and the output container of back propagation have the same "structure", which is called the input container of the layer in MetaNN; the output container of forward propagation and the input container of back propagation have the same "structure", It is called the output container of the layer in MetaNN. This section will first discuss some characteristics of the container, and then discuss some characteristics of the forward and backward propagation functions.

#### Container
Take forward propagation as an example. Generally speaking, the layer receives one or more data as input, and generates one or more data as output after calculation. Some deep learning frameworks use linear tables like ```std::vector``` as input and output data structures, input data is represented by a ```vector```, and output data is represented by another ``` vector```. This method has two disadvantages: First, the parameters passed to the layer cannot be distinguished by the ```vertor``` data structure itself. For example, for a layer that calculates the subtraction of two matrices, you can specify that the first parameter passed in ```vector``` is the minuend, and the second is the subtrahend. But this kind of regulation is not reflected from the point of view of procedure, and there may be misuse.

The second problem is that containers like ```vector``` can only store elements of the same type. The input and output data types of the layer are ever-changing, and the use of a containers like ```vector``` undoubtedly limits the input and output data types.

MetaNN defines a compile-time container to solve the above problems. The first step in using the container is to declare its structure:
```cpp
struct A; struct B; struct Weight;
struct FParams : public VarTypeDict<A, B, Weight> {};
```
Here it defines a container ```VarTypeDict<A, B, Weight>```. Among them, ```A, B, Weight``` are keywords, and each keyword can store a data object correspondingly. In order to simplify the subsequent use, an alias ```FParams``` is introduced for the container using derivation. Of course, aliases can also be introduced in the following ways:
```cpp
struct A; struct B; struct Weight;
using FParams = VarTypeDict<A, B, Weight>;
```
Furthermore, the above code can be further simplified as:
```cpp
using FParams = VarTypeDict<struct A, struct B, struct Weight>;
```
An additional benefit can be gained by using derived methods. If the container only needs to contain one key, then we can make the container name the same as the key name:
```cpp
struct LayerIO : public VarTypeDict<LayerIO> {};
```

After declaring the structure of the container, we can create the container object and add data to it:
```cpp
auto data = FParams::Create().Set<A>(true)
                             .Set<B>(std::string("abc"))
                             .Set<Weight>(15);
```
The above code is not difficult to understand. It creates an instance of the ```FParams``` container and assigns a corresponding value to each key.

Here are a few points to explain. First, it is not difficult to see that the value type corresponding to each key can be different. This seems a bit like the support provided by some weakly typed languages. But MetaNN still relies on the construction of a strongly typed language like C++. Consider the following code:
```cpp
auto data1 = FParams::Create().Set<A>(true)
                              .Set<B>(std::string("abc"))
                              .Set<Weight>(15);

auto data2 = FParams::Create().Set<A>(12)
                              .Set<B>(3.5)
                              .Set<Weight>(std::vector<int>());

data1 = data2;    // error!
```
The types of ```data1``` and ```data2``` are different: the types of the corresponding values are different.

Secondly, the result type obtained after ```FParams::Create().Set...``` is not ```FParams```, but a type deduced by ```FParams```. The user does not need to pay attention to the specific type, only needs to use ```auto``` to define the variable, whose type is automatically deduced by the compiler.

Third, the setting order of the elements in the container does not need to be the same as the key order when the container is declared. We can also write:
```cpp
auto data = FParams::Create().Set<B>(std::string("abc"))
                             .Set<A>(true)
                             .Set<Weight>(15);
```

The objects in the container can be obtained in the following ways:
```cpp
auto a = data.template Get<A>();
```
Note that the compiler will automatically deduce and specify the correct data type for ```a```. For example, the object corresponding to ```A``` is of type ```float```, then after executing the above code , ```a``` is also a variable of type ```float```.

#### Containers and Forward / Backward Propagation
The layer connection in MetaNN provides forward and backward propagation interfaces. These interfaces receive container objects as parameters and return container objects as results. Take ```AddLayer``` as an example:
```cpp
template <typename TPolicies>
class AddLayer {
    using InputPortSet = LayerPortSet<struct LeftOperand, struct RightOperand>;
    using OutputPortSet = LayerPortSet<struct LayerOutput>;

	// ...
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val1 = p_in.template Get<LeftOperand>();
        const auto& val2 = p_in.template Get<RightOperand>();

        return OutputType::Create().template Set<LayerOutput>(val1 + val2);
    }

    template <typename TGrad>
    auto FeedBackward(TGrad&& p_grad)
    {
        const auto& res = p_grad.template Get<LayerOutput>();
	    return AddLayerInput::Create().template Set<LeftOperand>(res)
    	                              .template Set<RightOperand>(res);
    }
};
```
By using the container, we avoid the problems caused by using the structure of ```vector``` as mentioned above. First, each object in the container is retrieved by key, which reduces the possibility of misuse. Second, the objects corresponding to each key in the container can be of different types, this simplifies code writing.

#### The Forward / Backward Propagation is not Actually Calculated
Consider the following code:
```cpp
auto i1 = Matrix<float, DeviceTags::CPU>(2, 3);
auto i2 = Matrix<float, DeviceTags::CPU>(2, 3);

auto input = AddLayerInput::Create().Set<LeftOperand>(i1)
                                    .Set<RightOperand>(i2);
auto out = layer.FeedForward(input).Get<LayerOutput>();
```
At this time, the type of ```out``` is:
```cpp
Operation<OpTags::Add, 
          OperandContainer<Matrix<float, DeviceTags::CPU>, Matrix<float, DeviceTags::CPU>>
          PolicyContainer<>
         >
```
It is a template expression, not an actual calculation result. It is precisely because we set the forward and backward propagation interface of the layer as template functions, and use the container as the input and output object, thus achieving this effect. For many deep learning frameworks, the introduction of layers means that the calculation is limited to the inside of the layer (the input and output of the layer are calculated results), which has an adverse effect on the overall performance optimization of the system. But MetaNN uses templates and compile-time calculations, which effectively solves this problem! In MetaNN, layers are only used to organize code and template expressions, and the final calculations will be left to the last.

### Other Interfaces of the Layer
Each layer in MetaNN needs to provide interfaces for forward and backward propagation. In addition, they may also provide the following interfaces:
 * ```Init```: used to initialize the parameter ontained in the layer.
 * ```LoadWeights```: used to read parameters from outside.
 * ```SaveWeights```: used to export parameters to external.
 * ```GradCollect```: used to obtain gradient information.
 * ```NeutralInvariant```: used to assert that the layer is in a neutral state.

#### About ```NeutralInvariant```
Here is an explanation of the last interface. The MetaNN layer is responsible for saving some intermediate results of forward and backward propagation for use in operations such as gradient calculation. An unused layer should not contain any intermediate results. We call this state the "neutral" state. After the layer has performed a forward propagation and a back propagation, and after obtaining the corresponding gradient information, it should not have any intermediate results. That is, the layer should also be in a neutral state at this time. We can assert this by calling ```NeutralInvariant``` after a forward propagation and a back propagation to ensure the correctness of the program.

#### General Interfaces
Except for forward and backward propagation, the remaining interfaces are optional: not all layers will contain all interfaces. For example, some layers do not have internal parameters, so there is no need for interfaces such as ```Init```. Therefore, given a layer object ```layer```, calling ```layer.Init``` may be wrong. To solve this problem, MetaNN introduces the following general interface functions (the last two functions are introduced for consistency):
 * ```LayerInit```: encapsulation of ```Init```
 * ```LayerLoadWeights```: encapsulation of ```LoadWeights```
 * ```LayerSaveWeights```: encapsulation of ```SaveWeights```
 * ```LayerGradCollect```: encapsulation of ```GradCollect```
 * ```LayerNeutralInvariant```: encapsulation of ```NeutralInvariant```
 * ```LayerFeedForward```: encapsulation of ```FeedForward```
 * ```LayerFeedBackward```: encapsulation of ```FeedBackward```

For example, we can call ```LayerInit(layer, ...)``` for initialization. MetaNN is smart enough, if the ```layer``` object does not contain the ```Init``` interface, then ```LayerInit``` does nothing. Otherwise, the ```Init``` interface of the ```layer``` object will be called.

## Composite Layer
The composite layer realizes a typical compose pattern. It encapsulates the basic layer or other composite layers, and uses relatively basic elements to build complex network structures.

### Design Concept
Compared with the basic layers, composite layers introduce two new concepts due to their natural characteristics: parent-child relationship and topological structure.

A composite layer is a combination of basic layers. Naturally, a composite layer and the basic layers it contains form a parent-child relationship. For a basic layer, we can adjust its behavior details through Policy. Similarly, we also want to use Policy to adjust the behavior of the composite layer. But because of this parent-child relationship, we need to introduce a mechanism that can easily specify the behavior details of each child.

Generally speaking, the basic layers in the composite layer form a directed acyclic graph structure. The composite layer needs to introduce a mechanism to describe this structure, and realize automatic forward and reverse propagation based on the given structure.

Several simple composite layers are implemented in MetaNN: mainly for display purposes, through which you can learn how to implement composite layers. Next, we will take ```WeightLayer``` as an example to describe the structure of the composite layer. ```WeightLayer``` takes an input tensor, multiplies it with another tensor and returns it.

### Describe the Topology
The first step in establishing a composite layer is to describe the sub-layers contained in the composite layer and the connection relationship between these sub-layers, i.e., the topological structure of the composite layer. First, we need to give a unique name to each sublayer in the composite layer. Note that this name is not the type name of the layer. A composite layer may contain multiple sublayers of the same type, and they also need to be distinguished:
```cpp
struct ParamSublayer;
struct DotSublayer;
```
The above code shows that: ```WeightLayer``` contains two sublayers, named as ```ParamSublayer``` and ```DotSublayer```.

On this basis, we can define the topology of the composite layer:
```cpp
struct LayerInput; struct LayerOutput;
using Topology = ComposeTopology<Sublayer<ParamSublayer, ParamSourceLayer>,
                                 Sublayer<DotSublayer, DotLayer>,
                                 InConnect<LayerInput, DotSublayer, LeftOperand>,
                                 InternalConnect<ParamSublayer, LayerOutput, DotSublayer, RightOperand>,
                                 OutConnect<DotSublayer, LayerOutput, LayerOutput>>;
```
```ComposeTopology``` is a structure in MetaNN. Used to describe the topology of the composite layer. It is a template and can receive four types of template parameters:
  * ```Sublayer```: associate the name of the sublayer with the type of the layer.
  * ```InConnect```: connect the input of the composite layer to the input of a sublayer.
  * ```OutConnect```: connect the output of a certain sublayer to the output of the composite layer.
  * ```InternalConnect```: connect the output of one sublayer to the input of another sublayer.

Therefore, the meaning of the sentences in the above ```Topology``` in order are:
  * The sublayer ```ParamSublayer``` is a ```ParamSourceLayer```
  * The sublayer ```DotSublayer``` is a ```DotLayer```
  * In the input container of ```WeightLayer```, there is an element whose key is ``LayerInput```, this element should be connected to ```LeftOperand``` of the input container of ```DotSublayer```
  * In the output container of the ```ParamSublayer```, there is an element whose key is ``LayerOutput```, associate this element to the ```RightOperand` of the input of the ```DotSublayer``` sublayer
  * The element whose key is ```LayerOutput``` in the output container of the ```DotSublayer``` should be associated with the element whose key is ```LayerOutput``` in the output container of ```WeightLayer```

Combined with the functions implemented by ```WeightLayer```, this description is easy to understand. It should be noted that the order of the sentences in the description is arbitrary. For example: the following description is equivalent to the above description:
```cpp
using Topology = ComposeTopology<Sublayer<ParamSublayer, ParamSourceLayer>,
                                 InConnect<LayerInput, DotSublayer, LeftOperand>,
                                 InternalConnect<ParamSublayer, LayerOutput, DotSublayer, RightOperand>,
                                 Sublayer<DotSublayer, DotLayer>,
                                 OutConnect<DotSublayer, LayerOutput, LayerOutput>>;
```

### Define Composite Layer
After having the topology, the definition of the composite layer is very simple. The following is the complete definition of ```WeightLayer```:
```cpp
template <typename TInputs, typename TPolicies>
using Base = ComposeKernel<LayerPortSet<LayerInput>, LayerPortSet<LayerOutput>, TInputs, TPolicies, Topology>;

template <typename TInputs, typename TPolicies>
class WeightLayer : public Base<TInputs, TPolicies>
{
    using TBase = Base<TInputs, TPolicies>;

public:
    template <typename... TShapeParams>
    WeightLayer(const std::string& p_name, TShapeParams&&... shapeParams)
        : TBase(TBase::CreateSublayers()
                    .template Set<ParamSublayer>(p_name + "/param", std::forward<TShapeParams>(shapeParams)...)
                    .template Set<DotSublayer>(p_name + "/add"))
    { }
};
```
```ComposeKernel``` is a template in MetaNN. It achieves almost all the functions required by a composite layer. This template receives 5 parameters, which respectively represent the input/output port set of the composite layer, input type map, Policy container, and topology. The reason for deriving here is that the constructor is not defined in ```ComposeKernel``` (we need to call the sublayers' constructors in the constructor of the composite layer). This is the only work that needs to be done in the definition of ```WeightLayer```.

The definition of the composite layer is not troublesome, because ```ComposeKernel``` has completed most of the work, including a logic of topological ordering implemented during compile time. It is this logic that allows us to realize automatic forward and backward propagation on the basis of only defining the above information.

### Use of Composite Layer
The use of the composite layer is very similar to the use of basic layer, the only difference is the specification of the Policy. In MetaNN, the policy of the sublayer will "inherit" from the Policy of the composite layer. At the same time, the sublayer can also specify its own Policy. If the composite layer conflicts with the Policy of its sublayer, then the sublayer’s policy shall prevail:
```cpp
// use default policies
using RootLayer1 = MakeLayer<WeightLayer>;

// All sublayers have Policy1
using RootLayer2 = MakeLayer<WeightLayer, Policy1Is<true>>;

// ParamSublayer has Policy1Is<false>, 
// DotSublayer has Policy1Is<true>
using RootLayer3 =
    MakeLayer<WeightLayer, Policy1Is<true>,
              SubPolicyContainer<ParamSublayer, Policy1Is<false>>>;

// Similar to RootLayer3
using RootLayer4 =
    MakeLayer<WeightLayer, Policy1Is<false>
              SubPolicyContainer<DotSublayer, Policy1Is<true>>>;
```
```SubPolicyContainer``` is a container specially introduced by MetaNN to describe the policy difference between the sublayer and its parent.

## Evaluation
One of the main ideas of MetaNN is to delay the evaluation process. The result of the forward and backward propagation of the layer is represented as expression templates, and no actual evaluation is performed. Generally speaking, after performing forward/backward propagation, we will get the objects composed of several expression templates. At this time, we can perform unified evaluation. We will see in this section: in this way, we can optimize the evaluation process in multiple dimensions.

### The Simplest Evaluation Statement
The simplest way to call the evaluation statement is as follows:
```cpp
auto rm1 = Matrix<int, DeviceTags::CPU>(4, 5);
auto rm2 = Matrix<int, DeviceTags::CPU>(4, 5);
auto add = rm1 + rm2;

auto add_r = Evaluate(add);
```
```Evaluate``` receives an expression template and return the value corresponding to the expression template.

### Implementation Details of ```Evaluate```
The internal definition of ```Evaluate``` is as follows:
```cpp
template <typename TData>
auto Evaluate(const TData& data)
{
    auto evalHandle = data.EvalRegister();
    EvalPlan::Eval();
    return evalHandle.Data();
}
```
The internal part of ```Evaluate``` is actually an encapsulation of the general evaluation process of the framework. For any expression template (even most basic data types in MetaNN), we can call the ```EvalRegister``` function to register the evaluation, and the registration will return a handle. After that, we need to call ```EvalPlan::Eval()``` to evaluate the registered function. After the evaluation, we can get the result of the calculation through the ```Data``` method of the handle.

Compared to ```Evaluate```, this general evaluation process may be more useful. Consider the following code:
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
Before calling ```EvalPlan::Eval```, the program has registered two evaluation procedures, then the system can be parallelized to a certain extent during the specific evaluation of ```EvalPlan::Eval``` , thereby improving its performance.

And this involves the optimization problem of evaluation. Next, we will discuss several possible optimizations (it should be noted that the existing MetaNN does not implement most of the optimization techniques discussed later. But MetaNN has already built the framework, and implementing these techniques on this basis is relatively ordinary work):

### Avoid Duplicate Calculation
There may be some relationships between multiple expression templates that are registered. For example, some of their subtrees are the same. MetaNN introduces an automatic analysis mechanism to ensure that in this case, the same subtree will only be evaluated once. At the same time, if a subtree has been evaluated before, it will not be evaluated again.

### Optimization based on Computing Characteristics
Additional parameters can be introduced in the calculation, and some of the parameters that can be specified at compile time through the Policy. These parameters specified during compilation will adjust the behavior of the operation. If the behavior of the operation satisfies certain characteristics, then it can be optimized according to it. Compile-time branches can be introduced to optimize according to the characteristics of the operation. Compile-time branches will not adversely affect runtime performance.

### Combination of Similar Operations
In order to achieve higher computing speed, the bottom layer of the framework needs to call some efficient computing libraries. Such as Intel's MKL, or CUDA and so on. These libraries provide some functions that can combine multiple calculations. For example, calculate multiple sets of matrix dot products at the same time. Using the method of evaluation registration, we can combine multiple tasks of this type and complete them at the same time when calling ```EvalPlan::Eval```, thereby increasing the calculation speed.

### Combination of Different Operations
In most cases, the evaluation of the expression template can be seen in the topological order of the graph: to evaluate a node, you need to evaluate the predecessor node first. But in some cases, we can simplify this evaluation process. For example, if the current node is logarithm and its child node is exponent, then we can cancel out the operations of logarithm and exponent. We can also introduce more complex mechanisms to simplify the evaluation process to a certain extent.

In fact, this is a very typical evaluation optimization method. The ```OperSeq_``` template is specially introduced in MetaNN for this purpose to achieve this function. ```OperSeq_``` implements a typical responsibility chain pattern, which can handle "special" situations in expression templates. Taking ```RowSoftmaxGrad``` as an example, MetaNN defines:
```cpp
template <>
struct OperSeq_<OpTags::SoftmaxGrad>
{
    using type = OperCalAlgoChain<NSCaseNLLLossGrad::Calculator,
                                  TailCalculator<NSCaseGen::EvalItem,
                                                 NSCaseGen::EvalGroup,
                                                 PolicyContainer<PPassPolicy>>>;
};
```
It means: first check whether the current structure satisfies ```NSCaseNLLLossGrad::Calculator```, and if so, it will be calculated according to the method defined by ```NSCaseNLLLossGrad::Calculator```. Otherwise, try the normal method (```TailCalculator```).

## Summary
The above is a brief introduction of MetaNN. Compared with other deep learning frameworks, MetaNN uses template meta-programming technology in a deeper level. As mentioned at the beginning of the article, it is currently only a skeleton, and there is still a certain distance from the actual application. At the same time, using this framework also requires a certain template meta-programming foundation, which may be an unfamiliar area for many C++ developers. But it also has its advantages.

The framework uses template meta-programming technology to completely separate calculation optimization from model structure. This can bring many benefits. For example, we may not need the framework's explicit support for mini-batch. In most cases (except for Batch Normalization), batch is introduced to speed up calculation, but the introduction of batch will make the system more complicated. For example, in neural network machine translation, in order to use batch, usually we need to sort the training corpus according to the length of the source language, introduce padding tags, etc., to ensure that batch can be performed more efficiently. We even have to deal with high-dimensional data: many times we need to use matrices to represent intermediate results, and use three-dimensional or higher-dimensional structures to represent batch intermediate results. This greatly increases the complexity of the program and affects its understanding. But in MetaNN, in most cases, all we need to do is introduce loops, and only process one entry in each loop. After the loop is over, call ```EvalPlan::Eval``` uniformly, and ```EvalPlan::Eval``` will help us complete the calculation and merging operations that previously required batch. The logic of the program will be clearer.

Another obvious benefit is that users will be more comfortable in using it, without having to consider some tips. A very typical example is: many deep learning frameworks contain layers of ```Softmax``` and ```SoftmaxLoss``` or similar structures, because the latter needs to handle the numerical stability problem introduced by ```Softmax```. With MetaNN, this problem is concealed in the evaluation optimization. Users no longer need to pay attention to the detailed differences in the behavior of these layers.

In addition to the deep learning framework, MetaNN also contributes to C++ template metaprogramming. The concept of Policy is not original MetaNN, but it extends the implementation of the original method, so that the number of policies in a template is no longer limited (in fact, the number of policies will also depend on the compiler's support for variable-length templates degree). MetaNN also implements a compile-time container, provides in-depth discussions on data classification and policy organization. The compile-time topological sorting realized by it also expands the technology of meta-programming from "tips" to practical algorithms. It can be said that if you can read this set of codes, you will have a relatively in-depth understanding of C++ metaprogramming.

The last thing to say is: this set of codes has no comments. This document is an introduction to MetaNN, but it does not go deep into the details. It is difficult to understand the implementation of the program through this document. MetaNN contains a lot of details. The reason why there is no comment is because if readers have some understanding of C++ template metaprogramming, then I think reading the logic in this code is more straightforward. Conversely, if the reader does not understand C++ template metaprogramming, or knows less, then adding comments will not help much. To fully describe the implementation details in this framework, you need the length of a book-the Chinese version of the book ("C++ Template Metaprogramming Practice") has been published. However, the content in the book was completed before refactor, so the book is somewhat different from the content on the current branch. The content in the book corresponds to the "book" branch. At present, I am considering publishing another version corresponding to the current system framework.
