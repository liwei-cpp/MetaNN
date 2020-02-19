##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=DataOpTest2
ConfigurationName      :=Debug
WorkspacePath          :=/home/a/MetaNN
ProjectPath            :=/home/a/MetaNN/Tests/DataOpTest2
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=
Date                   :=01/28/20
CodeLitePath           :=/home/a/.codelite
LinkerName             :=g++
SharedObjectLinkerName :=g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.o.i
DebugSwitch            :=-gstab
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E 
ObjectsFileList        :="DataOpTest2.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). $(IncludeSwitch)../ $(IncludeSwitch)../.. 
IncludePCH             := 
RcIncludePath          := 
Libs                   := 
ArLibs                 :=  
LibPath                := $(LibraryPathSwitch). 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := ar rcus
CXX      := g++
CC       := gcc
CXXFLAGS :=  -g -O0 -Wall -std=c++17 $(Preprocessors)
CFLAGS   :=  -g -O0 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_scalable_tensor.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_bias_vector.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_zero_tensor.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_trival_tensor.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_tensor.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_scalar.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_elementwise_test_add.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_activation_test_tanh.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_activation_test_relu.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_mutating_test_permute.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/data_facilities_test_shape.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

MakeIntermediateDirs:
	@test -d ./Debug || $(MakeDirCommand) ./Debug


$(IntermediateDirectory)/.d:
	@test -d ./Debug || $(MakeDirCommand) ./Debug

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/main.cpp$(ObjectSuffix): main.cpp $(IntermediateDirectory)/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM main.cpp

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) main.cpp

$(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix): data/test_dynamic.cpp $(IntermediateDirectory)/data_test_dynamic.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/data/test_dynamic.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_dynamic.cpp$(DependSuffix): data/test_dynamic.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_dynamic.cpp$(DependSuffix) -MM data/test_dynamic.cpp

$(IntermediateDirectory)/data_test_dynamic.cpp$(PreprocessSuffix): data/test_dynamic.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_dynamic.cpp$(PreprocessSuffix) data/test_dynamic.cpp

$(IntermediateDirectory)/data_test_scalable_tensor.cpp$(ObjectSuffix): data/test_scalable_tensor.cpp $(IntermediateDirectory)/data_test_scalable_tensor.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/data/test_scalable_tensor.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_scalable_tensor.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_scalable_tensor.cpp$(DependSuffix): data/test_scalable_tensor.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_scalable_tensor.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_scalable_tensor.cpp$(DependSuffix) -MM data/test_scalable_tensor.cpp

$(IntermediateDirectory)/data_test_scalable_tensor.cpp$(PreprocessSuffix): data/test_scalable_tensor.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_scalable_tensor.cpp$(PreprocessSuffix) data/test_scalable_tensor.cpp

$(IntermediateDirectory)/data_test_bias_vector.cpp$(ObjectSuffix): data/test_bias_vector.cpp $(IntermediateDirectory)/data_test_bias_vector.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/data/test_bias_vector.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_bias_vector.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_bias_vector.cpp$(DependSuffix): data/test_bias_vector.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_bias_vector.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_bias_vector.cpp$(DependSuffix) -MM data/test_bias_vector.cpp

$(IntermediateDirectory)/data_test_bias_vector.cpp$(PreprocessSuffix): data/test_bias_vector.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_bias_vector.cpp$(PreprocessSuffix) data/test_bias_vector.cpp

$(IntermediateDirectory)/data_test_zero_tensor.cpp$(ObjectSuffix): data/test_zero_tensor.cpp $(IntermediateDirectory)/data_test_zero_tensor.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/data/test_zero_tensor.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_zero_tensor.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_zero_tensor.cpp$(DependSuffix): data/test_zero_tensor.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_zero_tensor.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_zero_tensor.cpp$(DependSuffix) -MM data/test_zero_tensor.cpp

$(IntermediateDirectory)/data_test_zero_tensor.cpp$(PreprocessSuffix): data/test_zero_tensor.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_zero_tensor.cpp$(PreprocessSuffix) data/test_zero_tensor.cpp

$(IntermediateDirectory)/data_test_trival_tensor.cpp$(ObjectSuffix): data/test_trival_tensor.cpp $(IntermediateDirectory)/data_test_trival_tensor.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/data/test_trival_tensor.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_trival_tensor.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_trival_tensor.cpp$(DependSuffix): data/test_trival_tensor.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_trival_tensor.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_trival_tensor.cpp$(DependSuffix) -MM data/test_trival_tensor.cpp

$(IntermediateDirectory)/data_test_trival_tensor.cpp$(PreprocessSuffix): data/test_trival_tensor.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_trival_tensor.cpp$(PreprocessSuffix) data/test_trival_tensor.cpp

$(IntermediateDirectory)/data_test_tensor.cpp$(ObjectSuffix): data/test_tensor.cpp $(IntermediateDirectory)/data_test_tensor.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/data/test_tensor.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_tensor.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_tensor.cpp$(DependSuffix): data/test_tensor.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_tensor.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_tensor.cpp$(DependSuffix) -MM data/test_tensor.cpp

$(IntermediateDirectory)/data_test_tensor.cpp$(PreprocessSuffix): data/test_tensor.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_tensor.cpp$(PreprocessSuffix) data/test_tensor.cpp

$(IntermediateDirectory)/data_test_scalar.cpp$(ObjectSuffix): data/test_scalar.cpp $(IntermediateDirectory)/data_test_scalar.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/data/test_scalar.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_scalar.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_scalar.cpp$(DependSuffix): data/test_scalar.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_scalar.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_scalar.cpp$(DependSuffix) -MM data/test_scalar.cpp

$(IntermediateDirectory)/data_test_scalar.cpp$(PreprocessSuffix): data/test_scalar.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_scalar.cpp$(PreprocessSuffix) data/test_scalar.cpp

$(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(ObjectSuffix): operators/elementwise/test_negative.cpp $(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/elementwise/test_negative.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(DependSuffix): operators/elementwise/test_negative.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(DependSuffix) -MM operators/elementwise/test_negative.cpp

$(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(PreprocessSuffix): operators/elementwise/test_negative.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_elementwise_test_negative.cpp$(PreprocessSuffix) operators/elementwise/test_negative.cpp

$(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(ObjectSuffix): operators/elementwise/test_asin.cpp $(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/elementwise/test_asin.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(DependSuffix): operators/elementwise/test_asin.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(DependSuffix) -MM operators/elementwise/test_asin.cpp

$(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(PreprocessSuffix): operators/elementwise/test_asin.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_elementwise_test_asin.cpp$(PreprocessSuffix) operators/elementwise/test_asin.cpp

$(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(ObjectSuffix): operators/elementwise/test_sign.cpp $(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/elementwise/test_sign.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(DependSuffix): operators/elementwise/test_sign.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(DependSuffix) -MM operators/elementwise/test_sign.cpp

$(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(PreprocessSuffix): operators/elementwise/test_sign.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_elementwise_test_sign.cpp$(PreprocessSuffix) operators/elementwise/test_sign.cpp

$(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(ObjectSuffix): operators/elementwise/test_acos.cpp $(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/elementwise/test_acos.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(DependSuffix): operators/elementwise/test_acos.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(DependSuffix) -MM operators/elementwise/test_acos.cpp

$(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(PreprocessSuffix): operators/elementwise/test_acos.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_elementwise_test_acos.cpp$(PreprocessSuffix) operators/elementwise/test_acos.cpp

$(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(ObjectSuffix): operators/elementwise/test_abs.cpp $(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/elementwise/test_abs.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(DependSuffix): operators/elementwise/test_abs.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(DependSuffix) -MM operators/elementwise/test_abs.cpp

$(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(PreprocessSuffix): operators/elementwise/test_abs.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_elementwise_test_abs.cpp$(PreprocessSuffix) operators/elementwise/test_abs.cpp

$(IntermediateDirectory)/operators_elementwise_test_add.cpp$(ObjectSuffix): operators/elementwise/test_add.cpp $(IntermediateDirectory)/operators_elementwise_test_add.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/elementwise/test_add.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_elementwise_test_add.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_elementwise_test_add.cpp$(DependSuffix): operators/elementwise/test_add.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_elementwise_test_add.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_elementwise_test_add.cpp$(DependSuffix) -MM operators/elementwise/test_add.cpp

$(IntermediateDirectory)/operators_elementwise_test_add.cpp$(PreprocessSuffix): operators/elementwise/test_add.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_elementwise_test_add.cpp$(PreprocessSuffix) operators/elementwise/test_add.cpp

$(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(ObjectSuffix): operators/elementwise/test_substract.cpp $(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/elementwise/test_substract.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(DependSuffix): operators/elementwise/test_substract.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(DependSuffix) -MM operators/elementwise/test_substract.cpp

$(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(PreprocessSuffix): operators/elementwise/test_substract.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_elementwise_test_substract.cpp$(PreprocessSuffix) operators/elementwise/test_substract.cpp

$(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(ObjectSuffix): operators/elementwise/test_multiply.cpp $(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/elementwise/test_multiply.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(DependSuffix): operators/elementwise/test_multiply.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(DependSuffix) -MM operators/elementwise/test_multiply.cpp

$(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(PreprocessSuffix): operators/elementwise/test_multiply.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_elementwise_test_multiply.cpp$(PreprocessSuffix) operators/elementwise/test_multiply.cpp

$(IntermediateDirectory)/operators_activation_test_tanh.cpp$(ObjectSuffix): operators/activation/test_tanh.cpp $(IntermediateDirectory)/operators_activation_test_tanh.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/activation/test_tanh.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_activation_test_tanh.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_activation_test_tanh.cpp$(DependSuffix): operators/activation/test_tanh.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_activation_test_tanh.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_activation_test_tanh.cpp$(DependSuffix) -MM operators/activation/test_tanh.cpp

$(IntermediateDirectory)/operators_activation_test_tanh.cpp$(PreprocessSuffix): operators/activation/test_tanh.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_activation_test_tanh.cpp$(PreprocessSuffix) operators/activation/test_tanh.cpp

$(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(ObjectSuffix): operators/activation/test_sigmoid.cpp $(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/activation/test_sigmoid.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(DependSuffix): operators/activation/test_sigmoid.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(DependSuffix) -MM operators/activation/test_sigmoid.cpp

$(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(PreprocessSuffix): operators/activation/test_sigmoid.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_activation_test_sigmoid.cpp$(PreprocessSuffix) operators/activation/test_sigmoid.cpp

$(IntermediateDirectory)/operators_activation_test_relu.cpp$(ObjectSuffix): operators/activation/test_relu.cpp $(IntermediateDirectory)/operators_activation_test_relu.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/activation/test_relu.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_activation_test_relu.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_activation_test_relu.cpp$(DependSuffix): operators/activation/test_relu.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_activation_test_relu.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_activation_test_relu.cpp$(DependSuffix) -MM operators/activation/test_relu.cpp

$(IntermediateDirectory)/operators_activation_test_relu.cpp$(PreprocessSuffix): operators/activation/test_relu.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_activation_test_relu.cpp$(PreprocessSuffix) operators/activation/test_relu.cpp

$(IntermediateDirectory)/operators_mutating_test_permute.cpp$(ObjectSuffix): operators/mutating/test_permute.cpp $(IntermediateDirectory)/operators_mutating_test_permute.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/operators/mutating/test_permute.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_mutating_test_permute.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_mutating_test_permute.cpp$(DependSuffix): operators/mutating/test_permute.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_mutating_test_permute.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_mutating_test_permute.cpp$(DependSuffix) -MM operators/mutating/test_permute.cpp

$(IntermediateDirectory)/operators_mutating_test_permute.cpp$(PreprocessSuffix): operators/mutating/test_permute.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_mutating_test_permute.cpp$(PreprocessSuffix) operators/mutating/test_permute.cpp

$(IntermediateDirectory)/data_facilities_test_shape.cpp$(ObjectSuffix): data/facilities/test_shape.cpp $(IntermediateDirectory)/data_facilities_test_shape.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/a/MetaNN/Tests/DataOpTest2/data/facilities/test_shape.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_facilities_test_shape.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_facilities_test_shape.cpp$(DependSuffix): data/facilities/test_shape.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_facilities_test_shape.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_facilities_test_shape.cpp$(DependSuffix) -MM data/facilities/test_shape.cpp

$(IntermediateDirectory)/data_facilities_test_shape.cpp$(PreprocessSuffix): data/facilities/test_shape.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_facilities_test_shape.cpp$(PreprocessSuffix) data/facilities/test_shape.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


