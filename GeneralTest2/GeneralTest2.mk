##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=GeneralTest2
ConfigurationName      :=Debug
WorkspacePath          :=/home/liwei/MetaNN/new/MetaNN
ProjectPath            :=/home/liwei/MetaNN/new/MetaNN/GeneralTest2
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=liwei
Date                   :=14/12/18
CodeLitePath           :=/home/liwei/.codelite
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
ObjectsFileList        :="GeneralTest2.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). $(IncludeSwitch).. 
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
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_duplicate.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_collapse.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_abs.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_sign.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_add.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_substract.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_multiply.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_divide.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/operators_test_acos.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_acos_grad.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_sigmoid.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_asin.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_asin_grad.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_tanh.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_tanh_grad.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_transpose.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_batch_test_static_batch.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(ObjectSuffix) 



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
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM main.cpp

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) main.cpp

$(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix): data/test_dynamic.cpp $(IntermediateDirectory)/data_test_dynamic.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/test_dynamic.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_test_dynamic.cpp$(DependSuffix): data/test_dynamic.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_test_dynamic.cpp$(DependSuffix) -MM data/test_dynamic.cpp

$(IntermediateDirectory)/data_test_dynamic.cpp$(PreprocessSuffix): data/test_dynamic.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_test_dynamic.cpp$(PreprocessSuffix) data/test_dynamic.cpp

$(IntermediateDirectory)/operators_test_duplicate.cpp$(ObjectSuffix): operators/test_duplicate.cpp $(IntermediateDirectory)/operators_test_duplicate.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_duplicate.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_duplicate.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_duplicate.cpp$(DependSuffix): operators/test_duplicate.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_duplicate.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_duplicate.cpp$(DependSuffix) -MM operators/test_duplicate.cpp

$(IntermediateDirectory)/operators_test_duplicate.cpp$(PreprocessSuffix): operators/test_duplicate.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_duplicate.cpp$(PreprocessSuffix) operators/test_duplicate.cpp

$(IntermediateDirectory)/operators_test_collapse.cpp$(ObjectSuffix): operators/test_collapse.cpp $(IntermediateDirectory)/operators_test_collapse.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_collapse.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_collapse.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_collapse.cpp$(DependSuffix): operators/test_collapse.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_collapse.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_collapse.cpp$(DependSuffix) -MM operators/test_collapse.cpp

$(IntermediateDirectory)/operators_test_collapse.cpp$(PreprocessSuffix): operators/test_collapse.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_collapse.cpp$(PreprocessSuffix) operators/test_collapse.cpp

$(IntermediateDirectory)/operators_test_abs.cpp$(ObjectSuffix): operators/test_abs.cpp $(IntermediateDirectory)/operators_test_abs.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_abs.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_abs.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_abs.cpp$(DependSuffix): operators/test_abs.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_abs.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_abs.cpp$(DependSuffix) -MM operators/test_abs.cpp

$(IntermediateDirectory)/operators_test_abs.cpp$(PreprocessSuffix): operators/test_abs.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_abs.cpp$(PreprocessSuffix) operators/test_abs.cpp

$(IntermediateDirectory)/operators_test_sign.cpp$(ObjectSuffix): operators/test_sign.cpp $(IntermediateDirectory)/operators_test_sign.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_sign.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_sign.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_sign.cpp$(DependSuffix): operators/test_sign.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_sign.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_sign.cpp$(DependSuffix) -MM operators/test_sign.cpp

$(IntermediateDirectory)/operators_test_sign.cpp$(PreprocessSuffix): operators/test_sign.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_sign.cpp$(PreprocessSuffix) operators/test_sign.cpp

$(IntermediateDirectory)/operators_test_add.cpp$(ObjectSuffix): operators/test_add.cpp $(IntermediateDirectory)/operators_test_add.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_add.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_add.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_add.cpp$(DependSuffix): operators/test_add.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_add.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_add.cpp$(DependSuffix) -MM operators/test_add.cpp

$(IntermediateDirectory)/operators_test_add.cpp$(PreprocessSuffix): operators/test_add.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_add.cpp$(PreprocessSuffix) operators/test_add.cpp

$(IntermediateDirectory)/operators_test_substract.cpp$(ObjectSuffix): operators/test_substract.cpp $(IntermediateDirectory)/operators_test_substract.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_substract.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_substract.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_substract.cpp$(DependSuffix): operators/test_substract.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_substract.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_substract.cpp$(DependSuffix) -MM operators/test_substract.cpp

$(IntermediateDirectory)/operators_test_substract.cpp$(PreprocessSuffix): operators/test_substract.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_substract.cpp$(PreprocessSuffix) operators/test_substract.cpp

$(IntermediateDirectory)/operators_test_multiply.cpp$(ObjectSuffix): operators/test_multiply.cpp $(IntermediateDirectory)/operators_test_multiply.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_multiply.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_multiply.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_multiply.cpp$(DependSuffix): operators/test_multiply.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_multiply.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_multiply.cpp$(DependSuffix) -MM operators/test_multiply.cpp

$(IntermediateDirectory)/operators_test_multiply.cpp$(PreprocessSuffix): operators/test_multiply.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_multiply.cpp$(PreprocessSuffix) operators/test_multiply.cpp

$(IntermediateDirectory)/operators_test_divide.cpp$(ObjectSuffix): operators/test_divide.cpp $(IntermediateDirectory)/operators_test_divide.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_divide.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_divide.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_divide.cpp$(DependSuffix): operators/test_divide.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_divide.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_divide.cpp$(DependSuffix) -MM operators/test_divide.cpp

$(IntermediateDirectory)/operators_test_divide.cpp$(PreprocessSuffix): operators/test_divide.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_divide.cpp$(PreprocessSuffix) operators/test_divide.cpp

$(IntermediateDirectory)/operators_test_acos.cpp$(ObjectSuffix): operators/test_acos.cpp $(IntermediateDirectory)/operators_test_acos.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_acos.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_acos.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_acos.cpp$(DependSuffix): operators/test_acos.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_acos.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_acos.cpp$(DependSuffix) -MM operators/test_acos.cpp

$(IntermediateDirectory)/operators_test_acos.cpp$(PreprocessSuffix): operators/test_acos.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_acos.cpp$(PreprocessSuffix) operators/test_acos.cpp

$(IntermediateDirectory)/operators_test_acos_grad.cpp$(ObjectSuffix): operators/test_acos_grad.cpp $(IntermediateDirectory)/operators_test_acos_grad.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_acos_grad.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_acos_grad.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_acos_grad.cpp$(DependSuffix): operators/test_acos_grad.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_acos_grad.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_acos_grad.cpp$(DependSuffix) -MM operators/test_acos_grad.cpp

$(IntermediateDirectory)/operators_test_acos_grad.cpp$(PreprocessSuffix): operators/test_acos_grad.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_acos_grad.cpp$(PreprocessSuffix) operators/test_acos_grad.cpp

$(IntermediateDirectory)/operators_test_sigmoid.cpp$(ObjectSuffix): operators/test_sigmoid.cpp $(IntermediateDirectory)/operators_test_sigmoid.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_sigmoid.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_sigmoid.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_sigmoid.cpp$(DependSuffix): operators/test_sigmoid.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_sigmoid.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_sigmoid.cpp$(DependSuffix) -MM operators/test_sigmoid.cpp

$(IntermediateDirectory)/operators_test_sigmoid.cpp$(PreprocessSuffix): operators/test_sigmoid.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_sigmoid.cpp$(PreprocessSuffix) operators/test_sigmoid.cpp

$(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(ObjectSuffix): operators/test_sigmoid_grad.cpp $(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_sigmoid_grad.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(DependSuffix): operators/test_sigmoid_grad.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(DependSuffix) -MM operators/test_sigmoid_grad.cpp

$(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(PreprocessSuffix): operators/test_sigmoid_grad.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_sigmoid_grad.cpp$(PreprocessSuffix) operators/test_sigmoid_grad.cpp

$(IntermediateDirectory)/operators_test_asin.cpp$(ObjectSuffix): operators/test_asin.cpp $(IntermediateDirectory)/operators_test_asin.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_asin.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_asin.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_asin.cpp$(DependSuffix): operators/test_asin.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_asin.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_asin.cpp$(DependSuffix) -MM operators/test_asin.cpp

$(IntermediateDirectory)/operators_test_asin.cpp$(PreprocessSuffix): operators/test_asin.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_asin.cpp$(PreprocessSuffix) operators/test_asin.cpp

$(IntermediateDirectory)/operators_test_asin_grad.cpp$(ObjectSuffix): operators/test_asin_grad.cpp $(IntermediateDirectory)/operators_test_asin_grad.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_asin_grad.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_asin_grad.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_asin_grad.cpp$(DependSuffix): operators/test_asin_grad.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_asin_grad.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_asin_grad.cpp$(DependSuffix) -MM operators/test_asin_grad.cpp

$(IntermediateDirectory)/operators_test_asin_grad.cpp$(PreprocessSuffix): operators/test_asin_grad.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_asin_grad.cpp$(PreprocessSuffix) operators/test_asin_grad.cpp

$(IntermediateDirectory)/operators_test_tanh.cpp$(ObjectSuffix): operators/test_tanh.cpp $(IntermediateDirectory)/operators_test_tanh.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_tanh.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_tanh.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_tanh.cpp$(DependSuffix): operators/test_tanh.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_tanh.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_tanh.cpp$(DependSuffix) -MM operators/test_tanh.cpp

$(IntermediateDirectory)/operators_test_tanh.cpp$(PreprocessSuffix): operators/test_tanh.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_tanh.cpp$(PreprocessSuffix) operators/test_tanh.cpp

$(IntermediateDirectory)/operators_test_tanh_grad.cpp$(ObjectSuffix): operators/test_tanh_grad.cpp $(IntermediateDirectory)/operators_test_tanh_grad.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_tanh_grad.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_tanh_grad.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_tanh_grad.cpp$(DependSuffix): operators/test_tanh_grad.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_tanh_grad.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_tanh_grad.cpp$(DependSuffix) -MM operators/test_tanh_grad.cpp

$(IntermediateDirectory)/operators_test_tanh_grad.cpp$(PreprocessSuffix): operators/test_tanh_grad.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_tanh_grad.cpp$(PreprocessSuffix) operators/test_tanh_grad.cpp

$(IntermediateDirectory)/operators_test_transpose.cpp$(ObjectSuffix): operators/test_transpose.cpp $(IntermediateDirectory)/operators_test_transpose.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/operators/test_transpose.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/operators_test_transpose.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/operators_test_transpose.cpp$(DependSuffix): operators/test_transpose.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/operators_test_transpose.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/operators_test_transpose.cpp$(DependSuffix) -MM operators/test_transpose.cpp

$(IntermediateDirectory)/operators_test_transpose.cpp$(PreprocessSuffix): operators/test_transpose.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/operators_test_transpose.cpp$(PreprocessSuffix) operators/test_transpose.cpp

$(IntermediateDirectory)/data_batch_test_static_batch.cpp$(ObjectSuffix): data/batch/test_static_batch.cpp $(IntermediateDirectory)/data_batch_test_static_batch.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/batch/test_static_batch.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_batch_test_static_batch.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_batch_test_static_batch.cpp$(DependSuffix): data/batch/test_static_batch.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_batch_test_static_batch.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_batch_test_static_batch.cpp$(DependSuffix) -MM data/batch/test_static_batch.cpp

$(IntermediateDirectory)/data_batch_test_static_batch.cpp$(PreprocessSuffix): data/batch/test_static_batch.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_batch_test_static_batch.cpp$(PreprocessSuffix) data/batch/test_static_batch.cpp

$(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(ObjectSuffix): data/batch/test_dynamic_batch.cpp $(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/batch/test_dynamic_batch.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(DependSuffix): data/batch/test_dynamic_batch.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(DependSuffix) -MM data/batch/test_dynamic_batch.cpp

$(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(PreprocessSuffix): data/batch/test_dynamic_batch.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(PreprocessSuffix) data/batch/test_dynamic_batch.cpp

$(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(ObjectSuffix): data/sequence/test_static_sequence.cpp $(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/sequence/test_static_sequence.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(DependSuffix): data/sequence/test_static_sequence.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(DependSuffix) -MM data/sequence/test_static_sequence.cpp

$(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(PreprocessSuffix): data/sequence/test_static_sequence.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(PreprocessSuffix) data/sequence/test_static_sequence.cpp

$(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(ObjectSuffix): data/sequence/test_dynamic_sequence.cpp $(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/sequence/test_dynamic_sequence.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(DependSuffix): data/sequence/test_dynamic_sequence.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(DependSuffix) -MM data/sequence/test_dynamic_sequence.cpp

$(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(PreprocessSuffix): data/sequence/test_dynamic_sequence.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(PreprocessSuffix) data/sequence/test_dynamic_sequence.cpp

$(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(ObjectSuffix): data/batch_sequence/test_static_batch_sequence.cpp $(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/batch_sequence/test_static_batch_sequence.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(DependSuffix): data/batch_sequence/test_static_batch_sequence.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(DependSuffix) -MM data/batch_sequence/test_static_batch_sequence.cpp

$(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(PreprocessSuffix): data/batch_sequence/test_static_batch_sequence.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(PreprocessSuffix) data/batch_sequence/test_static_batch_sequence.cpp

$(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(ObjectSuffix): data/batch_sequence/test_dynamic_batch_sequence.cpp $(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/batch_sequence/test_dynamic_batch_sequence.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(DependSuffix): data/batch_sequence/test_dynamic_batch_sequence.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(DependSuffix) -MM data/batch_sequence/test_dynamic_batch_sequence.cpp

$(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(PreprocessSuffix): data/batch_sequence/test_dynamic_batch_sequence.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(PreprocessSuffix) data/batch_sequence/test_dynamic_batch_sequence.cpp

$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix): data/cardinal/scalar/test_scalar.cpp $(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/cardinal/scalar/test_scalar.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(DependSuffix): data/cardinal/scalar/test_scalar.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(DependSuffix) -MM data/cardinal/scalar/test_scalar.cpp

$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(PreprocessSuffix): data/cardinal/scalar/test_scalar.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(PreprocessSuffix) data/cardinal/scalar/test_scalar.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix): data/cardinal/matrix/test_matrix.cpp $(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/cardinal/matrix/test_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(DependSuffix): data/cardinal/matrix/test_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(DependSuffix) -MM data/cardinal/matrix/test_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(PreprocessSuffix): data/cardinal/matrix/test_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(PreprocessSuffix) data/cardinal/matrix/test_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix): data/cardinal/matrix/test_zero_matrix.cpp $(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/cardinal/matrix/test_zero_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(DependSuffix): data/cardinal/matrix/test_zero_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(DependSuffix) -MM data/cardinal/matrix/test_zero_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(PreprocessSuffix): data/cardinal/matrix/test_zero_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(PreprocessSuffix) data/cardinal/matrix/test_zero_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix): data/cardinal/matrix/test_trival_matrix.cpp $(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/cardinal/matrix/test_trival_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(DependSuffix): data/cardinal/matrix/test_trival_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(DependSuffix) -MM data/cardinal/matrix/test_trival_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(PreprocessSuffix): data/cardinal/matrix/test_trival_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(PreprocessSuffix) data/cardinal/matrix/test_trival_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(ObjectSuffix): data/cardinal/matrix/test_vector.cpp $(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/cardinal/matrix/test_vector.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(DependSuffix): data/cardinal/matrix/test_vector.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(DependSuffix) -MM data/cardinal/matrix/test_vector.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(PreprocessSuffix): data/cardinal/matrix/test_vector.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(PreprocessSuffix) data/cardinal/matrix/test_vector.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(ObjectSuffix): data/cardinal/matrix/test_one_hot_vector.cpp $(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/cardinal/matrix/test_one_hot_vector.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(DependSuffix): data/cardinal/matrix/test_one_hot_vector.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(DependSuffix) -MM data/cardinal/matrix/test_one_hot_vector.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(PreprocessSuffix): data/cardinal/matrix/test_one_hot_vector.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(PreprocessSuffix) data/cardinal/matrix/test_one_hot_vector.cpp

$(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(ObjectSuffix): data/cardinal/3d_array/test_3d_array.cpp $(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/GeneralTest2/data/cardinal/3d_array/test_3d_array.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(DependSuffix): data/cardinal/3d_array/test_3d_array.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(DependSuffix) -MM data/cardinal/3d_array/test_3d_array.cpp

$(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(PreprocessSuffix): data/cardinal/3d_array/test_3d_array.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(PreprocessSuffix) data/cardinal/3d_array/test_3d_array.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


