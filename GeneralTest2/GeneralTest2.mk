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
Date                   :=05/12/18
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
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_test_dynamic.cpp$(ObjectSuffix) $(IntermediateDirectory)/operators_test_duplicate.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_batch_test_static_batch.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_batch_test_dynamic_batch.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_sequence_test_static_sequence.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_sequence_test_dynamic_sequence.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_batch_sequence_test_static_batch_sequence.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_batch_sequence_test_dynamic_batch_sequence.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_vector.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_one_hot_vector.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_3d_array_test_3d_array.cpp$(ObjectSuffix) 



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


