##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=OtherTests
ConfigurationName      :=Debug
WorkspacePath          :=/home/liwei/MetaNN/new/MetaNN
ProjectPath            :=/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=liwei
Date                   :=29/11/19
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
ObjectsFileList        :="OtherTests.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). $(IncludeSwitch).. $(IncludeSwitch)../.. 
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
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/facilities_test_sequential.cpp$(ObjectSuffix) $(IntermediateDirectory)/facilities_test_map.cpp$(ObjectSuffix) $(IntermediateDirectory)/facilities_test_multi_map.cpp$(ObjectSuffix) $(IntermediateDirectory)/facilities_test_set.cpp$(ObjectSuffix) $(IntermediateDirectory)/policies_test_policy_operations.cpp$(ObjectSuffix) $(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(ObjectSuffix) $(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(ObjectSuffix) $(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(ObjectSuffix) $(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(ObjectSuffix) \
	$(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(ObjectSuffix) $(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(ObjectSuffix) 



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
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM main.cpp

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) main.cpp

$(IntermediateDirectory)/facilities_test_sequential.cpp$(ObjectSuffix): facilities/test_sequential.cpp $(IntermediateDirectory)/facilities_test_sequential.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/facilities/test_sequential.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/facilities_test_sequential.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/facilities_test_sequential.cpp$(DependSuffix): facilities/test_sequential.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/facilities_test_sequential.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/facilities_test_sequential.cpp$(DependSuffix) -MM facilities/test_sequential.cpp

$(IntermediateDirectory)/facilities_test_sequential.cpp$(PreprocessSuffix): facilities/test_sequential.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/facilities_test_sequential.cpp$(PreprocessSuffix) facilities/test_sequential.cpp

$(IntermediateDirectory)/facilities_test_map.cpp$(ObjectSuffix): facilities/test_map.cpp $(IntermediateDirectory)/facilities_test_map.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/facilities/test_map.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/facilities_test_map.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/facilities_test_map.cpp$(DependSuffix): facilities/test_map.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/facilities_test_map.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/facilities_test_map.cpp$(DependSuffix) -MM facilities/test_map.cpp

$(IntermediateDirectory)/facilities_test_map.cpp$(PreprocessSuffix): facilities/test_map.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/facilities_test_map.cpp$(PreprocessSuffix) facilities/test_map.cpp

$(IntermediateDirectory)/facilities_test_multi_map.cpp$(ObjectSuffix): facilities/test_multi_map.cpp $(IntermediateDirectory)/facilities_test_multi_map.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/facilities/test_multi_map.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/facilities_test_multi_map.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/facilities_test_multi_map.cpp$(DependSuffix): facilities/test_multi_map.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/facilities_test_multi_map.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/facilities_test_multi_map.cpp$(DependSuffix) -MM facilities/test_multi_map.cpp

$(IntermediateDirectory)/facilities_test_multi_map.cpp$(PreprocessSuffix): facilities/test_multi_map.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/facilities_test_multi_map.cpp$(PreprocessSuffix) facilities/test_multi_map.cpp

$(IntermediateDirectory)/facilities_test_set.cpp$(ObjectSuffix): facilities/test_set.cpp $(IntermediateDirectory)/facilities_test_set.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/facilities/test_set.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/facilities_test_set.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/facilities_test_set.cpp$(DependSuffix): facilities/test_set.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/facilities_test_set.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/facilities_test_set.cpp$(DependSuffix) -MM facilities/test_set.cpp

$(IntermediateDirectory)/facilities_test_set.cpp$(PreprocessSuffix): facilities/test_set.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/facilities_test_set.cpp$(PreprocessSuffix) facilities/test_set.cpp

$(IntermediateDirectory)/policies_test_policy_operations.cpp$(ObjectSuffix): policies/test_policy_operations.cpp $(IntermediateDirectory)/policies_test_policy_operations.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/policies/test_policy_operations.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/policies_test_policy_operations.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/policies_test_policy_operations.cpp$(DependSuffix): policies/test_policy_operations.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/policies_test_policy_operations.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/policies_test_policy_operations.cpp$(DependSuffix) -MM policies/test_policy_operations.cpp

$(IntermediateDirectory)/policies_test_policy_operations.cpp$(PreprocessSuffix): policies/test_policy_operations.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/policies_test_policy_operations.cpp$(PreprocessSuffix) policies/test_policy_operations.cpp

$(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(ObjectSuffix): model/param_initializer/test_constant_filler.cpp $(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/model/param_initializer/test_constant_filler.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(DependSuffix): model/param_initializer/test_constant_filler.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(DependSuffix) -MM model/param_initializer/test_constant_filler.cpp

$(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(PreprocessSuffix): model/param_initializer/test_constant_filler.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/model_param_initializer_test_constant_filler.cpp$(PreprocessSuffix) model/param_initializer/test_constant_filler.cpp

$(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(ObjectSuffix): model/param_initializer/test_gaussian_filler.cpp $(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/model/param_initializer/test_gaussian_filler.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(DependSuffix): model/param_initializer/test_gaussian_filler.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(DependSuffix) -MM model/param_initializer/test_gaussian_filler.cpp

$(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(PreprocessSuffix): model/param_initializer/test_gaussian_filler.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/model_param_initializer_test_gaussian_filler.cpp$(PreprocessSuffix) model/param_initializer/test_gaussian_filler.cpp

$(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(ObjectSuffix): model/param_initializer/test_uniform_filler.cpp $(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/model/param_initializer/test_uniform_filler.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(DependSuffix): model/param_initializer/test_uniform_filler.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(DependSuffix) -MM model/param_initializer/test_uniform_filler.cpp

$(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(PreprocessSuffix): model/param_initializer/test_uniform_filler.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/model_param_initializer_test_uniform_filler.cpp$(PreprocessSuffix) model/param_initializer/test_uniform_filler.cpp

$(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(ObjectSuffix): model/param_initializer/test_param_initializer.cpp $(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/model/param_initializer/test_param_initializer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(DependSuffix): model/param_initializer/test_param_initializer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(DependSuffix) -MM model/param_initializer/test_param_initializer.cpp

$(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(PreprocessSuffix): model/param_initializer/test_param_initializer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/model_param_initializer_test_param_initializer.cpp$(PreprocessSuffix) model/param_initializer/test_param_initializer.cpp

$(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(ObjectSuffix): model/param_initializer/param_initializer.cpp $(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/model/param_initializer/param_initializer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(DependSuffix): model/param_initializer/param_initializer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(DependSuffix) -MM model/param_initializer/param_initializer.cpp

$(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(PreprocessSuffix): model/param_initializer/param_initializer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/model_param_initializer_param_initializer.cpp$(PreprocessSuffix) model/param_initializer/param_initializer.cpp

$(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(ObjectSuffix): model/param_initializer/test_var_scalr_filler.cpp $(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/new/MetaNN/Tests/OtherTests/model/param_initializer/test_var_scalr_filler.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(DependSuffix): model/param_initializer/test_var_scalr_filler.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(DependSuffix) -MM model/param_initializer/test_var_scalr_filler.cpp

$(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(PreprocessSuffix): model/param_initializer/test_var_scalr_filler.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/model_param_initializer_test_var_scalr_filler.cpp$(PreprocessSuffix) model/param_initializer/test_var_scalr_filler.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


