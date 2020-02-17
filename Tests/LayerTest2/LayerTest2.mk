##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=LayerTest2
ConfigurationName      :=Debug
WorkspacePath          :=/home/MetaNN
ProjectPath            :=/home/MetaNN/Tests/LayerTest2
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=root
Date                   :=02/17/20
CodeLitePath           :=/root/.codelite
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
Preprocessors          :=$(PreprocessorSwitch)METANN_CHECKSHAPE 
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E 
ObjectsFileList        :="LayerTest2.txt"
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
Objects0=$(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/elementary_test_permute_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/elementary_test_add_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/elementary_test_relu_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/elementary_test_abs_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(ObjectSuffix) $(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(ObjectSuffix) 



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
$(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(ObjectSuffix): elementary/test_interpolate_layer.cpp $(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/elementary/test_interpolate_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(DependSuffix): elementary/test_interpolate_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(DependSuffix) -MM elementary/test_interpolate_layer.cpp

$(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(PreprocessSuffix): elementary/test_interpolate_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/elementary_test_interpolate_layer.cpp$(PreprocessSuffix) elementary/test_interpolate_layer.cpp

$(IntermediateDirectory)/elementary_test_permute_layer.cpp$(ObjectSuffix): elementary/test_permute_layer.cpp $(IntermediateDirectory)/elementary_test_permute_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/elementary/test_permute_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/elementary_test_permute_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/elementary_test_permute_layer.cpp$(DependSuffix): elementary/test_permute_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/elementary_test_permute_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/elementary_test_permute_layer.cpp$(DependSuffix) -MM elementary/test_permute_layer.cpp

$(IntermediateDirectory)/elementary_test_permute_layer.cpp$(PreprocessSuffix): elementary/test_permute_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/elementary_test_permute_layer.cpp$(PreprocessSuffix) elementary/test_permute_layer.cpp

$(IntermediateDirectory)/main.cpp$(ObjectSuffix): main.cpp $(IntermediateDirectory)/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM main.cpp

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) main.cpp

$(IntermediateDirectory)/elementary_test_add_layer.cpp$(ObjectSuffix): elementary/test_add_layer.cpp $(IntermediateDirectory)/elementary_test_add_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/elementary/test_add_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/elementary_test_add_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/elementary_test_add_layer.cpp$(DependSuffix): elementary/test_add_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/elementary_test_add_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/elementary_test_add_layer.cpp$(DependSuffix) -MM elementary/test_add_layer.cpp

$(IntermediateDirectory)/elementary_test_add_layer.cpp$(PreprocessSuffix): elementary/test_add_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/elementary_test_add_layer.cpp$(PreprocessSuffix) elementary/test_add_layer.cpp

$(IntermediateDirectory)/elementary_test_relu_layer.cpp$(ObjectSuffix): elementary/test_relu_layer.cpp $(IntermediateDirectory)/elementary_test_relu_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/elementary/test_relu_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/elementary_test_relu_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/elementary_test_relu_layer.cpp$(DependSuffix): elementary/test_relu_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/elementary_test_relu_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/elementary_test_relu_layer.cpp$(DependSuffix) -MM elementary/test_relu_layer.cpp

$(IntermediateDirectory)/elementary_test_relu_layer.cpp$(PreprocessSuffix): elementary/test_relu_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/elementary_test_relu_layer.cpp$(PreprocessSuffix) elementary/test_relu_layer.cpp

$(IntermediateDirectory)/elementary_test_abs_layer.cpp$(ObjectSuffix): elementary/test_abs_layer.cpp $(IntermediateDirectory)/elementary_test_abs_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/elementary/test_abs_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/elementary_test_abs_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/elementary_test_abs_layer.cpp$(DependSuffix): elementary/test_abs_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/elementary_test_abs_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/elementary_test_abs_layer.cpp$(DependSuffix) -MM elementary/test_abs_layer.cpp

$(IntermediateDirectory)/elementary_test_abs_layer.cpp$(PreprocessSuffix): elementary/test_abs_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/elementary_test_abs_layer.cpp$(PreprocessSuffix) elementary/test_abs_layer.cpp

$(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(ObjectSuffix): elementary/test_sigmoid_layer.cpp $(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/elementary/test_sigmoid_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(DependSuffix): elementary/test_sigmoid_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(DependSuffix) -MM elementary/test_sigmoid_layer.cpp

$(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(PreprocessSuffix): elementary/test_sigmoid_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/elementary_test_sigmoid_layer.cpp$(PreprocessSuffix) elementary/test_sigmoid_layer.cpp

$(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(ObjectSuffix): elementary/test_multiply_layer.cpp $(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/elementary/test_multiply_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(DependSuffix): elementary/test_multiply_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(DependSuffix) -MM elementary/test_multiply_layer.cpp

$(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(PreprocessSuffix): elementary/test_multiply_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/elementary_test_multiply_layer.cpp$(PreprocessSuffix) elementary/test_multiply_layer.cpp

$(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(ObjectSuffix): elementary/test_tanh_layer.cpp $(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/MetaNN/Tests/LayerTest2/elementary/test_tanh_layer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(DependSuffix): elementary/test_tanh_layer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(DependSuffix) -MM elementary/test_tanh_layer.cpp

$(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(PreprocessSuffix): elementary/test_tanh_layer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/elementary_test_tanh_layer.cpp$(PreprocessSuffix) elementary/test_tanh_layer.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


