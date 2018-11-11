##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=GeneralTest2
ConfigurationName      :=Debug
WorkspacePath          :=/home/liwei/MetaNN
ProjectPath            :=/home/liwei/MetaNN/GeneralTest2
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=liwei
Date                   :=11/11/18
CodeLitePath           :=/home/liwei/.codelite
LinkerName             :=/usr/bin/g++
SharedObjectLinkerName :=/usr/bin/g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
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
AR       := /usr/bin/ar rcu
CXX      := /usr/bin/g++
CC       := /usr/bin/gcc
CXXFLAGS :=  -g -O0 -Wall -std=c++17 $(Preprocessors)
CFLAGS   :=  -g -O0 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix) $(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix) 



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
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/GeneralTest2/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM main.cpp

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) main.cpp

$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix): data/cardinal/scalar/test_scalar.cpp $(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/GeneralTest2/data/cardinal/scalar/test_scalar.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(DependSuffix): data/cardinal/scalar/test_scalar.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(DependSuffix) -MM data/cardinal/scalar/test_scalar.cpp

$(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(PreprocessSuffix): data/cardinal/scalar/test_scalar.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_scalar_test_scalar.cpp$(PreprocessSuffix) data/cardinal/scalar/test_scalar.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix): data/cardinal/matrix/test_matrix.cpp $(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/GeneralTest2/data/cardinal/matrix/test_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(DependSuffix): data/cardinal/matrix/test_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(DependSuffix) -MM data/cardinal/matrix/test_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(PreprocessSuffix): data/cardinal/matrix/test_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_matrix_test_matrix.cpp$(PreprocessSuffix) data/cardinal/matrix/test_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix): data/cardinal/matrix/test_zero_matrix.cpp $(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/GeneralTest2/data/cardinal/matrix/test_zero_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(DependSuffix): data/cardinal/matrix/test_zero_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(DependSuffix) -MM data/cardinal/matrix/test_zero_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(PreprocessSuffix): data/cardinal/matrix/test_zero_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_matrix_test_zero_matrix.cpp$(PreprocessSuffix) data/cardinal/matrix/test_zero_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix): data/cardinal/matrix/test_trival_matrix.cpp $(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/liwei/MetaNN/GeneralTest2/data/cardinal/matrix/test_trival_matrix.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(DependSuffix): data/cardinal/matrix/test_trival_matrix.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(DependSuffix) -MM data/cardinal/matrix/test_trival_matrix.cpp

$(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(PreprocessSuffix): data/cardinal/matrix/test_trival_matrix.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/data_cardinal_matrix_test_trival_matrix.cpp$(PreprocessSuffix) data/cardinal/matrix/test_trival_matrix.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


