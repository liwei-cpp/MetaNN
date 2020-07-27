###############################################################################
#
# MODULE      : CMakeLists.txt
# DESCRIPTION : CMake file for MetaNN
# COPYRIGHT   : (C) 2020-2020  PikachuHy
#



### --------------------------------------------------------------------
### Basic settings (project independent)
### --------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

### --------------------------------------------------------------------
### Project name
### --------------------------------------------------------------------

project(MetaNN CXX)

### --------------------------------------------------------------------
### Testing
### --------------------------------------------------------------------

set(Tests_DIR ${CMAKE_SOURCE_DIR}/Tests)
file(GLOB Tests_DIRs RELATIVE ${Tests_DIR} ${Tests_DIR}/*)
foreach(tests_subdir ${Tests_DIRs})
    if(IS_DIRECTORY ${Tests_DIR}/${tests_subdir})
        message("Add test ${tests_subdir}")
        file(GLOB_RECURSE ${tests_subdir}_SRCS ${Tests_DIR}/${tests_subdir}/*.cpp)
        if(${tests_subdir}_SRCS)
            add_executable(${tests_subdir} ${${tests_subdir}_SRCS})
            target_include_directories(${tests_subdir} PRIVATE Tests Tests/${tests_subdir})
        else()
            message("Skip ${tests_subdir}")
        endif()
    endif()
endforeach()