cmake_minimum_required(VERSION 2.6)

project( CausalJazz )


set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


enable_language(C CXX)


# include cmake scripts
#############################################################################
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )

# default install prefix. can be set with ccmake
set( CMAKE_INSTALL_PREFIX "/usr/local" CACHE STRING "Install prefix." )

# On linux, we define all libs as shared so they can be used in
# generated python import libraries. On windows, we don't need to do that.
# If we need dlls in windows instead of static lib files, 
# we need to set DLL export macro here and will need to go through *ALL*
# classes we want to export... yeesh...
#if (WIN32)
#add_compile_definitions("DllExport=__declspec(dllexport)")
#else(WIN32)
#add_compile_definitions("DllExport=")
#endif(WIN32)

##############################################################################
# cmake options
##############################################################################

set(DEBUGLEVEL "logINFO" CACHE STRING "Define the Debug Level")
add_definitions( -DDEBUGLEVEL=${DEBUGLEVEL})

##############################################################################
# compile flags
##############################################################################
if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra -Wno-unused-parameter -Wno-error=unused-parameter -O0 -pedantic")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals -fPIC")
	set(GLUT_LIBRARY "-lGL -lGLU -lglut")
endif(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")

IF (WIN32 OR APPLE) # For windows packaging, we want to include the glut dll so we need FindGLUT to find it during install
find_package(GLUT REQUIRED)
if (GLUT_FOUND AND APPLE)
add_definitions( -DUSING_APPLE_GLUT=1 ) # this tells TwoDLib/display.cpp to avoid non mac-glut functions.
endif(GLUT_FOUND AND APPLE)
ENDIF (WIN32 OR APPLE)

find_package( Boost COMPONENTS unit_test_framework filesystem program_options thread serialization timer system REQUIRED)

if( Boost_FOUND )
   message(STATUS "Boost found setting up directories")
   message(STATUS "BOOST_ROOT: " ${BOOST_ROOT} )
   message(STATUS "Boost include directories:" ${Boost_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} )
   message(STATUS "Boost library directories:" ${Boost_LIBRARY_DIRS} ${Boost_LIBRARY_DIR_RELEASE} ${Boost_LIBRARY_DIR} )
   include_directories( ${Boost_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} )
   link_directories( ${Boost_LIBRARY_DIRS} ${Boost_LIBRARY_DIR_RELEASE} ${Boost_LIBRARY_DIR} )
else ( Boost_FOUND )
   message("Boost is not found.")
endif( Boost_FOUND )

include_directories( ./libs )
message("========================================Common settings done going on with libs===================================")
ADD_SUBDIRECTORY( libs )
#message("========================================libs set and done continuing with apps====================================")
#ADD_SUBDIRECTORY( apps )
message("========================================compiler flags====================================")
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})

if (NOT SKBUILD) # If miind is being built as a standalone (not a pip wheel)
message("======================================== Please update ENV variables ====================================")
message("")
message("export PATH=$PATH:${CMAKE_INSTALL_PREFIX}/share/miind/python/miind")
message("export PYTHONPATH=$PYTHONPATH:${CMAKE_INSTALL_PREFIX}/share/miind/python/")
message("export OMP_NUM_THREADS=<NUMBER OF AVAILABLE CORES>" ) 
message("")
message("==========================================================================================================")
endif (NOT SKBUILD)

