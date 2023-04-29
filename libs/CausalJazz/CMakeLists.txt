
project( causaljazz )

if (WIN32) # When cmake generates the VS files, it needs to know this is a cuda project
enable_language(CUDA)
endif (WIN32)

find_package(CUDA  REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS} )

link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )

set( LIB_TYPE STATIC )

include_directories( ${ROOT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )


#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)

list(APPEND TWOD_SRC
NdGrid.cpp
MassSimulation.cu
MassPopulation.cu
CudaEuler.cu
display.cpp
)
list(APPEND TWOD_HEADERS
NdGrid.hpp
MassSimulation.cuh
MassPopulation.cuh
CudaEuler.cuh
display.hpp
)

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${gsl_LINK_FLAGS}
	gsl
  ${GLUT_LIBRARY}
)

add_library( causaljazz ${LIB_TYPE} ${TWOD_SRC} ${TWOD_HEADERS})

target_link_libraries( causaljazz ${LIBLIST})

install( FILES ${TWOD_HEADERS}
    DESTINATION include/causaljazz )

install( TARGETS causaljazz
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
