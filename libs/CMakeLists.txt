if(APPLE)
message("No good Cuda support for Mac")
else(APPLE)
ADD_SUBDIRECTORY( CausalJazz )
ADD_SUBDIRECTORY( PyCausalJazz )
endif(APPLE)
