include(../init.cmake)
add_subdirectory(pybind11)

pybind11_add_module( gpAmreX gp.cpp )
target_link_libraries(gpAmreX PUBLIC gpAmrex)
setup(gpAmreX)