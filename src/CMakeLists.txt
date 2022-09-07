
cmake_minimum_required(VERSION 3.20)
include(../init.cmake)
# List of source files
set(_sources AdvancePhiAllLevels.cpp AmrCoreAdv.cpp  AmrCoreAdv.H  )



add_library( gpAmrex gpLevel.cpp operators.cpp timeStepper.cpp )
setup(gpAmrex)

add_executable(levelTest main.cpp )
setup(levelTest)
target_link_libraries(levelTest PUBLIC gpAmrex)


add_executable(amrTest mainAdvect.cpp ${_sources} )
setup(amrTest)
target_link_libraries(amrTest PUBLIC gpAmrex)


unset( _sources )
unset( _input_files   )


