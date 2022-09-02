cmake_minimum_required(VERSION 3.14)
project( amrGp C CXX Fortran )
set (CMAKE_CXX_STANDARD 17)


set(AMReX_SPACEDIM 2 )


if (AMReX_SPACEDIM EQUAL 1)
   return()
endif ()


# compilaer options 

set(AMReX_DIR "" CACHE STRING "Root directory of amrex implementation")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Compilation mode")
set_property(CACHE CMAKE_BUILD_TYPE  PROPERTY STRINGS Release Debug)
mark_as_advanced( CMAKE_INSTALL_PREFIX)


set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_CXX_FLAGS_RELEASE "")


add_compile_options(
  -Wfatal-errors
       $<$<CONFIG:RELEASE>:-O3>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-g>
)

add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
)


# List of source files
set(_sources AdvancePhiAllLevels.cpp AmrCoreAdv.cpp  AmrCoreAdv.H bc_fill.H)
list(APPEND _sources Src_K/Adv_K.H  Src_K/compute_flux_${AMReX_SPACEDIM}D_K.H  Src_K/slope_K.H)
list(APPEND _sources Exec/Prob.H)


function (addLIBPATH target)
    string(REPLACE ":" " -L"  LIB_FLAGS "$ENV{LIBPATH}" )
    if ( NOT (LIB_FLAGS STREQUAL "") )
        set_target_properties(${target} PROPERTIES LINK_FLAGS "-L ${LIB_FLAGS}" )
    endif()

endfunction()

function(setup target)
    target_include_directories(${target} PUBLIC /home/luca/source/GP5/GP5/amrex_gp/diffusion_amr_level/src/Source ${AMReX_DIR}/include   )
    target_link_directories(${target} PUBLIC ${AMReX_DIR}/lib   )
    target_link_libraries(${target} PUBLIC amrex gfortran hdf5 )
    addLIBPATH(${target})
endfunction()


add_library( gpAmrex gpLevel.cpp operators.cpp )
setup(gpAmrex)

add_executable(levelTest main.cpp )
setup(levelTest)
target_link_libraries(levelTest PUBLIC gpAmrex)


add_executable(amrTest mainAdvect.cpp ${_sources} )
setup(amrTest)
target_link_libraries(amrTest PUBLIC gpAmrex)


unset( _sources )
unset( _input_files   )


