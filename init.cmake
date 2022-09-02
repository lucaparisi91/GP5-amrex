set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(AMReX_SPACEDIM 2 )

if (AMReX_SPACEDIM EQUAL 1)
   return()
endif ()

# compiler options 

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