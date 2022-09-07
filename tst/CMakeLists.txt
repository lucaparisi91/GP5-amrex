
cmake_minimum_required(VERSION 3.20)
include(../init.cmake)

# include gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols INSTALL_GTEST
    FETCHCONTENT_SOURCE_DIR_GOOGLETEST FETCHCONTENT_BASE_DIR FETCHCONTENT_QUIET FETCHCONTENT_FULLY_DISCONNECTED FETCHCONTENT_UPDATES_DISCONNECTED FETCHCONTENT_UPDATES_DISCONNECTED_GOOGLETEST
)


include(GoogleTest)

# List of source files
set( SOURCES main_test.cpp levels_test.cpp )



add_executable(tst ${SOURCES} )
setup(tst)
target_link_libraries(tst PUBLIC gpAmrex )
target_link_libraries(tst PRIVATE gtest)


unset( _sources )
unset( _input_files   )


