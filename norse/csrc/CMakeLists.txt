add_library(norse SHARED op.cpp)

target_compile_features(norse PRIVATE cxx_std_17)
target_include_directories(norse PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(norse PRIVATE include)
target_link_libraries(norse ${Python3_LIBRARIES})
target_link_libraries(norse ${TORCH_LIBRARIES})

# Testing
if (CMAKE_BUILD_TYPE STREQUAL Debug)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)
  set(SOURCES ${SOURCES})
  add_library(norse_lib STATIC ${SOURCES})
  target_include_directories(norse_lib PRIVATE ${Python3_INCLUDE_DIRS})
  target_include_directories(norse_lib PRIVATE include)
  target_link_libraries(norse_lib ${Python3_LIBRARIES})
  target_link_libraries(norse_lib ${TORCH_LIBRARIES})
  
  add_executable(lif_test lif_test.cpp)

  target_link_libraries(lif_test ${Python3_LIBRARIES})
  target_link_libraries(lif_test gtest gtest_main)
  target_link_libraries(lif_test norse_lib)

  include(GoogleTest)
  gtest_discover_tests(lif_test)

endif()