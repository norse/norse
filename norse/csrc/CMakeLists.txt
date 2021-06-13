add_library(norse SHARED op.cpp)

target_compile_features(norse PRIVATE cxx_std_17)
target_include_directories(norse PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(norse PRIVATE include)
target_link_libraries(norse ${Python3_LIBRARIES})
target_link_libraries(norse ${TORCH_LIBRARIES})

# Testing
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable( lif_test lif_test.cpp)
target_link_libraries(
  lif_test
  gtest_main
  ${TORCH_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(lif_test)
