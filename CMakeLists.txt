cmake_minimum_required(VERSION 3.10)
project(vectordb)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the built-in Threads package
find_package(Threads REQUIRED)

# Tell CMake to find our 'lib' subdirectory
# This will find and execute lib/hnsw/CMakeLists.txt
add_subdirectory(lib/hnsw)

# --- Main 'vectordb' CLI Executable ---
add_executable(
    vectordb
    src/main.cpp
    src/vectordb.cpp
)

target_include_directories(vectordb
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include     # For vectordb.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/json    # For json.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/hnsw    # For hnsw.h
)

target_link_libraries(
    vectordb 
    hnsw_lib
    Threads::Threads
)

# --- NEW: 'vectordb_test' Executable ---
# We add this new target to build our test file.
# It needs to be compiled with vectordb.cpp and linked
# to the same libraries.
add_executable(
    vectordb_test
    src/test.cpp
    src/vectordb.cpp # It also needs the DB implementation
)

# Tell the test executable where to find headers
target_include_directories(vectordb_test
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/json
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/hnsw
)

# Link the test executable to the same libraries
target_link_libraries(
    vectordb_test
    hnsw_lib
    Threads::Threads
)

