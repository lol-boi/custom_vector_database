# This file compiles the HNSW source files into a static library

# Find the built-in Threads package
# This creates the 'Threads::Threads' target
find_package(Threads REQUIRED)

# Create a static library named 'hnsw_lib'
add_library(
    hnsw_lib
    STATIC
    hnsw.cpp  # This is the only .cpp file
)

# Specify that this library needs std::thread
target_link_libraries(hnsw_lib PUBLIC Threads::Threads)

# Tell any target that links this library (our main app)
# that it should also include the current directory ('.')
# to find the .h file (hnsw.h)
target_include_directories(hnsw_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)



