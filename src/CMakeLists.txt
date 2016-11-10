include_directories(${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR})

set(EX ex0-vecadd)
add_executable(${EX} ${EX}.cpp)
target_link_libraries(${EX} ${OpenCL_LIBRARIES})

set(EX ex1-matmat)
add_executable(${EX} ${EX}.cpp)
target_link_libraries(${EX} ${OpenCL_LIBRARIES})

set(EX ex2-pi)
add_executable(${EX} ${EX}.cpp)
target_link_libraries(${EX} ${OpenCL_LIBRARIES})
