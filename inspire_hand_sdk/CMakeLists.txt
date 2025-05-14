cmake_minimum_required(VERSION 3.10)
project(h1_inspire_service)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  message("Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
else()
  message("Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG -fPIC")
endif()

find_package(Boost COMPONENTS program_options REQUIRED)


set(CMAKE_PREFIX_PATH /opt/unitree_robotics)
find_package(unitree_sdk2 REQUIRED)

# 输出相关变量
if(unitree_sdk2_FOUND)
    message(STATUS "Found unitree_sdk2.")
    get_target_property(unitree_sdk2_includes unitree_sdk2 INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "unitree_sdk2 include: ${unitree_sdk2_includes}")
    get_target_property(ddscxx_includes ddscxx INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "ddscxx  include: ${ddscxx_includes}")

    get_target_property(ddsc_includes ddsc INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "ddsc  include: ${ddsc_includes}")

else()
    message(WARNING "unitree_sdk2 not found.")
endif()

file(GLOB_RECURSE SRC_LIST
    "hand_idl/*.cpp"
)
# include_directories(${unitree_sdk2_includes} ${ddscxx_includes} ./library/unitree_sdk2/include)
link_directories(/usr/local/lib)

message(STATUS "SRC_LIST: ${SRC_LIST}")
# ${SRC_LIST}
add_executable(hand_dds cpp_example/Hand_dds_example.cpp ${SRC_LIST})
target_include_directories(hand_dds PRIVATE ${unitree_sdk2_includes} ${ddscxx_includes} ${ddsc_includes} hand_idl )

target_link_libraries(hand_dds unitree_sdk2 rt pthread ddscxx ddsc)
target_link_libraries(hand_dds -pthread)




