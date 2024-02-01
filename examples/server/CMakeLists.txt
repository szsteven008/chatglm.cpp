cmake_minimum_required(VERSION 3.12)

find_package(absl)

add_executable(server server.cpp)
target_link_libraries(server absl::flags absl::flags_parse absl::strings chatglm)