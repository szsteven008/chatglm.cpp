cmake_minimum_required(VERSION 3.12)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Set correct Homebrew install folder for Apple Silicon and Intel Macs
    if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
        set(HOMEBREW_DEFAULT_PREFIX "/opt/homebrew")
    else()
        set(HOMEBREW_DEFAULT_PREFIX "/usr/local")
    endif()

    link_directories("${HOMEBREW_DEFAULT_PREFIX}/lib")
    include_directories("${HOMEBREW_DEFAULT_PREFIX}/include")
endif()

find_package(Threads REQUIRED)

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
find_program(_PROTOBUF_PROTOC protoc)
set(_GRPC_GRPCPP gRPC::grpc++)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

# Proto file
get_filename_component(hw_proto "./backend.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/backend.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/backend.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/backend.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/backend.grpc.pb.h")
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# hw_grpc_proto
add_library(hw_grpc_proto
  ${hw_grpc_srcs}
  ${hw_grpc_hdrs}
  ${hw_proto_srcs}
  ${hw_proto_hdrs})
target_link_libraries(hw_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

find_package(absl)

add_executable(server server.cpp service.cpp)
target_link_libraries(server absl::flags absl::flags_parse absl::strings chatglm hw_grpc_proto)