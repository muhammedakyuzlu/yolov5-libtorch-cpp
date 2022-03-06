cmake_minimum_required(VERSION 3.5.1)
project(libtorch-yolov5)




set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


find_package(OpenCV 4 REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "version: ${OpenCV_VERSION}")
    message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()


set(Torch_DIR ./libtorch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCE_FILES src/*.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries (
        ${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)
