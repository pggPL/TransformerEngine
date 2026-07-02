cmake_minimum_required(VERSION 3.15...3.27)
project(nanobind_stable_poc LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_PREFIX)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
find_package(Torch REQUIRED)

if(DEFINED ENV{STABLE_TORCH_COMPILE})
    set(STABLE_TORCH_COMPILE "$ENV{STABLE_TORCH_COMPILE}")
else()
    set(STABLE_TORCH_COMPILE 0)
endif()

nanobind_add_module(my_ext extension.cpp)
target_link_libraries(my_ext PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(my_ext PRIVATE "${TORCH_INCLUDE_DIRS}")
set_property(TARGET my_ext PROPERTY CXX_STANDARD 17)

# Both modes need torch_python here: the classic path uses THPVariable_*, and
# the stable path now uses the from_pyobject/to_pyobject shims, which also live
# in libtorch_python. (This is the tradeoff of the python-interop shims -- the
# extension is no longer libtorch-only.)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch, os; print(os.path.join(torch.__path__[0], 'lib'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_LIB_DIR)
find_library(TORCH_PYTHON_LIB torch_python PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH)
target_link_libraries(my_ext PRIVATE "${TORCH_PYTHON_LIB}")

if(STABLE_TORCH_COMPILE)
    target_compile_definitions(my_ext PRIVATE STABLE_TORCH_COMPILE=1)
    message(STATUS "Building with PyTorch STABLE ABI (+ torch_python for the python-interop shims)")
else()
    target_compile_definitions(my_ext PRIVATE STABLE_TORCH_COMPILE=0)
    message(STATUS "Building with PyTorch non-stable ABI (torch_python: ${TORCH_PYTHON_LIB})")
endif()
