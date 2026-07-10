# ESP-DL 官方工具要求模型按 16 字节对齐嵌入 rodata。
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
else()
    message(FATAL_ERROR "esp-dl component was not resolved")
endif()

set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

idf_component_register(
    SRCS "app_main.cpp"
    REQUIRES esp-dl
)

target_add_aligned_binary_data(
    ${COMPONENT_LIB}
    "models/model.espdl"
    BINARY
)
