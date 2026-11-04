# Minimal interface config for the header-only AMGCL overlay port (vcpkg layout).
if(NOT TARGET amgcl::amgcl)
    add_library(amgcl::amgcl INTERFACE IMPORTED)
    set_target_properties(amgcl::amgcl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include")
endif()
