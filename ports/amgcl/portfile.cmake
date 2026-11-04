vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO ddemidov/amgcl
    REF 1.5.0
    SHA512 c0faa4894dfb797404afa749bde14721be3bf828c6bbf9538eb7caea180bfca9bcf2952a420cce2914ce532203065b5b739188bc000005801a84120bba4e7e33
    HEAD_REF master
)
file(COPY "${SOURCE_PATH}/amgcl" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(COPY "${CMAKE_CURRENT_LIST_DIR}/amgclConfig.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/amgcl")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.md")
