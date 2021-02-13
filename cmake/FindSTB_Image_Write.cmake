# Additional modules
include(FindPackageHandleStandardArgs)

if (WIN32)
	# Find include files
	find_path(
		STB_INCLUDE_DIR
		NAMES stb/stb_image_write.h
		PATHS
		$ENV{PROGRAMFILES}/include
		${GLM_ROOT_DIR}/include
		DOC "The directory where stb/stb_image_write.h resides")
else()
	# Find include files
	find_path(
		STB_INCLUDE_DIR
		NAMES stb/stb_image_write.h
		PATHS
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		${GLM_ROOT_DIR}/include
		DOC "The directory where stb/stb_image_write.h resides")
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(STB DEFAULT_MSG STB_INCLUDE_DIR)

# Define STB_INCLUDE_DIRS
if (STB_FOUND)
	set(STB_INCLUDE_DIRS ${STB_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(STB_INCLUDE_DIR)