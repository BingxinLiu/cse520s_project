# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lbx/cse520s/woad

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lbx/cse520s/woad/build

# Include any dependencies generated for this target.
include CMakeFiles/rec_photo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rec_photo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rec_photo.dir/flags.make

CMakeFiles/rec_photo.dir/rec_photo.cpp.o: CMakeFiles/rec_photo.dir/flags.make
CMakeFiles/rec_photo.dir/rec_photo.cpp.o: ../rec_photo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lbx/cse520s/woad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rec_photo.dir/rec_photo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rec_photo.dir/rec_photo.cpp.o -c /home/lbx/cse520s/woad/rec_photo.cpp

CMakeFiles/rec_photo.dir/rec_photo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rec_photo.dir/rec_photo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lbx/cse520s/woad/rec_photo.cpp > CMakeFiles/rec_photo.dir/rec_photo.cpp.i

CMakeFiles/rec_photo.dir/rec_photo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rec_photo.dir/rec_photo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lbx/cse520s/woad/rec_photo.cpp -o CMakeFiles/rec_photo.dir/rec_photo.cpp.s

CMakeFiles/rec_photo.dir/rec_photo.cpp.o.requires:

.PHONY : CMakeFiles/rec_photo.dir/rec_photo.cpp.o.requires

CMakeFiles/rec_photo.dir/rec_photo.cpp.o.provides: CMakeFiles/rec_photo.dir/rec_photo.cpp.o.requires
	$(MAKE) -f CMakeFiles/rec_photo.dir/build.make CMakeFiles/rec_photo.dir/rec_photo.cpp.o.provides.build
.PHONY : CMakeFiles/rec_photo.dir/rec_photo.cpp.o.provides

CMakeFiles/rec_photo.dir/rec_photo.cpp.o.provides.build: CMakeFiles/rec_photo.dir/rec_photo.cpp.o


# Object files for target rec_photo
rec_photo_OBJECTS = \
"CMakeFiles/rec_photo.dir/rec_photo.cpp.o"

# External object files for target rec_photo
rec_photo_EXTERNAL_OBJECTS =

rec_photo: CMakeFiles/rec_photo.dir/rec_photo.cpp.o
rec_photo: CMakeFiles/rec_photo.dir/build.make
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
rec_photo: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
rec_photo: CMakeFiles/rec_photo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lbx/cse520s/woad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rec_photo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rec_photo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rec_photo.dir/build: rec_photo

.PHONY : CMakeFiles/rec_photo.dir/build

CMakeFiles/rec_photo.dir/requires: CMakeFiles/rec_photo.dir/rec_photo.cpp.o.requires

.PHONY : CMakeFiles/rec_photo.dir/requires

CMakeFiles/rec_photo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rec_photo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rec_photo.dir/clean

CMakeFiles/rec_photo.dir/depend:
	cd /home/lbx/cse520s/woad/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lbx/cse520s/woad /home/lbx/cse520s/woad /home/lbx/cse520s/woad/build /home/lbx/cse520s/woad/build /home/lbx/cse520s/woad/build/CMakeFiles/rec_photo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rec_photo.dir/depend

