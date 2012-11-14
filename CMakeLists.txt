#############################################################################
# Verdione - Technology for mixed reality and live arts
#
# File:     CMakeLists.txt
# Author:   Christian Tryti
# Contents: Depth Estimation on GPU Library
#
#
# Verdione software is released under the terms of the GNU Lesser
# General Public License Version 2 (LGPLv2). See file COPYING.LESSER
# for details.
#############################################################################

project(LIBDEPTH)

set(target_name "libdepth")
set(msg "${target_name} disabled")

# depends on; don't build when unavailable
requires(HAVE_NORTHLIGHT ${msg})
requires(HAVE_OPENCV ${msg})
requires(HAVE_OPENCL ${msg})

MESSAGE("-- Adding cotryti::${target_name}")

include_directories(${NORTHLIGHT_INCLUDES})

link_directories(${NORTHLIGHT_LIBDIRS})

# CFlags
add_definitions(${NORTHLIGHT_CFLAGS})

install_extra_data("data")

add_library(${target_name}
  depthestimation.cpp
  depthestimation_opencl.cpp
  oclUtils.cpp
  )

# libraries to link the target against (a list of CMake variables)
target_link_libraries(${target_name} NorthLight opencv_gpu)
