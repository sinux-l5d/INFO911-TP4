cmake_minimum_required(VERSION 3.5)
project( DisplayImage )
set (CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
find_package( OpenCV REQUIRED COMPONENTS highgui)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

SET( MAIN_EXEC
     prog
     # vous pouvez ajouter d'autres programmes ici
     )

FOREACH(FILE ${MAIN_EXEC})
            add_executable( ${FILE} ${FILE}.cpp )
            target_link_libraries( ${FILE} ${OpenCV_LIBS} )
ENDFOREACH(FILE)