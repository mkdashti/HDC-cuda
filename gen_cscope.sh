#!/bin/bash

find . -name "*.cu" -o -name "*.c" -o -name "*.cpp" -o -name "*.h"  > cscope.files
cscope -b -icscope.files -q -u
#ctags -R --c++-kinds=+p --fields=+iaS --extra=+q 
ctags -R .
