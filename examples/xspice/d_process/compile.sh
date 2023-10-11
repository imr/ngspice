#!/bin/bash
set -v
gcc -Wall -Wpedantic -g -o prog1in4out prog1in4out.c
gcc -Wall -Wpedantic -g -o prog4in1out prog4in1out.c
gcc -Wall -Wpedantic -g -o graycode graycode.c
