#!/bin/bash

run_multiple_times() {
    for i in {1..5}; do
        srun ./parellel "$1" "$2"
    done
}

run_multiple_times test_images/720x480.png 592x480.png
run_multiple_times test_images/1024x768.png 896x768.png
run_multiple_times test_images/1920x1200.png 1892x1200.png
run_multiple_times test_images/3840x2160.png 3712x2160.png
run_multiple_times test_images/7680x4320.png 7552x4320.png