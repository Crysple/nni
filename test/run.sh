#!/bin/bash
# Integration Test
cd naive && python3 run.sh
# Built-in Tuner Test 
cd ../test_builtin_tuner/ && python3 run.sh