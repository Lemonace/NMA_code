#!/usr/bin/env bash

LOG_PATH="./"
if [[ ! -d ${LOG_PATH} ]]; then
        mkdir ${LOG_PATH}
fi

project_path=$(cd `dirname $0`; pwd)
project_name="${project_path##*/}"
begin_time=$(date "+%Y-%m-%d %H:%M:%S")
author="gmz"

LOG_FILENAME="log"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --author ${author} --project ${project_name} --begintime ${begin_time} > ${LOG_PATH}${LOG_FILENAME} 2>&1 &
