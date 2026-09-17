#!/usr/bin/env bash
# msmodelslim 源码树以 editable 安装时，CLI 会从"包内"找 config/lab_practice/lab_calib，
# 而仓库把它们放在根目录 —— 这里补软链（幂等）。
set -euo pipefail
R=/home/user/projects/dsv41/src/msmodelslim
[ -f $R/msmodelslim/config/config.ini ] || { mkdir -p $R/msmodelslim/config; cp -f $R/config/config.ini $R/msmodelslim/config/config.ini; }
ln -sfn ../config $R/msmodelslim/config_repo
[ -e $R/msmodelslim/lab_practice ] || ln -s ../lab_practice $R/msmodelslim/lab_practice
[ -e $R/msmodelslim/lab_calib ] || ln -s ../lab_calib $R/msmodelslim/lab_calib
ls -l $R/msmodelslim/lab_practice $R/msmodelslim/lab_calib $R/msmodelslim/config/config.ini
