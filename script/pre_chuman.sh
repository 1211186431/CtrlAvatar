#!/bin/bash

# 创建一个数组存储Subject, Take和Gender
subjects=("00067" "00068" "00070" "00073" "00074" "00075" "00076" "00078" "00079" "00080" "00084" "00093")
takes=("06" "03" "04" "03" "01" "04" "01" "01" "03" "01" "01" "02")
genders=("m" "f" "m" "f" "f" "m" "m" "m" "f" "m" "m" "f" "m")
cd /home/ps/dy/OpenAvatar/data_process
# 循环遍历数组并运行Python脚本
for i in ${!subjects[@]}; do
  subject=${subjects[$i]}
  take=${takes[$i]}
  gender=${genders[$i]}

  # 转换gender为male或female
  if [ "$gender" = "m" ]; then
    formatted_gender="male"
  else
    formatted_gender="female"
  fi

  # 运行Python脚本
  python process_CustomHumans.py --subject "$subject" --task_id "$take" --gender "$formatted_gender"
done
