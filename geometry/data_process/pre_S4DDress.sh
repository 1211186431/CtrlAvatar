#!/bin/bash

# 创建一个数组存储Subject, Take和Gender
subjects=("00122" "00123" "00127" "00129" "00134" "00135" "00136" "00137")
genders=("m" "f" "m" "f" "m" "m" "f" "f")
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
  python preprocess_4DDress.py --subj "$subject" --outfit Inner --gender "$formatted_gender"
done
