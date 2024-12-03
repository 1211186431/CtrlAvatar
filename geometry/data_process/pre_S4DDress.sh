#!/bin/bash

subjects=("00122" "00123" "00127" "00129" "00134" "00135" "00136" "00137")
genders=("m" "f" "m" "f" "m" "m" "f" "f")
for i in ${!subjects[@]}; do
  subject=${subjects[$i]}
  take=${takes[$i]}
  gender=${genders[$i]}

  if [ "$gender" = "m" ]; then
    formatted_gender="male"
  else
    formatted_gender="female"
  fi

  python preprocess_4DDress.py --subj "$subject" --outfit Inner --gender "$formatted_gender"
done
