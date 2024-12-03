#!/bin/bash

subjects=("00067" "00068" "00070" "00073" "00074" "00075" "00076" "00078" "00079" "00080" "00084" "00093")
takes=("06" "03" "04" "03" "01" "04" "01" "01" "03" "01" "01" "02")
genders=("m" "f" "m" "f" "f" "m" "m" "m" "f" "m" "m" "f" "m")
for i in ${!subjects[@]}; do
  subject=${subjects[$i]}
  take=${takes[$i]}
  gender=${genders[$i]}

  if [ "$gender" = "m" ]; then
    formatted_gender="male"
  else
    formatted_gender="female"
  fi
  python preprocess_SCustomHumans.py --subject "$subject" --task_id "$take" --gender "$formatted_gender"
done
