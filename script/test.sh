#data_list=("00016" "00017" "00018" "00019" "00020" "00021" "00024" "00025" "00027" "00028" "00034" "00035" "00036" "00039" "00041" "00058" "00085" "00086" "00087" "00088")
# for item in "${data_list[@]}"; do
#    echo "pre ${item} subject"
#    python ./main.py --config ./config/config${item}.yaml --mode test
# done

data_list=("00067" "00068" "00070" "00074" "00076" "00078" "00079" "00080" "00084" "00093")
for item in "${data_list[@]}"; do
   echo "pre ${item} subject"
   python ./main.py --config config/SXHumans.yaml --mode test --subject ${item}
done