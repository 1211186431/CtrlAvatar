# data_list=("00016" "00017" "00018" "00019" "00020" "00021" "00024" "00025" "00027" "00028" "00034" "00035" "00036" "00039" "00041" "00058" "00085" "00086" "00087" "00088")
# for item in "${data_list[@]}"; do
#    echo "eval oursfit ${item} subject"
#    python ./save_eval_data.py --subject ${item} --data_path /home/ps/dy/OpenAvatar/outputs/test/${item}/mesh_test --is_gt False --method Oursfit --out_dir /home/ps/dy/eval_oursfit
# done

# data_list=("00016" "00017" "00018" "00019" "00020" "00021" "00024" "00025" "00027" "00028" "00034" "00035" "00039" "00041" "00085" "00086" "00087" "00088")
# for item in "${data_list[@]}"; do
#    echo "eval ${item} subject"
#    python ./save_eval_data.py --subject ${item} --data_path /home/ps/dy/HaveFun_test/${item}_havefun/meshes_test_obj --is_gt False --method HaveFun --out_dir /home/ps/dy/eval_havefun
# done

# data_list=("00067" "00068" "00070" "00074" "00076" "00078" "00079" "00080" "00084" "00093")
# for item in "${data_list[@]}"; do
#    echo "eval oursfit ${item} subject"
#    python ./save_eval_data.py --subject ${item} --data_path /home/ps/dy/OpenAvatar/outputs/test/${item}/mesh_test --is_gt False --method Oursfit --out_dir /home/ps/dy/eval0807/eval_oursfit
# done


# for item in "${data_list[@]}"; do
#    echo "eval GT ${item} subject"
#    python ./save_eval_data.py --subject ${item} --data_path /home/ps/dy/chuman --is_gt True --method gt --out_dir /home/ps/dy/eval0807/eval_oursfit
# done
