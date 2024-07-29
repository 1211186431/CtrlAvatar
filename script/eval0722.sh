data_list=("00016" "00017" "00018" "00019" "00020" "00021" "00024" "00025" "00027" "00028" "00034" "00035" "00036" "00039" "00041" "00058" "00085" "00086" "00087" "00088")
for item in "${data_list[@]}"; do
   echo "eval oursfit ${item} subject"
   python ./eval0722.py --subject ${item} --data_path /home/ps/dy/OpenAvatar/outputs/test/${item}/mesh_test --is_gt False --method Oursfit --out_dir /home/ps/dy/eval_oursfit
done

# data_list=("00016" "00017" "00018" "00019" "00020" "00021" "00024" "00025" "00027" "00028" "00034" "00035" "00039" "00041" "00085" "00086" "00087" "00088")
# for item in "${data_list[@]}"; do
#    echo "eval ${item} subject"
#    python ./eval0722.py --subject ${item} --data_path /home/ps/dy/HaveFun_test/${item}_havefun/meshes_test_obj --is_gt False --method HaveFun --out_dir /home/ps/dy/eval_havefun
# done