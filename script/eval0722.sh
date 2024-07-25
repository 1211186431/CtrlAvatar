# data_list=("00016" "00017" "00018" "00019" "00020" "00021" "00024" "00025" "00027" "00028" "00034" "00035" "00036" "00039" "00041" "00058" "00085" "00086" "00087" "00088")
# for item in "${data_list[@]}"; do
#    echo "eval ${item} subject"
#    python eval0722.py --subject ${item} --data_path /home/ps/dy/X-Avatar/outputs/XHumans_smplx/${item}_delta/meshes_test --is_gt False --method Ours --out_dir /home/ps/dy/eval_ours
# done
data_list=("00016" "00017" "00018" "00019" "00020" "00021" "00024" "00025" "00027" "00028" "00034" "00035" "00039" "00041" "00085" "00086" "00087" "00088")
# for item in "${data_list[@]}"; do
#    echo "eval ${item} subject"
#    python eval0722.py --subject ${item} --data_path /home/ps/dy/HaveFun_test/${item}_havefun/meshes_test_obj --is_gt False --method HaveFun --out_dir /home/ps/dy/eval_havefun
# done

for item in "${data_list[@]}"; do
   echo "eval havefun ${item} subject"
   python eval.py --subject ${item} --gt_npy /home/ps/dy/eval_gt/gt_${item}.npy --pre_npy /home/ps/dy/eval_havefun/HaveFun_${item}.npy --method HaveFun --out_dir /home/ps/dy/mycode2/t0717/eval_03
done


# for item in "${data_list[@]}"; do
#    echo "eval base ${item} subject"
#    python eval.py --subject ${item} --gt_npy /home/ps/dy/eval_gt/gt_${item}.npy --pre_npy /home/ps/dy/eval_base/Base_${item}.npy --method Base --out_dir /home/ps/dy/mycode2/t0717/eval_03
# done

# for item in "${data_list[@]}"; do
#    echo "eval base nc ${item} subject"
#    python eval.py --subject ${item} --gt_npy /home/ps/dy/eval_gt/gt_${item}.npy --pre_npy /home/ps/dy/eval_base_nc/Base_Nc_${item}.npy --method Base_Nc --out_dir /home/ps/dy/mycode2/t0717/eval_03
# done

# for item in "${data_list[@]}"; do
#    echo "eval ours ${item} subject"
#    python eval.py --subject ${item} --gt_npy /home/ps/dy/eval_gt/gt_${item}.npy --pre_npy /home/ps/dy/eval_ours/Ours_${item}.npy --method Ours --out_dir /home/ps/dy/mycode2/t0717/eval_03
# done