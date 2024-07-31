data_list=("00016" "00017" "00018" "00019" "00020" "00021" "00024" "00025" "00027" "00028" "00034" "00035" "00039" "00041" "00085" "00086" "00087" "00088")
# 定义额外的subject列表
extra_subjects=("00036" "00058")

# 定义方法和对应的路径
declare -A methods
methods=( 
    ["HaveFun"]="/home/ps/dy/eval_havefun/HaveFun_"
    ["Base"]="/home/ps/dy/eval_base/Base_"
    ["Base_Nc"]="/home/ps/dy/eval_base_nc/Base_Nc_"
    ["Ours"]="/home/ps/dy/eval_ours/Ours_"
    ["Oursfit"]="/home/ps/dy/eval_oursfit/Oursfit_"
)

output_dir="/home/ps/dy/mycode2/t0717/eval_03"
gt_path="/home/ps/dy/eval_gt/gt_"

# 遍历数据列表和方法
for item in "${data_list[@]}"; do
    for method in "${!methods[@]}"; do
        echo "eval ${method} ${item} subject"
        python ./eval.py --subject ${item} --gt_npy ${gt_path}${item}.npy --pre_npy ${methods[$method]}${item}.npy --method ${method} --out_dir ${output_dir}
    done
done

# 针对额外的subject列表，处理除了HaveFun外的其他方法
for item in "${extra_subjects[@]}"; do
    for method in "${!methods[@]}"; do
        if [ "${method}" != "HaveFun" ]; then
            echo "eval ${method} ${item} subject"
            python ./eval.py --subject ${item} --gt_npy ${gt_path}${item}.npy --pre_npy ${methods[$method]}${item}.npy --method ${method} --out_dir ${output_dir}
        fi
    done
done