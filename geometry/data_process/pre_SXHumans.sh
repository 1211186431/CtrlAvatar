data_root = '/path/to/XHumans/{Person_ID}'
out_path = '/path/to/output'
subjects=("00016")
for i in ${!subjects[@]}; do
    subject=${subjects[$i]}
    python data_process/preprocess_XHumans.py --data_root="$data_root/$subject"
    python data_process/preprocess_SXHumans.py --subject "$subject" --xhumans_path "$data_root" --out_path "$out_path"
done