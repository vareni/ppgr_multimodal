#!/bin/bash
#SBATCH --job-name=predict_vlm_qwen36
#SBATCH --output=./logs/%A_predict_vlm_qwen36.out
#SBATCH --error=./logs/%A_predict_vlm_qwen36.err
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --nodes=1
##SBATCH --nodelist=mandalore  #kessel,corellia,mandalore,ithor
#SBATCH --exclude=iego

source ~/.bashrc
#source /opt/miniconda3/etc/profile.d/conda.sh
conda activate vlm_env

export PYTHONNOUSERSITE=1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

/home/guests/varvara_kondratyeva/.conda/envs/vlm_env/bin/python - <<'PY'
import torch
print("torch:", torch.__version__, torch.version.cuda)
print("cuda:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
    free, total = torch.cuda.mem_get_info(i)
    print("free/total GB:", free / 1024**3, total / 1024**3)
PY

/home/guests/varvara_kondratyeva/.conda/envs/vlm_env/bin/python vlm_direct_iauc_modes.py \
  --backend qwen \
  --model_id  Qwen/Qwen3.6-27B-FP8\
  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
  --image_root ../cgmacros \
  --input_mode clinical_macros \
  --train_txt ../cgmacros/lunch_sub_train_test_val/train.txt \
  --n_examples 3 \
  --out CHECKPOINTS/vlm_qwen36_macros_micro.csv \
  --keep_debug_cols

/home/guests/varvara_kondratyeva/.conda/envs/vlm_env/bin/python vlm_direct_iauc_modes.py \
  --backend qwen \
  --model_id  Qwen/Qwen3.6-27B-FP8\
  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
  --image_root ../cgmacros \
  --input_mode clinical_macros \
  --train_txt ../cgmacros/lunch_sub_train_test_val/train.txt \
  --n_examples 3 \
  --out CHECKPOINTS/vlm_qwen36_macros.csv \
  --keep_debug_cols \
  --no_microbiome

/home/guests/varvara_kondratyeva/.conda/envs/vlm_env/bin/python vlm_direct_iauc_modes.py \
  --backend qwen \
  --model_id  Qwen/Qwen3.6-27B-FP8\
  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
  --image_root ../cgmacros \
  --input_mode image_clinical \
  --train_txt ../cgmacros/lunch_sub_train_test_val/train.txt \
  --n_examples 3 \
  --out CHECKPOINTS/vlm_qwen36_image_micro.csv \
  --keep_debug_cols

/home/guests/varvara_kondratyeva/.conda/envs/vlm_env/bin/python vlm_direct_iauc_modes.py \
  --backend qwen \
  --model_id  Qwen/Qwen3.6-27B-FP8\
  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
  --image_root ../cgmacros \
  --input_mode image_clinical \
  --train_txt ../cgmacros/lunch_sub_train_test_val/train.txt \
  --n_examples 3 \
  --out CHECKPOINTS/vlm_qwen36_image.csv \
  --keep_debug_cols \
  --no_microbiome

#Qwen/Qwen2.5-VL-3B-Instruct \
# Qwen/Qwen3.6-27B-FP8
# lingshu-medical-mllm/Lingshu-7B

#vlm_direct_iauc_modes.py \
#  --backend internvl \
#  --model_id OpenGVLab/InternVL3-8B-Instruct \
#  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
#  --image_root ../cgmacros \
#  --input_mode clinical_macros \
#  --out CHECKPOINTS/vlm_internvl3_8b_fewshot_clinical_macros_micro.csv \
#  --keep_debug_cols

#  --train_txt ../cgmacros/lunch_sub_train_test_val/train.txt \
#  --n_examples 3 \

#vlm_direct_iauc.py \
#  --backend internvl \
#  --model_id OpenGVLab/InternVL3-8B-Instruct \
#  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
#  --image_root ../cgmacros \
#  --out CHECKPOINTS/vlm_internvl3_test_iauc_no_micro.csv


#vlm_direct_iauc_modes.py \
#  --backend qwen \
#  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
#  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
#  --image_root ../cgmacros \
#  --input_mode clinical_macros \
#  --out CHECKPOINTS/vlm_qwen25vl_3b_clinical_macros_no_micro.csv \
#  --keep_debug_cols






# debug_internvl_img.py

