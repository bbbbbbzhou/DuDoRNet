python test.py \
--resume './outputs/train_DuDoRN_R4_1Do_0pT1_Cartesian/checkpoints/model_49.pt' \
--experiment_name 'test_DuDoRN_R4_1Do_0pT1_Cartesian' \
--accelerations 5 \
--model_type 'model_recurrent_dual' \
--data_root '../Data/PROC/' \
--dataset 'Cartesian' \
--net_G 'DRDN' \
--n_recurrent 4 \
--gpu_ids 3
