#for joint steam
CUDA_VISIBLE_DEVICES=0 python pretrain_moco_mask.py --lr 0.02 --batch-size 128 --teacher-t 0.05 --student-t 0.1 --topk 16384 --mlp --contrast-t 0.07 --contrast-k 16384 --checkpoint-path mask_checkpoints/PCM3_wprompt --schedule 351 --epochs 451 --pre-dataset ntu60 --skeleton-representation graph-based --protocol cross_view --exp-descri PCM3_wprompt
#for bone and motion steam
CUDA_VISIBLE_DEVICES=0 python pretrain_moco_mask.py --lr 0.02 --batch-size 128 --teacher-t 0.05 --student-t 0.1 --topk 16384 --mlp --contrast-t 0.07 --contrast-k 16384 --checkpoint-path mask_checkpoints/PCM3_wprompt --schedule 351 --epochs 451 --pre-dataset ntu60 --skeleton-representation graph-based --protocol cross_view --exp-descri PCM3_woprompt
