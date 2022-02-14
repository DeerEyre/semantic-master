CUDA_VISIBLE_DEVICES=0 python -m model.word_model_sort_batch --port 39851 --work_num 15 --gpus 0,1,2,3,4 &
#CUDA_VISIBLE_DEVICES=3 python -m model.word_model_sort_batch --port 39853 --work_num 10 --gpus 0,1,2,4,5 &
#CUDA_VISIBLE_DEVICES=4 python -m model.word_model_sort_batch --port 39854 --work_num 10 --gpus 0,1,2,3,5 &