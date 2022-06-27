%%!
for N in {1..4}; do \
python mtcnn.py ./data/original/ ./data/cropped/ --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 \
& done