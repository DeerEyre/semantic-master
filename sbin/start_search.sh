CUDA_VISIBLE_DEVICES=5 python -u /usr/local/bin/gunicorn --workers=4 --bind=0.0.0.0:33366 server.new_version_tree_search_beamsearch:app --worker-class sanic.worker.GunicornWorker --timeout 6000000  &
python -m server.aggregate_search &
