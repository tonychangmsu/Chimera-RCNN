nvidia-docker run -it -p 8888:8888 -v $(pwd):/contents -w /contents --rm tonychangcsp/keras:latest  jupyter notebook --port 8888 --ip 0.0.0.0 --no-browser --allow-root

