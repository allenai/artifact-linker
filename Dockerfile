# The base image, which will be the starting point for the Docker image.
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /stage/

# Copy the `requirements.txt` to `/stage/requirements.txt/` and then install them.
# We do this first because it's slow and each of these commands are cached in sequence.
# If you'd prefer to use a pyproject.toml file that's fine as well.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the file `main.py` to `/stage/main.py`
# You might need multiple of these statements to copy all the files you need for your experiment.
COPY main.py .