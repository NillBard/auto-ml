# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN mkdir -p /tmp && chmod 1777 /tmp
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6

RUN ffmpeg -version  # Проверить версию ffmpeg
WORKDIR /app
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu 
# COPY requirements.txt .
# RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y gcc python3-dev build-essential && \
    pip install pycocotools

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Установка GStreamer и его зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    ffmpeg \
    libsm6 \
    libxext6

COPY src .
