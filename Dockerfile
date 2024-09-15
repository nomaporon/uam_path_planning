# Dockerfile
FROM python:3.12.0-slim

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    cmake \
    pkg-config \
    libssl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 環境変数を設定
ENV PYTHONUNBUFFERED=1

# GDALの環境変数を設定
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# GDALのバージョンを確認し、環境変数に設定
RUN gdal-config --version > /tmp/gdal_version.txt && \
    echo "GDAL_VERSION=$(cat /tmp/gdal_version.txt)" >> /etc/environment && \
    rm /tmp/gdal_version.txt

# pipをアップグレード
RUN pip install --upgrade pip

# 必要なPythonパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトファイルをコピー
COPY . .

# PYTHONPATHを設定
ENV PYTHONPATH=/app

# メインスクリプトを実行
CMD ["python", "geo_simulation_project/main.py"]