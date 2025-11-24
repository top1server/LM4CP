#!/bin/bash

set -e

echo "=================================================="
echo "   SETUP MÔI TRƯỜNG & CHẠY IOI GENERATOR"
echo "=================================================="

echo "[INFO] Đang kiểm tra và cài đặt thư viện..."
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt

export HF_TOKEN="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"
echo "[INFO] Cấu hình Git credential..."
git config --global credential.helper store
export PUSH_TO_HF=1

echo "[INFO] Biến môi trường PUSH_TO_HF đã được set = 1"

echo "=================================================="
echo "   BẮT ĐẦU QUÁ TRÌNH SINH CODE & UPLOAD"
echo "=================================================="

if [ -f "generator.py" ]; then
    python generator.py
else
    echo "[ERROR] Không tìm thấy file generator.py. Hãy đảm bảo bạn đã lưu code vào đúng file."
    exit 1
fi

echo "=================================================="
echo "   HOÀN TẤT!"
echo "=================================================="