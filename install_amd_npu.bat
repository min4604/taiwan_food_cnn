@echo off
echo ===============================================
echo AMD Ryzen AI 9HX NPU 支援安裝腳本
echo ===============================================
echo.

echo 正在安裝 AMD NPU 支援套件...
echo.

echo 1. 安裝 ONNX Runtime (DirectML 支援)
pip install onnxruntime-directml

echo.
echo 2. 安裝 torch-directml (如果可用)
pip install torch-directml || echo torch-directml 安裝失敗，將使用 ONNX Runtime

echo.
echo 3. 安裝其他相關套件
pip install onnx

echo.
echo 4. 檢查安裝結果
python -c "import onnxruntime as ort; print('ONNX Runtime 版本:', ort.__version__); print('可用提供者:', ort.get_available_providers())"

echo.
echo ===============================================
echo 安裝完成！
echo.
echo 📌 重要提醒:
echo 1. 確保 AMD 顯示卡驅動程式是最新版本
echo 2. 確保 Windows 版本支援 DirectML (Windows 10 1903+ 或 Windows 11)
echo 3. 如果 NPU 仍無法使用，請檢查 BIOS 設定中是否啟用了 NPU
echo ===============================================
pause