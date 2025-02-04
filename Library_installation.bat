chcp 65001
@echo off
REM Pythonライブラリ一括インストール用バッチファイル
echo 必要なライブラリのインストールを開始します...

REM pipのアップグレード
python -m pip install --upgrade pip

REM 主要ライブラリのインストール
pip install opencv-python
pip install torch torchvision
pip install numpy

echo インストールが完了しました。
pause
