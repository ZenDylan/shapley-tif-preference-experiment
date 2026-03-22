@echo off
REM Shapley vs TIF 实验 - Windows 自动运行脚本
REM 用法: 双击 run.bat 或在命令行运行

cd /d "%~dp0"

echo ==============================================
echo Shapley vs TIF 实验 - Windows 批处理脚本
echo ==============================================
echo.

REM ---- 检查 Python ----
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确保已安装 Python 并添加到 PATH
    pause
    exit /b 1
)

REM ---- 检查 GPU ----
echo [环境检查]
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"
echo.

REM ---- 安装依赖 ----
echo [1/5] 安装 Python 依赖（除 PyTorch）...
pip install transformers datasets accelerate trl peft scipy numpy matplotlib seaborn psutil tqdm
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)
echo.

REM ---- 重新安装 PyTorch CUDA 版本（必须！普通 pip install 会装成 CPU 版）----
echo [2/5] 安装 PyTorch CUDA 版本（RTX 4060 需要）...
echo    注意：如果已安装 CUDA 版，跳过此步不会造成问题
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps
if errorlevel 1 (
    echo [错误] PyTorch CUDA 安装失败，请检查 NVIDIA 驱动
    pause
    exit /b 1
)
echo.

REM ---- 验证 PyTorch CUDA ----
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"
echo.

REM ---- 阶段一：数据准备 ----
echo [3/5] 阶段一：数据准备...
echo ----------------------------------------------
python stage1_setup\prepare_data.py
if errorlevel 1 (
    echo [错误] 数据准备失败
    pause
    exit /b 1
)
echo ----------------------------------------------
echo 阶段一-数据准备完成!
echo.

REM ---- 阶段一：SFT 训练 ----
echo [4/5] 阶段一：SFT 训练...
echo （这可能需要 30-120 分钟，取决于 GPU 速度）
echo ----------------------------------------------
python stage1_setup\train_sft.py
if errorlevel 1 (
    echo [错误] SFT 训练失败
    pause
    exit /b 1
)
echo ----------------------------------------------
echo 阶段一-SFT训练完成!
echo.

REM ---- 阶段二：TIF 计算 ----
echo [5/5] 阶段二：TIF/LossDiff/IRM 计算...
echo （这可能需要 1-3 小时，取决于 GPU 速度）
echo ----------------------------------------------
python stage2_tif\compute_tif.py
if errorlevel 1 (
    echo [错误] TIF 计算失败
    pause
    exit /b 1
)
echo ----------------------------------------------
echo 阶段二-TIF计算完成!
echo.

echo ----------------------------------------------
python stage2_tif\compute_lossdiff_irm.py
if errorlevel 1 (
    echo [错误] LossDiff/IRM 计算失败
    pause
    exit /b 1
)
echo ----------------------------------------------
echo 阶段二-LossDiff/IRM完成!
echo.

echo ==============================================
echo 阶段一和阶段二全部完成!
echo 阶段三（Shapley）和阶段四（分析）的代码已就绪。
echo 你可以手动运行:
echo   python stage3_shapley\compute_shapley.py --budget 50
echo   python stage4_analysis\compare.py
echo ==============================================
pause
