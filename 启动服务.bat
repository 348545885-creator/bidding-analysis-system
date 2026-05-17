@echo off
REM ========================================
REM 投标决策辅助分析系统 - 一键启动脚本
REM 启动 FastAPI 服务并自动打开浏览器
REM ========================================
chcp 65001 >nul
echo ========================================
echo   投标报价智能分析系统
echo ========================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python 3.x
    pause & exit /b 1
)

netstat -ano | findstr :5001 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo [信息] 服务已在运行
) else (
    echo [信息] 正在启动服务...
    start "FastAPI服务" cmd /k "cd /d %~dp0 && python -m uvicorn app:app --host 0.0.0.0 --port 5001"
    timeout /t 5 /nobreak >nul
)

echo [信息] 打开浏览器...
start http://localhost:5001
echo.
echo 访问地址: http://localhost:5001
echo.
echo 按任意键关闭此窗口...
pause >nul
