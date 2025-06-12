@echo off
REM Script to create/update conda env and run application
set ENV_NAME=fake-reviews-detector
set ENV_FILE=environment.yml
set MAIN_MODULE=fake_reviews_detector.main

REM Check if environment exists
conda env list | findstr /R /C:"^%ENV_NAME%" >nul
if %ERRORLEVEL%==0 (
    echo Conda environment '%ENV_NAME%' exists. Skipping creation.
) else (
    echo Creating conda environment '%ENV_NAME%'...
    conda env create -f %ENV_FILE%
)

REM Activate environment
call conda activate %ENV_NAME%

REM Install package if not already installed
python -c "import %MAIN_MODULE:~0,-11%" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing package in editable mode...
    pip install -e .
)

REM Run the application
python -m %MAIN_MODULE%
