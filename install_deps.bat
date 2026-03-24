@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================
echo   Installing Stock Price Predictor Dependencies
echo ================================================
echo.

REM Install requirements
echo Installing from requirements.txt...
py -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo WARNING: Installation had issues. Trying individual packages...
    echo.
    
    echo Installing streamlit...
    py -m pip install streamlit
    
    echo Installing yfinance...
    py -m pip install yfinance
    
    echo Installing plotly...
    py -m pip install plotly
    
    echo Installing pandas...
    py -m pip install pandas
    
    echo Installing scikit-learn...
    py -m pip install scikit-learn
    
    echo Installing Prophet (this may take a minute)...
    py -m pip install pystan==2.19.1.1
    py -m pip install prophet
    
    if !errorlevel! neq 0 (
        echo.
        echo NOTE: Prophet installation via pip failed on Windows.
        echo You can skip Prophet and still use Linear Regression model.
        echo.
        echo ALTERNATIVE: Use conda to install Prophet:
        echo   conda install -c conda-forge prophet
        echo.
    )
    
    echo Done!
) else (
    echo.
    echo SUCCESS! All packages installed.
    echo You can now run:  streamlit run app.py
    echo.
)

pause
