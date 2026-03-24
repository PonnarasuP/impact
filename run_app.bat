@echo off
echo ============================================
echo   Stock Price Predictor - Streamlit App
echo ============================================

REM Try streamlit directly
streamlit run app.py 2>nul
if %errorlevel% == 0 goto :done

REM Try python -m streamlit
python -m streamlit run app.py 2>nul
if %errorlevel% == 0 goto :done

REM Try py launcher
py -m streamlit run app.py 2>nul
if %errorlevel% == 0 goto :done

echo.
echo ERROR: Could not locate Streamlit.
echo Please run:  pip install -r requirements.txt
echo Then retry:  streamlit run app.py
pause

:done
