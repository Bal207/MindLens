@echo off
echo Installing pyinstaller if not present...
pip install pyinstaller

echo Building MindLens package...
pyinstaller --clean MindLens.spec

echo Build complete. The application is located in the dist/ directory.
pause
