REM 7-zip command executable path
SET ZIP_CMD="C:\Program Files"\7-Zip\7z.exe

REM Create a folder named 'python'
mkdir python

REM cd into that folder
cd python

REM Install the required dependencies
pip install opensearch-py -t ./
pip install requests -t ./
pip install requests-aws4auth -t ./

REM cd out of that folder
cd ..

REM Zip that folder
%ZIP_CMD% a py312_opensearch-py_requests_and_requests-aws4auth.zip ./*

pause