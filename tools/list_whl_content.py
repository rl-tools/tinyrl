import zipfile
import sys

wheel_path = sys.argv[1]

with zipfile.ZipFile(wheel_path, 'r') as wheel:
    for file_info in wheel.infolist():
        print(file_info.filename)