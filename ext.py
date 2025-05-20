import zipfile, pathlib

zip_path = pathlib.Path("my_archive.zip")
with zipfile.ZipFile(zip_path) as z:
    z.extractall("data")   # create ./data and unpack there
