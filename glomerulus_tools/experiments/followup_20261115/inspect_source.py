import zipfile
from common import *
for file in ['skinet_source.zip','skinet_mest_main.zip','skinet_cortex.zip']:
 with zipfile.ZipFile(HERE/'assets'/file) as z:
  print(file,[(i.filename,i.file_size) for i in z.infolist()][:35])
  if 'source' in file:
   z.extractall(HERE/'assets')
  else:
   for n in z.namelist():
    if n.endswith(('.json','.config','.txt')):
     print(n,z.read(n).decode('utf8')[:15000])
