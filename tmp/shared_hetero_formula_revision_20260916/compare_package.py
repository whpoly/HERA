from pathlib import Path
import zipfile,hashlib,json
root=Path('C:/Users/User/Desktop/HERA')
src=root/'results/group_meeting_20260915/HERA_group_meeting_20260915.pptx'
dst=root/'tmp/shared_hetero_formula_revision_20260916/draft.pptx'
with zipfile.ZipFile(src) as a,zipfile.ZipFile(dst) as b:
    changed=[]
    for n in a.namelist():
        if n not in b.namelist() or a.read(n)!=b.read(n): changed.append(n)
    new=set(b.namelist())-set(a.namelist())
    print(json.dumps({'changed_parts':changed,'new_parts':sorted(new)},indent=2))
