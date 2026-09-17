from pathlib import Path
import zipfile,json
root=Path('C:/Users/User/Desktop/HERA')
build=root/'tmp/shared_hetero_recap_revision_20260916'
src=root/'results/group_meeting_20260915/HERA_group_meeting_20260915_shared_formulas.pptx'
# Preserve the original charts, their workbooks and all unrelated slide parts.
changed={'ppt/slides/slide2.xml','ppt/slides/_rels/slide2.xml.rels','ppt/notesSlides/notesSlide2.xml'}
with zipfile.ZipFile(src) as original,zipfile.ZipFile(build/'draft.pptx') as edited,zipfile.ZipFile(build/'candidate_preserved.pptx','w',zipfile.ZIP_DEFLATED) as dest:
    for entry in original.infolist():
        dest.writestr(entry,edited.read(entry.filename) if entry.filename in changed else original.read(entry.filename))
with zipfile.ZipFile(src) as original,zipfile.ZipFile(build/'candidate_preserved.pptx') as dest:
    actual={n for n in original.namelist() if original.read(n)!=dest.read(n)}
    assert actual==changed,actual
    assert set(original.namelist())==set(dest.namelist())
print(json.dumps({'changed_parts':sorted(actual),'all_other_parts':'byte-identical to source'},indent=2))
