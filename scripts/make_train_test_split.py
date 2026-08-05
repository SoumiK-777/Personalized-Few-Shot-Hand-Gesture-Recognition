"""Create an idempotent 80/20 split manifest from race/gesture image folders.

Existing race/train/gesture and race/test/gesture trees are left untouched. For an
unsplit race/gesture tree, the manifest records the deterministic allocation; use
the manifest to materialize/copy data only if your Kaggle input is read-only.
"""
import json, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DATA_DIR, SEED
from common.data import IMAGE_SUFFIXES

def main():
 base=Path(DATA_DIR); manifest=Path(os.environ.get("SPLIT_MANIFEST_PATH", base/"split_manifest.json"))
 manifest.parent.mkdir(parents=True, exist_ok=True)
 if manifest.exists(): print(f"split manifest already exists; leaving it unchanged: {manifest}");return
 rng=random.Random(SEED); out={"seed":SEED,"splits":{}}
 for race in sorted(p for p in base.iterdir() if p.is_dir()):
  # Already split datasets are the expected Kaggle layout.
  if (race/"train").is_dir() and (race/"test").is_dir():
   for split in ("train","test"):
    for cls in sorted(p for p in (race/split).iterdir() if p.is_dir()): out["splits"].setdefault(race.name,{}).setdefault(cls.name,{})[split]=[str(x.relative_to(base)) for x in sorted(cls.iterdir()) if x.suffix.lower() in IMAGE_SUFFIXES]
   continue
  for cls in sorted(p for p in race.iterdir() if p.is_dir()):
   files=[x for x in sorted(cls.iterdir()) if x.suffix.lower() in IMAGE_SUFFIXES];rng.shuffle(files);cut=max(1,int(.8*len(files))) if files else 0
   out["splits"].setdefault(race.name,{})[cls.name]={"train":[str(x.relative_to(base)) for x in files[:cut]],"test":[str(x.relative_to(base)) for x in files[cut:]]}
 manifest.write_text(json.dumps(out,indent=2));print(f"wrote {manifest}")
if __name__=="__main__":main()
