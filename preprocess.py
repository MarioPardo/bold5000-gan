"""
Converts BOLD5000 raw GLM beta volumes (TYPED-D) into self-contained compressed
numpy arrays for training the Graph Attention Network.

For each subject-session:
  1. Load beta .nii.gz (X, Y, Z, N_trials)
  2. Patch the broken affine using the subject's brainmask (correct MNI affine)
  3. Z-score betas per voxel within each session (recommended by dataset authors)
  4. Apply Schaefer-400 atlas masker -> (N_trials, 400) matrix
  5. Concatenate all sessions for the subject
  6. Save as .npz (see Output fields below)

Output .npz fields:
  betas          : float32  (N, 400)  — Schaefer-400 regional BOLD responses
  imgnames       : str      (N,)      — original stimulus filename (also used as AWS key)
  subject        : str      (N,)      — subject ID (e.g. 'CSI1')
  sessions       : int8     (N,)      — session number each trial came from (1–15) #used to fetch raw from aws
  local_idxs     : int16    (N,)      — within-session volume index for each trial  #used to fetch raw from aws
  dataset_sources: str      (N,)      — 'ImageNet', 'COCO', or 'Scene'
  labels         : object   (N,)      — list[str] of human-readable labels per trial
  parcel_names   : str      (400,)    — Schaefer-400 region names


  
AWS paths for raw beta files (if needed for visualization):
  s3://openneuro.org/ds001246/derivatives/betas/{subject}/
      {subject}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-{NN}.nii.gz

"""

import os
import re
import json
import time
import warnings
import numpy as np
import nibabel as nib
from scipy import stats
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

BOLD5000_DIR = "/media/hdd/BOLD5000"
OUTPUT_DIR   = "/home/mariop/Documents/Programming/bold5000-gan/processed_data"
N_ROIS       = 414   # 400 Schaefer regions + 14 subcortical regions

SESSIONS = {
    "CSI1": list(range(1, 16)),
    "CSI2": list(range(1, 16)), 
    "CSI3": list(range(1, 16)),
    "CSI4": list(range(1, 10)),  
}

# ── Label resolution ─────────────────────────────────────────────────────────

def load_label_resources(bold5000_dir: str) -> tuple[dict, dict]:
    """
    Load the two lookup tables needed to resolve human-readable labels:
      imagenet_synset_to_label : synset (e.g. 'n01945685') -> class name
      coco_img_to_categories   : COCO image_id (int)       -> comma-joined category names

    Both files live in label_mappings/ next to preprocess.py.
    If missing they are downloaded 
    """
    mappings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_mappings")
    os.makedirs(mappings_dir, exist_ok=True)

    # ImageNet 
    inet_path = os.path.join(mappings_dir, "imagenet_class_index.json")
    if not os.path.exists(inet_path):
        import urllib.request
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        print(f"  Downloading ImageNet class index from {url} ...")
        urllib.request.urlretrieve(url, inet_path)
        print(f"Saved: {inet_path}")
    with open(inet_path) as f:
        raw = json.load(f)
    imagenet_synset_to_label = {v[0]: v[1] for v in raw.values()}

    # COCO 
    coco_path = os.path.join(mappings_dir, "instances_train2014.json")
    if not os.path.exists(coco_path):
        import urllib.request, zipfile
        zip_url  = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        zip_path = os.path.join(mappings_dir, "annotations_trainval2014.zip")
        print(f"  Downloading COCO annotations zip (~18 MB) ...")
        urllib.request.urlretrieve(zip_url, zip_path)
        print(f"  Extracting instances_train2014.json ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("annotations/instances_train2014.json", mappings_dir)
        # Move out of the annotations/ subfolder
        extracted = os.path.join(mappings_dir, "annotations", "instances_train2014.json")
        os.rename(extracted, coco_path)
        os.rmdir(os.path.join(mappings_dir, "annotations"))
        os.remove(zip_path)
        print(f"Saved: {coco_path}")

    print("  Building COCO image→categories map ...")
    with open(coco_path) as f:
        coco_data = json.load(f)
    coco_categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    coco_img_to_categories: dict[int, list[str]] = {}
    for ann in coco_data["annotations"]:
        img_id   = ann["image_id"]
        cat_name = coco_categories.get(ann["category_id"], "unknown")
        coco_img_to_categories.setdefault(img_id, [])
        if cat_name not in coco_img_to_categories[img_id]:
            coco_img_to_categories[img_id].append(cat_name)
    print(f"COCO: {len(coco_img_to_categories)} images, {len(coco_categories)} categories")

    return imagenet_synset_to_label, coco_img_to_categories


def resolve_label(filename: str,
                  imagenet_synset_to_label: dict,
                  coco_img_to_categories: dict) -> tuple[str, list[str]]:
    """
    Return (dataset_source, labels) for a stimulus filename.
   
    EX:
    ImageNet : 'n01945685_2329.JPEG'             -> ('ImageNet', ['sea slug'])
    COCO     : 'COCO_train2014_000000396550.jpg' -> ('COCO',     ['dog', 'frisbee'])
    Scene    : 'concert5.jpg'                    -> ('Scene',    ['Concert'])
    """
    basename = os.path.basename(filename)

    # ImageNet: starts with n followed by digits, then underscore
    if re.match(r'^n\d+_', basename):
        synset = basename.split('_')[0]
        label  = imagenet_synset_to_label.get(synset, synset)
        return "ImageNet", [label]

    # COCO: filename contains 'COCO'
    if 'COCO' in basename:
        m      = re.search(r'(\d{6,12})', basename)
        img_id = int(m.group(1)) if m else -1
        cats   = coco_img_to_categories.get(img_id, [])
        return "COCO", cats if cats else [f"COCO image {img_id}"]

    # Scene / SUN / other
    scene = os.path.splitext(basename)[0]
    scene = re.sub(r'\d+', '', scene)
    scene = re.sub(r'([a-z])([A-Z])', r'\1 \2', scene)
    return "Scene", [scene.strip().title() or basename]


# ── Helpers ───────────────────────────────────────────────────────────────────

#load all the image names for a subject
def load_imgnames(subject: str) -> list[str]:

    path = os.path.join(BOLD5000_DIR, f"{subject}_imgnames.txt")
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


#Load the brainmask to get the correct MNI-space affine.
def get_correct_affine(subject: str) -> np.ndarray:
    """
    The TYPED-D beta .nii.gz files were saved with an incorrect affine
    The correct one was stored in fmriprep
    """

    path = os.path.join(BOLD5000_DIR, f"{subject}_brainmask.nii.gz")
    mask_img = nib.load(path)
    return mask_img.affine


def load_session_betas(subject: str, session: int, correct_affine: np.ndarray) -> nib.Nifti1Image:
    """
    Load one session's beta volume, patch the affine, z-score, and return
    a NIfTI image for atlas extraction.

    Returns:
        nib.Nifti1Image with shape (X, Y, Z, N_trials) and correct affine
    """
    filename = f"{subject}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-{session:02d}.nii.gz"
    path = os.path.join(BOLD5000_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Beta file not found: {path}")

    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)  # (X, Y, Z, N_trials)

    #z score for each entry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = stats.zscore(data, axis=3, nan_policy="omit")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) #replace NaN with 0s

    #NiftilImage is standard for brain data
    return nib.Nifti1Image(data, correct_affine)


def build_masker(n_rois: int = 400) -> tuple[NiftiLabelsMasker, np.ndarray, NiftiLabelsMasker, np.ndarray]:
    import nibabel as nib
    import numpy as np
    
    """
    Fetch the Schaefer atlas and build a fitted-ready masker.
    Also fetch the Harvard-Oxford subcortical atlas and build its masker.
    """
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=7, resolution_mm=2
    )
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=False,          # we already z-scored per session
        resampling_target="data",   # resample atlas to match beta grid
        verbose=0,
    )

    # Strip the leading 'Background' label
    raw_labels = [
        lbl.decode() if isinstance(lbl, bytes) else str(lbl)
        for lbl in atlas.labels
    ]
    parcel_names_ctx = np.array([l for l in raw_labels if l.lower() != "background"])
    assert len(parcel_names_ctx) == 400, \
        f"Expected 400 parcel names after stripping background, got {len(parcel_names_ctx)}"

    subcortical_atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    # 14 ENIGMA regions
    # Thalamus, Caudate, Putamen, Pallidum, Hippocampus, Amygdala, Accumbens
    target_labels = [
        'Left Thalamus', 'Right Thalamus',
        'Left Caudate', 'Right Caudate',
        'Left Putamen', 'Right Putamen',
        'Left Pallidum', 'Right Pallidum',
        'Left Hippocampus', 'Right Hippocampus',
        'Left Amygdala', 'Right Amygdala',
        'Left Accumbens', 'Right Accumbens'
    ]
    
    # create mapping
    label_list = subcortical_atlas.labels
    label_to_id = {label: i for i, label in enumerate(label_list)}
    target_ids = [label_to_id[label] for label in target_labels]
    
    maps = subcortical_atlas.maps
    data = maps.get_fdata()
    
    new_data = np.zeros_like(data)
    for i, target_id in enumerate(target_ids):
        new_data[data == target_id] = i + 1
        
    filtered_maps = nib.Nifti1Image(new_data, maps.affine, maps.header)
    
    sub_masker = NiftiLabelsMasker(
        labels_img=filtered_maps,
        standardize=False,
        resampling_target="data",
        verbose=0
    )
    parcel_names_sub = np.array(target_labels)
    
    return masker, parcel_names_ctx, sub_masker, parcel_names_sub


# ── Main processing function  ───────────────────────────────────────────────────

#Process all sessions for one subject and save a single self-contained .npz.
def process_subject(
    subject: str,
    masker: NiftiLabelsMasker,
    parcel_names_ctx: np.ndarray,
    sub_masker: NiftiLabelsMasker,
    parcel_names_sub: np.ndarray,
    imagenet_synset_to_label: dict,
    coco_img_to_categories: dict,
    sessions: list[int] | None = None,
    output_dir: str | None = None,
) -> str:
   
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if sessions is None:
        sessions = SESSIONS[subject]

    n_rois = len(parcel_names_ctx) + len(parcel_names_sub)

    print(f"\n{'='*60}")
    print(f"  Processing subject: {subject}  ({len(sessions)} sessions, {n_rois} ROIs)")
    print(f"{'='*60}")

    correct_affine = get_correct_affine(subject)
    imgnames_all   = load_imgnames(subject)

    all_betas          = []
    all_imgnames       = []
    all_sessions       = [] 
    all_local_idxs     = []   
    all_dataset_sources = []  
    all_labels         = []   

    trial_cursor = 0  # pointer into imgnames_all

    for sesh in sessions:
        print(f"  Session {sesh:02d} ...", end=" ", flush=True)
        t0 = time.time()

        beta_img = load_session_betas(subject, sesh, correct_affine)
        n_trials = beta_img.shape[3]

        # Extract Schaefer regional averages: (n_trials, 400)
        cortical_betas = masker.fit_transform(beta_img)
        # Extract subcortical: (n_trials, 14)
        subcortical_betas = sub_masker.fit_transform(beta_img)
        
        session_betas = np.hstack((cortical_betas, subcortical_betas))

        # Image names for this session
        session_imgnames = imgnames_all[trial_cursor: trial_cursor + n_trials]

        # Resolve labels for each trial in this session
        session_sources = []
        session_labels  = []
        for fname in session_imgnames:
            src, lbl = resolve_label(fname, imagenet_synset_to_label, coco_img_to_categories)
            session_sources.append(src)
            session_labels.append(lbl)

        # Session and local index bookkeeping
        session_nums  = np.full(n_trials, sesh,              dtype=np.int8)
        local_indices = np.arange(n_trials,                  dtype=np.int16)

        all_betas.append(session_betas.astype(np.float32))
        all_imgnames.extend(session_imgnames)
        all_sessions.append(session_nums)
        all_local_idxs.append(local_indices)
        all_dataset_sources.extend(session_sources)
        all_labels.extend(session_labels)

        trial_cursor += n_trials
        elapsed = time.time() - t0
        print(f"shape={session_betas.shape}  ({elapsed:.0f}s)  cumulative={trial_cursor}")

    # Concatenate and validate 
    betas_array           = np.concatenate(all_betas, axis=0)   
    imgnames_array        = np.array(all_imgnames)              
    sessions_array        = np.concatenate(all_sessions)        
    local_idxs_array      = np.concatenate(all_local_idxs)      
    dataset_sources_array = np.array(all_dataset_sources)        
    subject_array         = np.array([subject] * len(imgnames_array)) 


    labels_array = np.empty(len(all_labels), dtype=object)
    for i, lbl in enumerate(all_labels):
        labels_array[i] = lbl   # lbl is already a list[str]

    assert betas_array.shape[0] == len(imgnames_array), \
        f"Mismatch: {betas_array.shape[0]} betas vs {len(imgnames_array)} imgnames"

    # Save 
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{subject}_schaefer{n_rois}.npz")

    np.savez_compressed(
        out_path,
        betas           = betas_array,     
        imgnames        = imgnames_array,          
        subject         = subject_array,          
        sessions        = sessions_array,         
        local_idxs      = local_idxs_array,      
        dataset_sources = dataset_sources_array,   
        labels          = labels_array,           
        parcel_names    = np.concatenate((parcel_names_ctx, parcel_names_sub)),          
    )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    multi_label_count = sum(1 for lbl in labels_array if len(lbl) > 1)
    print(f"\n  ✓ Saved: {out_path}")
    print(f"    betas shape      : {betas_array.shape}")
    print(f"    file size        : {size_mb:.2f} MB")
    print(f"    dataset breakdown: " +
          ", ".join(f"{src}={np.sum(dataset_sources_array == src)}"
                    for src in ["ImageNet", "COCO", "Scene"]))
    print(f"    multi-label trials (COCO with >1 object): {multi_label_count}")

    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():

    load_label_resources("/media/hdd/BOLD5000")

    '''

    bold5000_dir = "/media/hdd/BOLD5000"
    output_dir   = "/home/mariop/Documents/Programming/bold5000-gan/processed_data"
    n_rois       = 400       # 400 Schaefer regions
    subjects     = ["CSI1", "CSI2", "CSI3", "CSI4"]
    sessions     = None         # None = all sessions for each subject

    print(f"BOLD5000 source : {bold5000_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Subjects        : {subjects}")
    print(f"Sessions        : {sessions if sessions else 'all'}")
    print(f"Atlas ROIs      : {n_rois} + 14")

    print("\nLoading label resources (ImageNet + COCO) ...")
    imagenet_synset_to_label, coco_img_to_categories = load_label_resources(bold5000_dir)
    print("✓ Label resources ready")

    print("\nFetching Schaefer & Subcortical atlas ...")
    masker, parcel_names_ctx, sub_masker, parcel_names_sub = build_masker(n_rois=n_rois)
    print(f"✓ Atlas ready — {len(parcel_names_ctx)} cortical + {len(parcel_names_sub)} subcortical parcels")

    saved_files = []
    for subject in subjects:
        if subject not in SESSIONS:
            print(f"  ✗ Unknown subject '{subject}', skipping.")
            continue

        subject_output_dir = os.path.join(output_dir, subject)

        out_path = process_subject(
            subject                  = subject,
            masker                   = masker,
            parcel_names_ctx         = parcel_names_ctx,
            sub_masker               = sub_masker,
            parcel_names_sub         = parcel_names_sub,
            imagenet_synset_to_label = imagenet_synset_to_label,
            coco_img_to_categories   = coco_img_to_categories,
            sessions                 = sessions,
            output_dir               = subject_output_dir,
        )
        saved_files.append(out_path)

    print(f"\n{'='*60}")
    print(f"  All done. {len(saved_files)} file(s) written.")
    for f in saved_files:
        print(f"    {f}")
    print(f"{'='*60}\n")

    '''


if __name__ == "__main__":
    main()
