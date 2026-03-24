# Coronary Artery Reconstruction from CTA using CNN-based Filtering

##  Overview

This project explores the reconstruction of coronary arteries from CTA (Computed Tomography Angiography) data using a **hybrid approach combining segmentation-derived seeds and CNN-based filtering**.

The work is inspired by:

* CNNTracker (Wolterink et al.)
* Vessel reconstruction approaches using CTA datasets

However, instead of classical tracking, this project adapts the method for **dense vessel representations**, as found in CTA datasets like ImageCAS.

---

##  Key Idea

Traditional CNNTracker:

* Starts from few seeds
* Tracks vessels step-by-step using direction predictions

Our approach:

* Starts from **dense vessel candidates (segmentation-based seeds)**
* Uses **ostia to localize vessel regions**
* Applies **CNN-based filtering to refine vessel points**
* Computes **radius separately**
* Uses both for reconstruction

---

##  Pipeline

```text
CTA Image
   ↓
Seed Generation (from segmentation)
   ↓
Ostia Detection
   ↓
Ostia-guided Seed Filtering (Distance + KMeans)
   ↓
CNN-based Point Validation (Tracking approximation)
   ↓
Radius Computation
   ↓
Vessel Reconstruction
```

---

## 📍 Role of Ostia (Important Contribution)

Unlike simple filtering, this project explicitly uses **ostia points** to:

* Localize vessel regions
* Separate coronary branches
* Improve seed distribution
* Guide structured sampling

This makes the pipeline **anatomically meaningful**, even without explicit tracking.

---

## Results

### 3D Vessel Representation

* Dense centerline-like structure recovered
* Smooth anatomical continuity observed

### Radius Overlay

* Radius aligns well with vessel thickness
* Supports downstream reconstruction

*(See `/results/` for visualizations)*

---

##  Key Components

### 1. Seed Generation

Extracts vessel candidate points from segmentation masks.

### 2. Ostia-based Filtering

* Filters seeds based on distance to ostia
* Applies KMeans clustering for uniform spatial distribution

### 3. CNN-based Filtering

* Uses pretrained CNNTracker model
* Classifies valid vessel points

### 4. Radius Computation

* Computes local vessel radius (separate module)

---

##  Important Note

This implementation is **not a direct replication of CNNTracker tracking**.

Instead, it is an adaptation for dense CTA data where:

* Vessel structure is already present
* Tracking is replaced by filtering + structuring

---

##  How to Run

```bash
# Step 1: Generate seeds
python preprocessing/generate_seeds.py

# Step 2: Generate ostia
python preprocessing/generate_ostia.py

# Step 3: Filter seeds
python preprocessing/filter_seeds_kmeans.py

# Step 4: Run CNN filtering
python tracking/track2.py --impath ... --seeds ... --ostia ... --tracknet ...

# Step 5: Compute radius
python radius/compute_radius.py

# Step 6: Visualize
python visualization/visualize_points.py
```

---

##  Dependencies

* Python 3.8+
* PyTorch
* SimpleITK
* NumPy
* scikit-learn
* matplotlib

---

##  Key Insight

> Dense CTA data does not require explicit tracking.
> Instead, structured filtering + geometric reconstruction can produce accurate vessel representations.

---

##  Author

* Mukul Bansal

---

##  References

* CNNTracker: Wolterink et al.
* CTA-based vessel reconstruction literature

---
