# YQuantum - Team Superposition Seekers

### Team Members
- Artem Tikhonov – University of California, Davis  
- Albin Franzén – University of California, Davis / Lund University, Sweden  
- Félix Wilhelmy – École de technologie supérieure (ÉTS Montréal)  
- Gabriel Lemay – École de technologie supérieure (ÉTS Montréal)  
- Grace Pang – University of California, Davis  

---

## Challenge Overview

**Wildfire Response Optimization with Quantum Computing**  
Wildfires pose an escalating global threat, demanding rapid and intelligent allocation of resources. This challenge tasked teams with leveraging quantum algorithms to optimize wildfire response strategies using real-world-inspired data and constraints.

### Problem Statement

We focused on real-time **resource allocation** for wildfire containment using real-world fire risk data from [USGS Fire Danger Maps](https://firedanger.cr.usgs.gov/apps/staticmaps). Our goal was to:

- Extract and adapt wildfire intensity data for computational modeling.
- Formulate a constrained allocation problem for dispatching limited firefighting resources to high-risk zones.
- Maximize fire coverage while avoiding redundancy or overlap.
- Solve the problem efficiently using binary optimization via quantum algorithms.

---

## How to Run the Application

To execute the wildfire response optimization pipeline, run the `main.py` script with the following arguments:

```bash
python main.py -f <file_path> -r <resources> [-d <block_height> <block_width>] [-s <spread_factor>] [-o <output_folder>] [-c <lon_min> <lat_min> <lon_max> <lat_max>]
```

### Required Arguments:
- `-f <file_path>`: Path to the heat map file (GeoTIFF) to be analyzed.  
- `-r <int>`: Number of available firefighting resources to dispatch.

### Optional Arguments:
- `-d <int> <int>`: Downsampling block size `(height, width)`. If not provided, no downsampling is applied.  
- `-s <float>`: Fire diffusion/spread factor. Default: `0.5`  
- `-o <folder_path>`: Output directory for results. Default: `./results`  
- `-c <lon_min> <lat_min> <lon_max> <lat_max>`: Longitude/latitude bounding box to crop a specific region of the GeoTIFF. If not provided, the entire map is used.

---

## References & Resources

- Quantum Annealing: A Primer (arXiv)  
- QUBO Formulation Examples (arXiv)  
- Quantum Approximate Optimization Algorithm (QAOA) Overview (arXiv)  
- Quantum Algorithms for Optimization Problems (IEEE)

---

## Acknowledgments

We sincerely thank the **Yale Quantum Institute**, **Tahoe Quantum**, **qBraid**, and all organizers for hosting this challenge and fostering an incredible environment for research and collaboration.

---

## Devpost Submission

🔗 Check out our full project submission on Devpost:  
[https://devpost.com/software/superposition-seekers](https://devpost.com/software/superposition-seekers)