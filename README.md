# 🧠 Multi-Layer Perceptron (MLP): XOR Logic Implementation

**Description:**
An Advanced Artificial Intelligence assignment focused on **Developing an MLP from Scratch** in Python language. MLP is a custom-built neural network architecture and acts as a virtual XOR gate. This assignment serves as a technical demonstration of solving non-linear classification problems by using and training the MLP without the aid of external libraries like NumPy or TensorFlow.

---


## 👤 Student Details

| Name | Matric No. | Assignment Focus |
| --- | --- | --- |
| **Lau Su Hui (Abby)** | MEC245045 | Developing a MLP from scratch (Python) |

---


## 📂 Project Modules

This repository contains the complete implementation and documentation for the MLP assignment focusing on the logic transformation from simple input to non-linear output.

### 1. Source Code: MLP from Scratch
**File:** `UTM MECS1023 AAI Assignment 1 MLP Code - Lau Su Hui MEC245045 (No NumPy).py` in src folder
* **Architecture:** 2-4-1 Topology (2 Input Nodes, 4 Hidden Nodes, 1 Output Node)
* **Constraints:** Built using pure Python lists and the `math` library to handle matrix logic manually.
* **Mechanism:** Implements Forward Propagation and Backpropagation with Gradient Descent.
* **Learning:** Successfully reduces Mean Squarred Error (MSE) from **0.94** to **0.001** over 10,000 epochs.


### 2. Project Report (Technical Analysis)
**File:** `UTM MECS1023 AAI Assignment 1 MLP Report - Lau Su Hui MEC245045 (No NumPy).py` in report folder
* **Theory:** Detailed explanation of the XOR problem and why a "Hidden Layer" is required for non-linear separation.
* **Analysis:** Step-by-step breakdown of training results, including pre-training random guesses vs. post-training logic verification.


### 3. Presentation Slides
**File:** `UTM MECS1023 AAI Assignment 1 MLP Slides - Lau Su Hui MEC245045 (No NumPy).py` in report folder
* **Summary:** Condensed visual version of the project methodology and outcome.

---

## 🛠️ Technologies Used

* **Language:** Python
* **Core Concepts:** Multi-Layer Perceptron (MLP), Backpropagation, Gradient Descent, Non-Linear Classification.
* **Libraries:**<br>
`math`: Used for the exponential function ($e^{-x}$) in Sigmoid calculations.<br>
`random`: Used for weight initialisation between -1 and 1 to break symmetry.<br>

---


## 🚀 How to Run

### 1. Execute the Script
* Ensure you have Python installed.
* No external dependencies are required.
* Open your terminal and run:
`python "UTM MECS1033 AAI Assignment 1 MLP Code - Lau Su Hui MECS245045 (No NumPy).py"`

---

### 2. Understand the Output
The program will automatically cycle through three phases:
* **Pre-Training Check:** Shows the "dumb" state of the network where predictions are random (~0.5).
* **Training Phase:** Prints the error loss every 1,000 iterations as the "machine" learns.
* **Final Results:** Prints the verified XOR logic where the network correctly outputs ~0 for $[0,0]/[1,1]$ and ~1 for $[0,1]/[1,0]$.


---


## 📑 Project Insights

The project demonstrates the mathematical journey of a neural network learning to "think" logically. Key Findings are shown below:
* **Non-Linearity:** Proved that a single-layer perceptron cannot solve XOR; a hidden layer is mandatory.
* **Learning Rate:** Optimized at 0.5 to ensure stable convergence without overshooting the global minimum.
* **Precision:** Achieved high accuracy with final loss dropping to 0.001180, successfully mimicking the XOR truth table.
  
