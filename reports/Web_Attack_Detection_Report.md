# Final Project Report: Cybersecurity Web Attack Detection

**Student Name:** Dakshini Anand Neel  
**Student ID:** 2026CS001  
**Topic:** Cyber Security - Web Attack Detection  

---

## 1. Project Overview
This project focuses on building an end-to-end Machine Learning pipeline to detect and classify web-based attacks from HTTP payloads [cite: 1].

*   **Dataset:** The project utilizes the `shengqin/web-attacks-long` dataset sourced from Hugging Face [cite: 1].
*   **Target Classes:** The application classifies payloads into three distinct categories: XSS, SQLi, and Normal [cite: 1].

## 2. Methodology & Pipeline Architecture
The development lifecycle of the attack detection model follows a structured 11-step pipeline [cite: 1]:
*   **Step 0:** Setup & Imports [cite: 1].
*   **Step 1:** Data Collection [cite: 1].
*   **Step 2:** Data Cleaning [cite: 1].
*   **Step 3:** Exploratory Data Analysis (EDA) [cite: 1].
*   **Step 4:** Data Preprocessing [cite: 1].
*   **Step 5:** Train-Test Split [cite: 1].
*   **Step 6:** Model Training [cite: 1].
*   **Step 7:** Model Evaluation [cite: 1].
*   **Step 8:** Hyperparameter Tuning [cite: 1].
*   **Step 9:** Final Prediction [cite: 1].
*   **Step 10:** Deployment [cite: 1].

## 3. Data Collection and Characteristics
During the initial Data Collection phase, the raw HTTP payload data was ingested and structurally analyzed [cite: 1].

### 3.1. Dataset Shape
*   The raw dataset contains 18,400 rows [cite: 1].
*   The dataset is composed of 4 columns: `Payload`, `Label`, `text_label`, and `ID` [cite: 1].

### 3.2. Class Distribution
A class imbalance exists within the dataset, distributed as follows [cite: 1]:
*   **SQLi (SQL Injection):** 8,471 samples [cite: 1].
*   **XSS (Cross-Site Scripting):** 6,690 samples [cite: 1].
*   **Normal:** 3,239 samples [cite: 1].

## 4. Exploratory Data Analysis (EDA)
An initial 3D Feature Space Projection was generated using Principal Component Analysis (PCA) to explore the data distribution [cite: 1].
*   To achieve this, local feature extraction was performed on the raw payloads [cite: 1].
*   The extracted features included payload length, the sum of digits, and the sum of specific special characters (`<`, `>`, `"`, `\`, `(`, `)`) [cite: 1].
*   These features were then mapped into a 3D scatter plot to visualize the clustering tendencies of the Normal, XSS, and SQLi classes [cite: 1].
