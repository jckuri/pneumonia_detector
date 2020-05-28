# FDA  Submission

**Your Name:** 
Juan Carlos Kuri Pinto

**Name of your Device:** 
Pneumonia Detector From Chest X-Rays

## Algorithm Description 

### 1. General Information

**Intended Use Statement:** 
This software will help doctors diagnose pneumonia from chest X-rays.

**Indications for Use:**
Take a chest X-ray. Use the resulting file as an input for this software. The software will suggest whether that patient has pneumonia or not with an accuracy of 93.75%. Human doctors can use this suggestion in order to help and to accelerate the diagnosis of pneumonia. In this way, doctor will avoid burnout.

**Device Limitations:**
The device only works with chest X-rays with the DICOM format. If you use other file format, the software won't be able to read the file. If you take other type of X-ray or other type of medical image, the software will produce erroneous results.

**Clinical Impact of Performance:**

### 2. Algorithm Design and Function

<< Insert Algorithm Flowchart >>

**DICOM Checking Steps:**

**Preprocessing Steps:**

**CNN Architecture:**


### 3. Algorithm Training

**Parameters:**
* Types of augmentation used during training
* Batch size
* Optimizer learning rate
* Layers of pre-existing architecture that were frozen
* Layers of pre-existing architecture that were fine-tuned
* Layers added to pre-existing architecture

<< Insert algorithm training performance visualization >> 

<< Insert P-R curve >>

**Final Threshold and Explanation:**

### 4. Databases
 (For the below, include visualizations as they are useful and relevant)

**Description of Training Dataset:** 


**Description of Validation Dataset:** 


### 5. Ground Truth



### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

**Ground Truth Acquisition Methodology:**

**Algorithm Performance Standard:**
