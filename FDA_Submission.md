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

If you carefully examine the distribution of ages of the patients of which the model was trained and validated, you will notice that the big majority of patients have an age between 20 and 70 years. Patients whose age is outside such age range will probably receive a wrong diagnose from this device.

<p align='center'><img src='EDA_images/age.png' width='50%'/></p>

This kind of bias does not occur regarding the gender of patients. Male patients and female patients have similar proportions. So, this device works properly in both sexes.

**Device Limitations:**

As stated above, the big majority of patients have an age between 20 and 70 years. Patients whose age is outside such age range will probably receive a wrong diagnose from this device. Because there are few patients who are too young or too old; and this device have not received enough training in those cases.

If you carefully examine the distribution of comorbidities, the diseases which are comorbid with pneumonia, you will notice that some diseases like Infiltration, Edema, Effusion, and Atelectasis are highly correlated with pneumonia. It is well known that the bigger the correlation of the comorbidity, the more confusion will cause in this device. In fact, the only mistake this device committed in the FDA Validation Dataset was a patient with **Effusion**, a comorbidity highly correlated with pneumonia. Hence, the device was confused. A good recommendation to doctors is to pay special attention to the automated diagnoses of patients with comorbidities highly correlated with pneumonia.

<p align='center'><img src='EDA_images/comorbidities_pneumonia.png' width='50%'/></p>

This device will also fail with patients with medical implants in the chest area like metal bones, screws, machines, and so on. Because those medical implants will appear very bright in the X-ray, confusing the device.

Moreover, the device only works with chest X-rays with the DICOM format. If you use other file format, the software won't be able to read the file. If you take other type of X-ray, other body part different than the chest, or other type of medical image, the software will produce erroneous results.

**Clinical Impact of Performance:**

How false positives might affect a patient?

False positives might alarm doctors in vain. They will lose some time confirming that patients really have pneumonia. Doctors will experience fatigue more often and will commit more mistakes, which negatively affects patients. If doctors also believe that a false positive is a positive case of pneumonia, patients will be alarmed in vain. Patients will spend more time and more money doing unnecessary additional exams in order to further investigate their cases. Some patients will even take unnecessary medication.

Patients with diseases comorbid with pneumonia can trigger false positives as well. And if doctors also misdiagnose pneumonia, perhaps patients with other diseases will wrongfully take medication for pneumonia.

How false negatives might affect a patient?

False negatives could cloud doctors' judgement. They could skip a real case of pneumonia, which could be fatal for patients who really have pneumonia.

### 2. Algorithm Design and Function

Basically, the features learned by a VGG16 convnet were transferred from the more general domain of ImageNet images to the specific domain of chest X-ray images. The original VGG16 convnet was trained with photos of many kinds of objects. It has 1000 categories. Whereas the new convnet lacks the last layer and adds 3 new layers. The new convnet was trained to discriminate between chest X-Ray images of patients with and without pneumonia. So, it has 2 categories: Pneumonia and non-pneumonia.

<p align='center'><img src='images/diagram.png' width='50%'/></p>

The function of this device could be summarize in these steps. First, a patient has some kind of breathing difficulty. Then, the patient goes to the hospital. Doctors take a chest X-ray of the patient. Then, the chest X-ray DICOM file is fed into the CNN of this device. This device suggests a diagnosis in 1 second: Pneumonia or non-pneumonia. This computerized diagnosis helps doctors to finally make a **Computer Aided Diagnosis (CAD)**, which is a powerful synergy between doctors and artificial intelligence. This computerized help also seeks to prevent physician burnout. Sometimes doctors are overwhelmed by work.

<p align='center'><img src='images/cad_diagram.png'/></p>

**DICOM Checking Steps:**

Doctors only need to take a chest X-ray. The machine will generate a DICOM file like this:

```
test1.dcm

(0008, 0016) SOP Class UID                       UI: Secondary Capture Image Storage
(0008, 0018) SOP Instance UID                    UI: 1.3.6.1.4.1.11129.5.5.110503645592756492463169821050252582267888
(0008, 0060) Modality                            CS: 'DX'
(0008, 1030) Study Description                   LO: 'No Finding'
(0010, 0020) Patient ID                          LO: '2'
(0010, 0040) Patient's Sex                       CS: 'M'
(0010, 1010) Patient's Age                       AS: '81'
(0018, 0015) Body Part Examined                  CS: 'CHEST'
(0018, 5100) Patient Position                    CS: 'PA'
(0020, 000d) Study Instance UID                  UI: 1.3.6.1.4.1.11129.5.5.112507010803284478207522016832191866964708
(0020, 000e) Series Instance UID                 UI: 1.3.6.1.4.1.11129.5.5.112630850362182468372440828755218293352329
(0028, 0002) Samples per Pixel                   US: 1
(0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
(0028, 0010) Rows                                US: 1024
(0028, 0011) Columns                             US: 1024
(0028, 0100) Bits Allocated                      US: 8
(0028, 0101) Bits Stored                         US: 8
(0028, 0102) High Bit                            US: 7
(0028, 0103) Pixel Representation                US: 0
(7fe0, 0010) Pixel Data                          OW: Array of 1048576 elements
```

If the DICOM file that was taken by the X-ray machine has similar data, everything should work correctly. Modality should be 'DX'. Body Part Examined should be 'CHEST'. Patient Position should be either 'PA' or 'AP'. Photometric Interpretation should be MONOCHROME. Brightness levels should be in the range [0,1]. Other color spaces are not supported.

**Preprocessing Steps:**

Basically, the image from the field `Pixel Data` should be resized to match the following dimensions: `IMG_SIZE = (1, 224, 224, 3)`. Where `1` means the batch size of `1` image. `3` means the RGB color space. The image of `Pixel Data` should be transformed from grayscale to the RGB colorspace. And `224, 224` is the input size of the convolutional neural network capable of recognizing the patterns of pneumonia and non-pneumonia.

**CNN Architecture:**

Basically, the CNN architecture is VGG16 with pretrained weights whose last layer was removed and 3 new fully-connected layers were added:

```
    new_model.add(Dense(1024 * 2, activation='relu'))
    new_model.add(Dropout(0.25))
    new_model.add(Dense(1024, activation='relu'))
    new_model.add(Dropout(0.25))
    new_model.add(Dense(1, activation='sigmoid'))
```

Here is a summary of the CNN architecture:

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
dense_1 (Dense)              (None, 2048)              8390656   
_________________________________________________________________
dropout_1 (Dropout)          (None, 2048)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              2098176   
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 1025      
=================================================================
Total params: 144,750,401
Trainable params: 10,489,857
Non-trainable params: 134,260,544
```

### 3. Algorithm Training

**Parameters:**
* Types of augmentation used during training: 
    ```
    rescale = 1. / 255,
    horizontal_flip = True, 
    vertical_flip = False, 
    height_shift_range = 0.1, 
    width_shift_range = 0.1, 
    rotation_range = 20, 
    shear_range = 0.1, 
    zoom_range = 0.1
    ```
* Batch size: 64
* Optimizer learning rate: Adam optimizer and learning rate of 0.0001.
* Layers of pre-existing architecture that were frozen: 134,260,544 Non-trainable params
* Layers of pre-existing architecture that were fine-tuned: 10,489,857 Trainable params
* Layers added to pre-existing architecture:
    ```
        new_model.add(Dense(1024 * 2, activation='relu'))
        new_model.add(Dropout(0.25))
        new_model.add(Dense(1024, activation='relu'))
        new_model.add(Dropout(0.25))
        new_model.add(Dense(1, activation='sigmoid'))
    ```

**Algorithm training performance:**

**Learning curve: Loss versus epochs**

The best model was saved at the third epoch in which `val_loss=0.5457` and `val_binary_accuracy=0.9375` (93.75%)

<p align='center'><img src="images/loss_learn_curve.png" width='50%'/></p>

**Learning curve: Accuracy versus epochs**

The best model was saved at the third epoch in which `val_loss=0.5457` and `val_binary_accuracy=0.9375` (93.75%)

<p align='center'><img src="images/acc_learn_curve.png" width='50%'/></p>

**P-R curve (Precision versus Recall)**<br/>
<p align='center'><img src="images/PR.png" width='50%'/></p>

**AUC curve (Area Under Curve)**<br/>
The closer is the curve to the upper left corner, the better. AUC = 0.82. The closer AUC is to 1, the better.
<p align='center'><img src="images/AUC.png" width='50%'/></p>

**Final Threshold and Explanation:**

The API generated many tentative thresholds for the final activation. If `activation >= threshold`, then the classifier suggests pneumonia. Otherwise, the classifier suggest non-pneumonia. All tentative thresholds produce different values for the F1-score. The optimal threshold is the one that produces the maximal F1-score.

<p align='center'><img src="images/f1-score-plot.png" width='50%'/></p>

```
F1-scores:
[0.19354839 0.13333333 0.13793103 0.14285714 0.14814815 0.15384615
 0.16       0.16666667 0.17391304 0.18181818 0.19047619 0.2
 0.21052632 0.22222222 0.23529412 0.25       0.26666667 0.28571429
 0.30769231 0.33333333 0.36363636 0.4        0.22222222 0.25
 0.28571429        nan        nan        nan 0.        ]

Index is: 21
Optimal threshold: 0.4909
Maximum F1-score: 0.4000
Accuracy: 0.8906
```

### 4. Databases

The database used for training and validation is the file `Data_Entry_2017.csv`. This database is described in the paper:

ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases (by Xiaosong Wang et al)<br/>
https://arxiv.org/abs/1705.02315

Here are some visual examples found in such database. The classifier is somewhat accurate: 93.75% accuracy in the validation dataset. In the graph, `G` means ground truth and `P` means prediction. For example: `1G,1P` means 1 (pneumonia found) in ground truth and 1 pneumonia predicted by the classifier.

<p align='center'><img src="images/x-rays.png" width='50%'/></p>

**Description of Training Dataset and Validation Dataset:** 

Both the training dataset and the validation dataset were randomly sampled from the file `Data_Entry_2017.csv`, with 112,104 patients. This database is described in the paper:

ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases (by Xiaosong Wang et al)<br/>
https://arxiv.org/abs/1705.02315

Given the training dataset and the validation dataset were randomly sampled from this database of chest X-ray samples and there is no bias at all, it's perfectly reasonable to describe both the training dataset and the validation dataset **equally**. And after describing them, some subtle differences will be exposed.

**Gender of patients**

<p align='center'><img src='EDA_images/gender.png' width='50%'/></p>

This dataset is slightly biased toward male patients.

**Age of patients**

<p align='center'><img src='EDA_images/age.png' width='50%'/></p>

Patients in their 50s are the most common type of patient in this dataset.

**Distribution of Diseases**

<p align='center'><img src='EDA_images/diseases.png' width='50%'/></p>

No Finding (53.84%) is the most common finding, followed by Infiltration (17.74%) and Effusion (11.87%).
The most uncommon finding is Hernia (0.20%), followed by Pneumonia (1.28%), the disease we want to detect.

**Distribution of diseases that are comorbid with pneumonia**

<p align='center'><img src='EDA_images/comorbidities_pneumonia.png' width='50%'/></p>

The most common comorbidities that accompany pneumonia are Infiltration (42.27%) and Edema (23.75%).
Given the high correlation of these diseases with Pneumonia, the datasets should be balanced regarding these correlations. Random sampling helps to alleviate this kind of biases.

**Pneumonia cases versus non-pneumonia cases**

<p align='center'><img src='EDA_images/pneumonia_cases.png' width='50%'/></p>

Pneumonia cases are really rare, creating an unbalanced dataset.
Sampling should be done in a special way in order to overcome this unbalance.

**Number of diseases per patient**

<p align='center'><img src='EDA_images/number_diseases.png' width='50%'/></p>

Patients with no diseases are very common (53.84%) in this dataset.
Patients with 1 disease are almost 30% of the dataset.
Patients with 2 diseases are slightly above 10% of the dataset.
Patients with 3 diseases or more are rare.

**Number of follow-ups of patients**

<p align='center'><img src='EDA_images/follow-ups.png' width='50%'/></p>

In this dataset, most patients have few follow-ups.

Now, some subtle differences between both datasets will be exposed.

**Description of Training Dataset:**

The whole dataset in the file `Data_Entry_2017.csv` has 112,104 patients. It has 1,431 pneumonia cases (1.28%) and 110,689 non-pneumonia cases (98.72%).

Due to the very imbalance nature of the whole dataset, the training dataset was rebalanced with a proportion of 1:1:

```
1144 (pneumonia train) + 1144 (non_pneumonia train) = 2288 (all train)
```

From a total of 1,431 pneumonia cases, 1,144 pneumonia cases (80% of the pneumonia cases) were randomly sampled (without repetition) and put in the training dataset.<br/>
From a total of 110,689 non-pneumonia cases, 1,144 non-pneumonia cases were randomly sampled (without repetition) and put in the training dataset.

The training dataset has many augmentations:

```
rescale = 1. / 255,
horizontal_flip = True, 
vertical_flip = False, 
height_shift_range = 0.1, 
width_shift_range = 0.1, 
rotation_range = 20, 
shear_range = 0.1, 
zoom_range = 0.1
```

**Description of Validation Dataset:**

Due to the very imbalance nature of the whole dataset, 1,431 pneumonia cases (1.28%) and 110,689 non-pneumonia cases (98.72%), the validation dataset was rebalanced with a proportion of 1:10:

```
287 (pneumonia val) + 2870 (non_pneumonia val) = 3157 (all val)
```

From a total of 1,431 pneumonia cases, 287 pneumonia cases (20% of the pneumonia cases) were randomly sampled (without repetition) and put in the validation dataset.<br/>
From a total of 110,689 non-pneumonia cases, 2870 non-pneumonia cases were randomly sampled (without repetition) and put in the validation dataset.

The validation dataset has no augmentations, except the normalization:

```
rescale = 1. / 255
```

### 5. Ground Truth

The **gold standard** for detecting pneumonia in chest X-ray images is to send a biopsy to the laboratory. This method is super accurate to consider it ground truth. But it is more expensive and slower.

The **silver standard** for detecting pneumonia in chest x-ray images is to make some experts vote with their diagnoses. Each expert has a different weight depending on his/her experience. Another method is to extract diagnoses from text sources via NLP algorithms. These methods are less accurate, cheaper, and faster.

Ideally, ground truth should be created using the gold standard. However, the silver standard is often used due to the limited availability of resources.

For more information about how the dataset with ground truth was created, please read the following paper:<br/>
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases (by Xiaosong Wang et al)<br/>
https://arxiv.org/abs/1705.02315

If you read that paper, you will notice that both the gold standard (biopsy to laboratory) and the silver standard (diagnoses of experts and NLP text-mining) were applied to label the dataset used to train and to validate this model.

### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

The FDA Validation Dataset has 3 patients.<br/>
Their ages are 58, 61, and 81 years.<br/>
The 3 patients are male. There is no female patient.<br/>
From all 6 exams:
- 5 exams are DX and 1 exam is CT;
- 4 exams show no finding; 1 exam show cardiomegaly; and 1 exam show effusion;
- 5 exams took an image of the chest and 1 exam took an image of the ribcage;
- 3 exams has the PA position; 2 exams has the AP position; and 1 exam has XX position.

**Ground Truth Acquisition Methodology:**

The **gold standard** for detecting pneumonia in chest X-ray images is to send a biopsy to the laboratory. This method is super accurate to consider it ground truth. But it is more expensive and slower.

The **silver standard** for detecting pneumonia in chest x-ray images is to make some experts vote with their diagnoses. Each expert has a different weight depending on his/her experience. Another method is to extract diagnoses from text sources via NLP algorithms. These methods are less accurate, cheaper, and faster.

Ideally, ground truth should be created using the gold standard. However, the silver standard is often used due to the limited availability of resources.

The test dataset (FDA Validation Dataset) most probably uses the gold standard (biopsy to laboratory) to create ground truth labels. And perhaps in some cases, it probably uses the silver standard (diagnoses of experts with many years of experience). At this level, the most probable method is a mixture of both gold standard and silver standard, which gives even more confidence to the ground-truth labels.

**Algorithm Performance Standard:**

In summary, the model was trained and it produced the best performance at the third epoch in which `val_loss=0.5457` and `val_binary_accuracy=0.9375 (93.75%)`. The best model was saved in the third epoch. 

Neural networks are continuous interpolators. In other words, their outputs are not binary: 0 and 1. Their outputs are rather a number between the range [0,1]. The metric `val_binary_accuracy` has a threshold of 0.5. When the neural network activation is greater than 0.5, the model diagnoses pneumonia. Otherwise, the model diagnoses non-pneumonia.

However, the value 0.5 is not the best threshold possible. Many thresholds were examined and the best threshold `0.4909` produced an F1-score of `0.40` and an accuracy of `0.8906 (89.06%)`.

If you read this ground-breaking paper "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning" (by Andrew Ng et al) <https://arxiv.org/pdf/1711.05225.pdf>, you will notice that the average F1-score of radiologists is `0.387`, which is below the F1-score of `0.40` of this model. This means the performance of this device is reasonably good to be accepted by the FDA as a CAD system (Computer Aided Diagnosis).

<p align='center'><img src='images/paper_f1-scores.png' width='50%'/></p>

Moreover, the average time this device takes to diagnose pneumonia is just `1.01` seconds, in a normal computer without GPUs, which is many orders of magnitude faster than the average time human radiologists take, i.e. many minutes per radiography.

```
test1.dcm, Study description: No Finding, ground_truth=False, prediction=False (CORRECT), time=1.12 seconds
test2.dcm, Study description: Cardiomegaly, ground_truth=False, prediction=False (CORRECT), time=0.96 seconds
test3.dcm, Study description: Effusion, ground_truth=False, prediction=True (WRONG), time=0.95 seconds
test4.dcm, This DICOM file is INVALID: Body Part Examined = RIBCAGE, PatientPosition = PA, Modality = DX
test5.dcm, This DICOM file is INVALID: Body Part Examined = CHEST, PatientPosition = PA, Modality = CT
test6.dcm, This DICOM file is INVALID: Body Part Examined = CHEST, PatientPosition = XX, Modality = DX
ACCURACY=66.67% (2/3)
Average time: 1.01 seconds
```

The accuracy is decent: `66.67%`. The only mistake this device committed in the FDA Validation Dataset was a patient with Effusion, a comorbidity highly correlated with pneumonia. It is well known that the bigger the correlation of the comorbidity, the more confusion will cause in this device. Hence, a good recommendation to doctors is to pay special attention to the automated diagnoses of patients with comorbidities highly correlated with pneumonia.
