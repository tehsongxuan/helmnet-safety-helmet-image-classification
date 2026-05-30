[TehSongXuan_HelmNet_Full_Code _2.html](https://github.com/user-attachments/files/28422062/TehSongXuan_HelmNet_Full_Code._2.html)


## Project Overview

This project develops a computer vision model to classify workplace images into two categories:

- **With Helmet**: workers wearing safety helmets
- **Without Helmet**: workers not wearing safety helmets

The project was designed around a practical workplace safety use case. In construction sites, industrial facilities, warehouses, and other operational environments, safety helmets are an important form of personal protective equipment. Manual monitoring can be time-consuming and inconsistent, especially when many workers or camera feeds are involved.

Using image classification techniques, this project explores how deep learning can support automated helmet detection and help safety teams identify potential non-compliance more efficiently.

## Business Context

Workplace safety monitoring often depends on supervisors, safety officers, or manual review of camera footage. This approach can be limited by human fatigue, delayed detection, inconsistent observation, and the practical difficulty of monitoring many locations at once.

A helmet detection model can support safety operations by flagging images where workers may not be wearing helmets. This does not replace human judgment, but it can act as a decision-support tool to help safety teams focus attention on higher-risk cases.

The key business concern in this project is not only whether the model is accurate overall, but whether it can detect the **Without Helmet** class well. In a safety context, missed **Without Helmet** cases are more serious than false alarms because they may represent unflagged safety risks.

## Project Objective

The objective of this project is to build and evaluate an image classification model that can identify whether a worker is wearing a safety helmet.

The project aims to demonstrate how computer vision and deep learning can be used to:

- Classify images into **With Helmet** and **Without Helmet**
- Compare different deep learning model architectures
- Evaluate model performance beyond accuracy
- Pay close attention to the minority and safety-critical class
- Translate technical results into business-focused recommendations
- Present the full workflow in a reproducible and stakeholder-friendly notebook

## Dataset

The dataset contains **4,125 images** across two classes:

- **With Helmet**: 3,161 images
- **Without Helmet**: 964 images

The dataset is imbalanced, with more **With Helmet** images than **Without Helmet** images. This is important because a model may achieve strong overall accuracy by performing well on the majority class, while still missing important minority-class cases.

The images include different workplace environments, lighting conditions, viewing angles, worker postures, and image quality levels. Some images are low-resolution or blurry, which makes the classification task more realistic but also more challenging.

## Methodology

This project follows a structured machine learning workflow.

### 1. Library Installation and Environment Setup

The notebook begins by installing and importing the required libraries for image processing, data handling, visualization, model development, and performance evaluation.

The project uses TensorFlow and Keras for deep learning, along with NumPy, Pandas, Matplotlib, Seaborn, OpenCV, and Scikit-learn. GPU availability was checked in Google Colab to support faster model training.

Random seeds were also set for Python, NumPy, and TensorFlow to improve reproducibility.

### 2. Data Loading

The image data and label data were loaded into the notebook. The image dataset was stored as a NumPy array, while the labels were stored in a CSV file.

The images were represented as RGB image arrays with a shape of **200 × 200 × 3**, where 200 × 200 represents the image height and width, and 3 represents the red, green, and blue colour channels.

### 3. Exploratory Data Analysis

Exploratory data analysis was performed to understand the dataset before modelling.

This included:

- Checking the total number of images
- Reviewing the class distribution
- Visualizing sample images from both classes
- Observing image quality, blurriness, lighting, and visual differences
- Identifying the class imbalance between **With Helmet** and **Without Helmet**

The EDA showed that the dataset is imbalanced, with **With Helmet** images forming the majority class. This made it important to use evaluation metrics such as recall, precision, F1-score, and confusion matrix, rather than relying only on accuracy.

### 4. Data Preprocessing

The dataset was split into training, validation, and test sets using stratified splitting. Stratification helped preserve the original class distribution across all three datasets.

The split was:

- **Training set**: 2,887 images
- **Validation set**: 619 images
- **Test set**: 619 images

The images were normalized by scaling pixel values from the original 0-255 range to the 0-1 range. This helps the model train more efficiently and supports more stable learning.

### 5. Model Evaluation Criteria

Since this is a safety-related classification task with an imbalanced dataset, model evaluation was not based on accuracy alone.

The following metrics were used:

- **Accuracy**: overall percentage of correct predictions
- **Precision**: how many predicted positive cases were actually correct
- **Recall**: how many actual cases were correctly detected
- **F1-score**: balanced measure of precision and recall
- **Confusion matrix**: detailed view of correct and incorrect predictions for each class

For this project, recall for the **Without Helmet** class is especially important because missed detections may represent unflagged safety risks.

### 6. Model Building

Four models were developed and compared:

1. **CNN from Scratch**
2. **VGG16 Base**
3. **VGG16 Base + Feedforward Neural Network**
4. **VGG16 Base + Feedforward Neural Network + Data Augmentation**

The CNN from Scratch model was built as a custom convolutional neural network to learn image features directly from the HelmNet dataset.

The VGG16 models used transfer learning, where a pre-trained image recognition model was adapted for the helmet classification task. Additional dense layers and data augmentation were also tested to explore whether they could improve performance.

### 7. Model Comparison

After training and evaluating the four models, their training and validation performances were compared using accuracy, recall, precision, and F1-score.

The **CNN from Scratch** achieved the strongest validation performance among the models:

- Validation accuracy: **0.9386**
- Validation recall: **0.8971**
- Validation precision: **0.9273**
- Validation F1-score: **0.9109**

Although VGG16 is a powerful pre-trained model, the VGG16-based approaches did not outperform the CNN from Scratch in this experiment. This may be because the dataset images were relatively small, low-resolution, and specific to the helmet classification task.

## Final Model Selection

Based on the validation results, the **CNN from Scratch** was selected as the final model.

This model achieved the strongest overall validation performance across accuracy, recall, precision, and F1-score. It also provided a strong baseline solution for helmet image classification.

## Final Test Performance

The selected final model was evaluated on the unseen test dataset.

The final CNN from Scratch model achieved:

- Test accuracy: **0.9192**
- Macro recall: **0.8731**
- Macro precision: **0.8967**
- Macro F1-score: **0.8840**

For the **With Helmet** class, the model achieved:

- Precision: **0.94**
- Recall: **0.96**
- F1-score: **0.95**

For the **Without Helmet** class, the model achieved:

- Precision: **0.86**
- Recall: **0.79**
- F1-score: **0.82**

The model performed strongly overall, especially for the **With Helmet** class. However, the lower recall for the **Without Helmet** class shows that some helmet non-compliance cases were still missed.

## Confusion Matrix Insights

The final model's test confusion matrix showed:

- **114** actual **Without Helmet** images were correctly predicted as **Without Helmet**
- **31** actual **Without Helmet** images were incorrectly predicted as **With Helmet**
- **455** actual **With Helmet** images were correctly predicted as **With Helmet**
- **19** actual **With Helmet** images were incorrectly predicted as **Without Helmet**

The 31 missed **Without Helmet** cases are important because they represent possible helmet non-compliance cases that the model failed to detect. In a safety monitoring setting, these missed cases are more serious than false alarms.

## Key Data Science and AI Concepts Applied

- Computer vision
- Image classification
- Deep learning
- Convolutional Neural Networks
- Transfer learning
- VGG16
- Feedforward Neural Networks
- Data augmentation
- Train-validation-test split
- Stratified sampling
- Image normalization
- Class imbalance handling
- Accuracy, precision, recall, and F1-score
- Confusion matrix interpretation
- Minority-class evaluation
- Model comparison
- Final model selection
- Business-focused model evaluation

## Business Insights

This project shows that computer vision can support workplace safety monitoring by helping detect whether workers are wearing safety helmets.

Several key insights emerged:

- Overall accuracy alone is not enough for safety-related model evaluation.
- The **Without Helmet** class is the safety-critical class and should receive closer attention.
- Missed **Without Helmet** detections are more serious than false alarms.
- Class imbalance can affect model performance and interpretation.
- A simpler CNN from Scratch can sometimes outperform transfer learning models, depending on the dataset and task.
- Image quality, camera angle, lighting, and helmet visibility strongly affect prediction reliability.
- Human review remains important for uncertain or borderline cases.

## Practical Recommendations

Based on the project findings, the following recommendations are proposed:

1. **Collect more Without Helmet images**  
   The dataset contains fewer **Without Helmet** images. Collecting more examples from different environments, angles, distances, lighting conditions, and worker positions may help the model detect helmet non-compliance more reliably.

2. **Improve image quality where possible**  
   Some images were blurry or unclear. Better camera placement, higher image resolution, improved lighting, and clearer viewing angles can improve model performance.

3. **Review the classification threshold**  
   The current model uses a probability threshold of 0.5. If the priority is to reduce missed **Without Helmet** cases, the threshold can be adjusted to make the model more sensitive to potential non-compliance.

4. **Explore class weighting or oversampling**  
   Since the **Without Helmet** class is the minority class, future training can give this class more importance through class weighting or oversampling.

5. **Fine-tune transfer learning models further**  
   Future work can explore unfreezing selected deeper VGG16 layers and training them with a small learning rate to better adapt the model to helmet-specific image patterns.

6. **Use human review for uncertain predictions**  
   Images with prediction probabilities close to the decision threshold should be flagged for manual review. This can improve safety and reduce over-reliance on automated predictions.

## Value of the Project

This project demonstrates my ability to:

- Build an end-to-end image classification workflow
- Perform exploratory data analysis on image data
- Handle imbalanced classification problems
- Build and compare CNN and transfer learning models
- Evaluate model performance using business-relevant metrics
- Focus on minority-class performance in a safety-critical use case
- Translate technical model results into practical recommendations
- Present machine learning work in a structured, reproducible, and professional notebook

## Tools and Techniques

- Python
- Google Colab
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- CNN from Scratch
- VGG16 Transfer Learning
- Data Augmentation
- Classification Report
- Confusion Matrix
- HTML / Markdown formatting for GitHub presentation

## Examiner Review

The following review was received for the submitted HelmNet project. The examiner's name has been intentionally omitted for privacy.

> This is an exceptional project that achieves a perfect score across every single evaluation criterion, demonstrating meticulous data handling, thorough exploratory data analysis, and strong reproducibility. Your comparative analysis of the custom CNN baseline against various VGG-16 iterations is highly analytic, and your final model selection is well-justified by a keen focus on critical minority-class metrics. The work concludes with practical, business-focused recommendations and is presented in a highly organized, professional notebook that is perfectly suited for both technical reviewers and business stakeholders.

## Conclusion

This project demonstrates how computer vision and deep learning can be applied to a practical workplace safety problem.

The final CNN from Scratch model achieved strong test performance and provided a useful baseline for helmet classification. However, the project also shows that safety-related AI systems should not be judged by accuracy alone. The **Without Helmet** class requires careful attention because missed detections may affect real-world safety outcomes.

Overall, HelmNet presents a structured, reproducible, and business-relevant computer vision workflow that connects technical model development with practical workplace safety decision support.
"""

path = Path("/mnt/data/HelmNet_GitHub_README.md")
path.write_text(readme, encoding="utf-8")
path
