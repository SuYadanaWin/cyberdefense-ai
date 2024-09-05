# cyberdefense-ai
This is the research project for hybrid AI for cyber defense and threat detection

implementation and evaluation explanation step by step


1. NSL-KDD Directory:
•KDDTest+_Binary_Classification.ipynb:
implements binary classification models to detect whether a network traffic instance is either normal or malicious (binary classification). It uses the NSL-KDD dataset, which is commonly used for network intrusion detection.
Key Implementation: Preprocessing the NSL-KDD dataset (like handling categorical features), applying algorithms such as Logistic Regression, SVM, or Decision Trees, and comparing their performance.
Evaluation: Models will be evaluated using accuracy, precision, recall, and F1-score.


•KDDTest+_Multiclass.ipynb:
the task to multiclass classification, where the models classify network traffic into several attack types like DoS, Probe, U2R, R2L.
Key Implementation: to discuss how different machine learning models are adapted for multiclass classification, such as softmax classifiers or using one-vs-rest approaches.
Evaluation: Models are evaluated using precision, recall, F1-score, and confusion matrices for each attack class.
KDDTest_21_Multiclass.ipynb:
similar to the previous multiclass notebook but focuses on a subset of the NSL-KDD dataset with 21 features (instead of the full set of features).
Key Implementation: Feature selection techniques could be important here, and you may need to discuss how reducing features affects model performance.
Evaluation: Comparisons of model performance on the reduced feature set versus the full feature set.
NSL_KDD_Test.csv, NSL_KDD_Test_21.csv, NSL_KDD_Train.csv:
the training and test datasets used in the above notebooks. The NSL_KDD_Train.csv contains training data, while NSL_KDD_Test.csv and NSL_KDD_Test_21.csv contain test data, with the latter likely focusing on the 21-feature subset.

2. UNSW Directory:

Exploratory Data Analysis.ipynb:
should explore the UNSW-NB15 dataset, which is another dataset used for intrusion detection. It might include:
Data visualization: Understanding the distribution of attacks and normal traffic, correlation between features, and initial data cleaning steps.
Evaluation: No direct model evaluation here, but understanding the data is crucial for the next steps.

Feature Engineering and Data Preparation.ipynb:
about transforming the raw data into a format suitable for machine learning algorithms. It includes steps like feature scaling, encoding categorical variables, and splitting data into training and test sets.
Key Implementation: Discuss feature selection methods, scaling techniques (such as standardization), and how these steps improve model performance.

ML Models and Results.ipynb:
the core of the machine learning implementation for the UNSW dataset. It includes training models such as Decision Trees, Random Forests, and Neural Networks.
Key Implementation: Model training and hyperparameter tuning for various algorithms.
Evaluation detailed evaluations using accuracy, confusion matrices, and possibly ROC-AUC curves for binary and multiclass classification tasks.



•Prediction of Raw Data.ipynb:
- seems to handle predictions on raw, unseen data. This could be a practical demonstration of how the trained models generalize to new traffic instances.
Key Implementation: Applying the trained models to real-world data.
Evaluation: Performance metrics, possibly demonstrating the model’s generalization ability beyond the test dataset.

•datapreprocessing.ipynb:
focuses on data preprocessing steps, such as handling missing values, normalization, and preparing the data for machine learning models.
Key Implementation: Discussion about the impact of preprocessing on model accuracy and robustness.

3. Adversarial Directory:
comparison_of_adversarial_attacks.ipynb:
compares the performance of models under different adversarial attacks. Adversarial attacks are small perturbations added to input data to fool machine learning models.
Key Implementation: Implementing attacks like FGSM (Fast Gradient Sign Method) and comparing the effect on the trained models.
Evaluation: discuss how the accuracy drops under these attacks and how adversarial training or other defense mechanisms mitigate these effects.

•evasion-attack.ipynb:
focuses on evasion attacks, where an adversary tries to bypass the defense mechanisms of the system.
Key Implementation: Creating adversarial samples that evade detection by altering network traffic patterns.
Evaluation: The success rate of the evasion attacks and how models defend against such threats.


iterative_adversarial_training_with_HSJA.ipynb:
implements iterative adversarial training, a method where a model is repeatedly trained on adversarial samples to become more robust.
Implementation: Implementing the HSJA (HopSkipJump Attack) and training models iteratively.
Evaluation: Improvements in model robustness and accuracy after adversarial training.

4. Webattack Directory:
web-attack-detection-using-CNN-BiLSTM.ipynb:
combines CNN (Convolutional Neural Networks) and BiLSTM (Bidirectional Long Short-Term Memory) models to detect web-based attacks. CNNs are typically used for feature extraction, while LSTMs are used for sequential data.
Key Implementation: Discussion of how CNN and LSTM architectures are combined, the role of each model, and how they process network traffic data.
Evaluation: includes metrics such as accuracy, precision, recall, and comparisons with other models.

web-attack-detection.ipynb:
This might be a simpler implementation focusing on detecting web-based attacks using machine learning or deep learning algorithms.
Key Implementation: Training models for specific web attack detection tasks.
Evaluation: Model evaluation with standard metrics like precision and recall.



