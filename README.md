# Pneumonia Diagnosis Classifier Using CNN
### Note: X-ray images folder is not uploaded due to large file size.

## Project Overview
This project detects pneumonia from chest X-ray images using a **Convolutional Neural Network (CNN)**. The system classifies whether a patient has **Pneumonia** or is **Normal**.  
The trained model is saved as `pneumonia_cnn_model.h5` for future predictions.

---

## Workflow
1. **Data Preprocessing**  
   - Images are resized, normalized, and augmented to improve generalization.  

2. **Model Architecture**  
   - CNN with multiple convolutional and pooling layers.  

3. **Training**  
   - Model trained on training data and validated on unseen data.  

4. **Evaluation**  
   - Accuracy, loss, and confusion matrix analyzed.  

5. **Prediction**  
   - Saved model (`pneumonia_cnn_model.h5`) is loaded to predict new X-ray images.  

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Streamlit (for UI)  

---

## Model Details
- The CNN consists of several **convolutional and pooling layers**, followed by **fully connected dense layers**.  
- The final activation function is **Sigmoid** for binary classification (**Normal vs Pneumonia**).

---

## Results
- The model achieved **high accuracy** in distinguishing pneumonia from normal cases.  
- Evaluation metrics and graphs (accuracy vs epochs, loss vs epochs) are included in the notebook.

---

## Deployment
- Integrated with **Streamlit** to allow users to **upload X-ray images** and receive instant predictions.  
- Outputs are displayed with **labels (Normal / Pneumonia)** along with **confidence scores**.

---

## Limitations & Future Work
- Dataset bias may affect accuracy.  
- Training on larger datasets can improve results.  
- Adding **explainability techniques** like Grad-CAM would enhance trust in predictions.
