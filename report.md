# Project Report — Tomato Healthy vs YLCV Classifier (Group EE8)

The dataset used was the PlantVillage Dataset (sourced from Kaggle),
narrowed down to the "Healthy" and "Yellow Leaf Curl Virus" tomato leaf
classes. Images were split into training, validation, and test sets, then
used to train a custom CNN and a MobileNetV3Small transfer learning model.
The transfer learning model performed best, reaching 100% accuracy,
precision, recall, and F1-score on the 98-image test set, and was deployed.
To use the application, a user uploads a photo of a tomato leaf and
receives an instant "Healthy" or "YLCV" prediction with confidence scores.
The main challenges were initially training on a CPU runtime before
switching to a GPU, specifically the T4 GPU, for practical training times, and the near-perfect test
accuracy, which reflects PlantVillage's controlled, consistent image
conditions rather than guaranteed real-world field performance. A possible
improvement would be testing on field-captured images.

The contributors to this project will be referenced below:

• Ogosi Francis Chukwuemeka 23/EG/EE/003 PrinceMC35

• Bamidele Marvellous Akachukwu 24/EG/EE/253 bamidelem300-commits

• Onobo Benjamin Joseph 23/EG/EE/073 Benjamin324-ai

• Iheasimuo Izuchukwu Chamberlain 23/EG/EE/023 BhigBloggs 

• Okon Stephen Mfon 23/EG/EE/063 Stephen933971

• Makoji Goodness Enyo-ojo 23/EG/EE/093
enyofx-rgb
