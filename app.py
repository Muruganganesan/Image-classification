import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

# Set dataset path
DATA_DIR = r'C:\Users\admin\Music\Guvi\Image Classification\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\fish_dataset'

st.title("🐟 Fish Image Classification")
st.write("Train a model to classify fish images.")

if st.button("📈 Start Training"):
    transform = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform['train'])
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    num_epochs = 5
    train_losses = []
    train_accuracies = []

    st.write("🛠 Training started...")
    progress = st.progress(0)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        st.write(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        progress.progress((epoch + 1) / num_epochs)

    st.success("✅ Training completed!")
    torch.save(model.state_dict(), 'fish_classification_resnet18.pth')

    # Evaluate
    st.subheader("📊 Evaluation on Validation Data")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    st.write("### Classification Report")
    st.dataframe(report)

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Plot training loss and accuracy
    st.write("### Training Metrics")
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, num_epochs+1), train_losses, label='Loss')
    ax2.plot(range(1, num_epochs+1), train_accuracies, label='Accuracy')
    ax2.set_xlabel("Epoch")
    ax2.set_title("Loss and Accuracy")
    ax2.legend()
    st.pyplot(fig2)


# ================================
# 🔍 Image Upload & Prediction UI
# ================================

st.header("🔎 Upload a Fish Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load model (with same class count)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load('fish_classification_resnet18.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_class = torch.max(probabilities, 0)
        predicted_label = class_names[predicted_class.item()]

    st.success(f"🎯 Prediction: **{predicted_label}**")

    # Show class probabilities
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(probabilities):
        st.write(f"{class_names[i]}: {prob.item():.4f}")
