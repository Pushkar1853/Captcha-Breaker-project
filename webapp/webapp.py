import matplotlib.image as mpimg
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model
model = torch.load('\model.pt')
model.eval()

# Define the image transformation to normalize the image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# def encode_targets():
#   # Encode images
#   lbl_enc = LabelEncoder()
#   # lbl_enc.fit(targets_flat)
#   # targets_enc = [lbl_enc.transform(x) for x in targets]
#   # targets_enc = np.array(targets_enc) + 1 # transform to np and remove 0 index
#
#   return lbl_enc

lbl_enc = LabelEncoder()

def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j,:]:
            k = k - 1
            if k == -1:
                temp.append("-")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_preds.append(tp)
    return cap_preds

def predict_function(model, data):
    model.eval()
    fin_preds = []
    with torch.no_grad():
        # for data in data_loader:
        for k, v in data.items():
            data[k] = v.to(DEVICE)
        batch_preds, _ = model(**data)
        fin_preds.append(batch_preds)
    return fin_preds

def clean_decoded_predictions(unclean_predictions):
    cleaned_predictions = []
    for i in unclean_predictions:
        if i != "-":
            cleaned_predictions.append(i)
    cleaned_predictions = "".join(cleaned_predictions)
    if len(cleaned_predictions) == 10:
        return cleaned_predictions
    else:
        prev = "-"
        new_cleaned_predictions = []
        for char in cleaned_predictions:
            if char == prev:
                continue
            new_cleaned_predictions.append(char)
            prev = char
        res = "".join(new_cleaned_predictions)
        return res

def predict_captcha(model, image_path):
    plt.figure(figsize=(15, 5))
    image = mpimg.imread(image_path[0])
    # target = image_path[0].split("/")[-1].split(".")[0]
    plt.title(image_path[0].split("/")[-1])
    plt.imshow(image)

    valid_preds = predict_function(model, image)
    current_preds = decode_predictions(valid_preds, lbl_enc)
    preds = clean_decoded_predictions(current_preds[0])
    # success = True if preds == target else False
    return preds

# Define the Streamlit app
def app():
    st.title("Captcha Breaker Project")
    st.write("by - Pushkar Ambastha")
    st.write("Upload an image of a captcha to recognize the text")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the image and transform it
        img = Image.open(uploaded_file)
        img = transform(img)

        # Make a prediction with the model
        with torch.no_grad():
            prediction = predict_captcha(model, img.unsqueeze(0))

        # Get the predicted text and display it
        captcha_text = "".join([chr(int(x)) for x in prediction])
        st.write(f"The captcha text is: {captcha_text}")

