import json
import firebase_admin
from firebase_admin import credentials, storage, firestore

# Initialize Firebase
cred = credentials.Certificate(
    "tarotclick-bf393-firebase-adminsdk-t2h0q-1b2c40a880.json"
)
firebase_admin.initialize_app(cred, {"storageBucket": "tarotclick-bf393.appspot.com"})
bucket = storage.bucket()
db = firestore.client()


# Function to upload image and return the public URL
def upload_image(image_path, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(image_path)
    blob.make_public()  # Make the image publicly accessible
    return blob.public_url


# Function to add tarot card document to Firestore
def add_tarot_card(card_data, image_url, id):
    card_ref = db.collection("tarot_cards").document(id)
    card_ref.set(
        {
            "id": id,
            "name": card_data["name"],
            "number": card_data["number"],
            "arcana": card_data["arcana"],
            "suit": card_data["suit"],
            "fortune_telling": card_data["fortune_telling"],
            "keywords": card_data["keywords"],
            "meanings": card_data["meanings"],
            "image_url": image_url,
            "elemental": card_data.get("Elemental", ""),
            "astrology": card_data.get("Astrology", ""),
            "questions": card_data["Questions to Ask"],
        }
    )


# Read the card data from the JSON file
with open("card_data.json", "r") as f:
    card_data_list = json.load(f)

# Process each card in the JSON file
for i, card_data in enumerate(card_data_list):
    image_path = "cards/" + card_data["img"]
    destination_blob_name = f"tarot_card_images/{card_data['arcana'].lower().replace(' ', '_')}/{card_data['name'].lower().replace(' ', '_')}.jpg"
    image_url = upload_image(image_path, destination_blob_name)
    add_tarot_card(card_data, image_url, str(i))
    print(f"Uploaded {card_data['name']} and added to Firestore with URL {image_url}")

print("All cards have been processed.")
