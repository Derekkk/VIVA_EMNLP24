import requests
import openpyxl
from PIL import Image
import io
import json
import os


def ensure_folder_exists(folder_path):
    """
    Check if a folder exists at the given path, and if not, create it.

    Parameters:
    folder_path (str): The path to the folder to check or create.

    Returns:
    None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def read_excel_rows(filename):
    # Load the workbook
    workbook = openpyxl.load_workbook(filename, data_only=True)
    # sheet = workbook.active
    sheet = workbook["situation_2"]
    # Initialize result list
    result = []

    # Iterate through rows
    sample_id = 0
    for row in sheet.iter_rows(min_row=2, max_row=300, values_only=True):
        # Extract values from the first to fifth columns
        url = row[0]
        # image_name = f"R{sample_id}.jpg"
        image_name = row[1]

        if url is None or image_name is None:
            continue
        result.append({"url": url, "image_name": image_name})
        sample_id += 1

    return result


def download_and_convert_image(image_url, filename):
    # Send a GET request to the image URL
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

    response = requests.get(image_url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Check if the content type is an image
        if 'image' in response.headers['Content-Type']:
            # Use io.BytesIO for in-memory binary streams
            img_data = io.BytesIO(response.content)
            # Open the image using Pillow
            img = Image.open(img_data)

            # Check if image is already a JPEG
            if img.format == 'JPEG':
                with open(filename, 'wb') as file:
                    file.write(response.content)
            else:
                # Convert to JPEG
                rgb_img = img.convert('RGB')  # Convert to RGB in case it's a different mode (like RGBA or P)
                rgb_img.save(filename, 'JPEG')

            # print(f"Image successfully downloaded and saved as {filename}")
        else:
            raise Exception(f"The downloaded data is not an image: {filename}.")
    else:
        raise Exception(f"Failed to download the image{filename}. Status code: {response.status_code}")


if __name__ == "__main__":
    data = json.load(open("data/VIVA_annotation.json"))
    for sample in data:
        image_url = sample["image_url"]
        image_file_name = f"data/VIVA_images/" + sample["image_file"]
        if os.path.exists(f"data/VIVA_images/" + sample["image_file"]):
            continue
        print(image_file_name)
        try:
            download_and_convert_image(image_url, image_file_name)
        except Exception as e:
            print(e)
            continue
