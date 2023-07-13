"""CLI tool accepting image of bookshelf as input and populating provided directory with the contents of bookshelf."""
from PIL import Image
from roboflow import Roboflow
import typer
import json
import pytesseract
import os
import requests


CONFIDENCE_THRESHOLD = 0.3

@staticmethod
def sanitize_book_title(approx_book_title: str, google_books_api_key: str) -> str:
    approx_book_title = ' '.join(approx_book_title.split())
    if len(approx_book_title) == 0:
        sanitize_book_title.unknown_index += 1
        return f'unknown{sanitize_book_title.unknown_index}'

    google_books_url = f'https://www.googleapis.com/books/v1/volumes?q={approx_book_title}&key={google_books_api_key}'
    response = requests.get(google_books_url)
    if response.status_code != 200 or response.json()['totalItems'] == 0:
        sanitize_book_title.unknown_index += 1
        return f'unknown{sanitize_book_title.unknown_index}'

    book_title = response.json()['items'][0]['volumeInfo']['title']
    if len(response.json()['items'][0]['volumeInfo']['authors']) > 0:
        return book_title + ' by ' + response.json()['items'][0]['volumeInfo']['authors'][0]

    return book_title


sanitize_book_title.unknown_index = 0


@staticmethod
def infer_on_image(image_path: str, roboflow_api_key: str) -> json:
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace().project("bookshelf-detector")
    model = project.version(1).model
    return model.predict(image_path, confidence=40, overlap=30).json()


def main(input_file: str, output_dir: str, google_api_key_file: str = '.google_api_key', roboflow_api_key_file: str = '.roboflow_api_key'):
    """
    Accepts photo of bookshelf located at input_file and saves images of books 
    named with title and author in output_dir.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(input_file)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(output_dir)
    if not os.path.isfile(google_api_key_file):
        raise FileNotFoundError(google_api_key_file)
    if not os.path.isfile(roboflow_api_key_file):
        raise FileNotFoundError(roboflow_api_key_file)
    
    roboflow_api_key = open(roboflow_api_key_file).read().split('\n')[0]
    google_books_api_key = open(google_api_key_file).read().split('\n')[0]
    inference_results = infer_on_image(input_file, roboflow_api_key)
    image = Image.open(input_file)
    for prediction in inference_results['predictions']:
        if prediction['confidence'] < CONFIDENCE_THRESHOLD:
            continue

        left = prediction['x'] - int(prediction['width']/2)
        right = prediction['x'] + int(prediction['width']/2)
        top = prediction['y'] - int(prediction['height']/2)
        bottom = prediction['y'] + int(prediction['height']/2)
        book_spine = image.crop((left, top, right, bottom))

        rotated = book_spine.transpose(Image.ROTATE_90)
        book_title_raw = pytesseract.image_to_string(rotated)
        book_title = sanitize_book_title(book_title_raw, google_books_api_key)

        book_spine.save(f'{output_dir}/{book_title}.jpg')


if __name__ == "__main__":
    typer.run(main)
