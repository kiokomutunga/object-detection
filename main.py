import cv2
import numpy as np
import wikipediaapi
from googlesearch import search

#function to capture the image

def capture_image (output_path = "catured.jpg"):
    """use the webcam for the image"""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cannot access camera")
        return None
    
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(output_path, frame)
        return output_path
    return None


def detect_keypoints(image_path, reference_path="reference.jpg"):
    """Detect keypoints in the image and compare with a reference image."""
    img1 = cv2.imread(reference_path, 0)  # Reference object image
    img2 = cv2.imread(image_path, 0)      # Captured image
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) > 10:
        return "Object Matched"  # Replace with object name if known
    return "Unknown Object"

def google_search(query):
    """Perform a Google search and return the first result."""
    for result in search(query, num=1, stop=1, pause=2):
        return result
    return "No results found."

def get_wikipedia_summary(query):
    """Fetch Wikipedia description of the detected object."""
    wiki = wikipediaapi.Wikipedia("en")
    page = wiki.page(query)
    
    if page.exists():
        return page.summary[:500]  # First 500 characters
    return "No Wikipedia description found."

if __name__ == "__main__":
    image_path = capture_image()
    
    if image_path:
        object_name = detect_keypoints(image_path)
        
        if object_name != "Unknown Object":
            print(f"Detected Object: {object_name}")
            
            # Google search
            google_result = google_search(object_name)
            print(f"Google Search Result: {google_result}")

            # Wikipedia summary
            wiki_summary = get_wikipedia_summary(object_name)
            print(f"Description: {wiki_summary}")
        else:
            print("Object not recognized.")
