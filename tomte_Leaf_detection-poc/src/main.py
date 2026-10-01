import gradio as gr 
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("model/diagnose_disease.pt")  # Load a pretrained YOLOv8 model    




def diagnose_disease(image):
    '''
    Diagnose plant disease from an image using a YOLOv8 model.
    Args:
        image (PIL.Image): The input image of the plant.
    Returns:
        annotated_frame (numpy.ndarray): The image with detected disease areas annotated.
        diagnosis_comment (str): A comment summarizing the diagnosis results.
    '''
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    results = model(image_rgb, conf=0.15) 
    annotated_frame = results[0].plot()
    classes = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    diagnosis_comment = generate_diagnosis_comment(classes, confs)
    return annotated_frame, diagnosis_comment

def generate_diagnosis_comment(classes, confs):
    '''
    Generate a diagnosis comment based on detected classes and their confidence scores.
    Args:
        classes (list): List of detected class IDs.
        confs (list): List of confidence scores corresponding to the detected classes.
    Returns:
        str: A comment summarizing the diagnosis results.
    '''
    if len(classes) == 0:
        return "The model did not detect any signs of disease."    
    
    class_names = {1: "اللفحة المتأخرة (Late Blight)", 0: "اللفحة المبكرة (Early Blight)"}
    summary = {}
    
    for cls_id, conf in zip(classes, confs):
        name = class_names.get(int(cls_id), "unknown disease")
        confidence = float(conf)

        if name not in summary:
            summary[name] = [confidence, 1]
        else:
            summary[name][1] += 1
            new_combined_conf = 1 - (1 - summary[name][0]) * (1 - confidence) # Combine confidences using the formula for independent events
            summary[name][0] = new_combined_conf

    comments = []
    for name, (confidence, count) in summary.items():
        percentage = int(confidence * 100)
        if percentage > 70:
            comments.append(f"High danger: {name} detected {count} spots with high confidence ({percentage}%).")
        elif percentage > 40:
            comments.append(f"Medium danger: {name} detected {count} spots with moderate confidence ({percentage}%).")
        else:
            comments.append(f"Minor concern: {name} detected {count} spots with low confidence ({percentage}%).")

    return "\n".join(comments)




app = gr.Interface(
    fn=diagnose_disease,
    inputs=gr.Image(type="pil", label="Plant photo"),
    outputs=[gr.Image(label="Diagnosis"), gr.Textbox(label="Diagnosis Comment")],
    title="Photo Diagnoser",
    description="Upload a plant photo to detect disease.",
    submit_btn="Diagnose",
    clear_btn="Clear",
    examples=[
        ["examples/tomato_test_1.jpg"],
        ["examples/tomato_test_2.jpg"],
        ["examples/tomato_test_3.jpg"],
    ],
)

app.launch()