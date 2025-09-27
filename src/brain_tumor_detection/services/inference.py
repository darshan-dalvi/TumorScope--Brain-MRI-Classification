import io
import numpy as np
from PIL import Image
import onnxruntime as ort

# Load ONNX model once globally
import os

# Get the absolute path to the models directory
# Calculate path to models directory from current file
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
models_dir = os.path.join(src_dir, "models")
model_path = os.path.join(models_dir, "BrainTumor.onnx")

# Load ONNX model with absolute path
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# ✅ Function to preprocess the image correctly


def preprocess_image(file_bytes: bytes):
    try:
        # Open and convert image to RGB (3 channels)
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image = image.resize((150, 150))  # Resize to 150x150
        image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize
        image_np = np.expand_dims(image_np, axis=0)  # (1, 150, 150, 3)
        return image_np
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")

# ✅ Prediction function


def predict(file):
    try:
        # 👉 FastAPI UploadFile object
        image_bytes = file.file.read() if hasattr(file, "file") else file

        input_tensor = preprocess_image(image_bytes)
        result = session.run(None, {input_name: input_tensor})
        prediction = np.argmax(result[0])
        confidence = float(np.max(result[0]))

        # Comprehensive tumor information
        tumor_details = {
            0: {  # Glioma Tumor
                "name": "Glioma Tumor",
                "type": "Primary Brain Tumor",
                "severity": "High",
                "description": "Gliomas are tumors that grow from glial cells, which support nerve cells in the brain. They are the most common type of primary brain tumor in adults.",
                "symptoms": [
                    "Persistent headaches",
                    "Seizures",
                    "Changes in personality or behavior",
                    "Weakness or numbness in limbs",
                    "Vision or speech problems",
                    "Memory issues"
                ],
                "subtypes": ["Astrocytoma", "Oligodendroglioma", "Glioblastoma"],
                "treatment_options": [
                    "Surgical resection",
                    "Radiation therapy",
                    "Chemotherapy",
                    "Targeted therapy",
                    "Clinical trials"
                ],
                "prognosis": "Variable depending on grade and location",
                "urgency": "Immediate medical attention required",
                "next_steps": [
                    "Consult with neurosurgeon",
                    "MRI with contrast",
                    "Possible biopsy",
                    "Multidisciplinary team consultation"
                ]
            },
            1: {  # Meningioma Tumor
                "name": "Meningioma Tumor",
                "type": "Primary Brain Tumor",
                "severity": "Low to Moderate",
                "description": "Meningiomas arise from the meninges, the protective membranes surrounding the brain and spinal cord. Most are benign but can cause symptoms due to pressure.",
                "symptoms": [
                    "Gradual onset headaches",
                    "Visual disturbances",
                    "Hearing problems",
                    "Memory loss",
                    "Weakness in arms or legs",
                    "Seizures (less common)"
                ],
                "subtypes": ["Grade I (Benign)", "Grade II (Atypical)", "Grade III (Malignant)"],
                "treatment_options": [
                    "Observation (for small, asymptomatic tumors)",
                    "Surgical removal",
                    "Stereotactic radiosurgery",
                    "Conventional radiation therapy"
                ],
                "prognosis": "Generally good, especially for benign types",
                "urgency": "Scheduled consultation recommended",
                "next_steps": [
                    "Neurological examination",
                    "Follow-up MRI",
                    "Neurosurgical evaluation",
                    "Discussion of treatment options"
                ]
            },
            2: {  # No Tumor
                "name": "No Tumor Detected",
                "type": "Normal",
                "severity": "None",
                "description": "No signs of brain tumor detected in the MRI scan. The brain tissue appears normal without abnormal growths or lesions.",
                "symptoms": [],
                "subtypes": [],
                "treatment_options": [],
                "prognosis": "Normal brain tissue",
                "urgency": "No immediate action required",
                "next_steps": [
                    "Continue regular health checkups",
                    "Monitor for any new symptoms",
                    "Maintain healthy lifestyle",
                    "Follow up with physician as needed"
                ]
            },
            3: {  # Pituitary Tumor
                "name": "Pituitary Tumor",
                "type": "Primary Brain Tumor",
                "severity": "Low to Moderate",
                "description": "Pituitary adenomas are tumors of the pituitary gland. Most are benign and can affect hormone production, leading to various endocrine symptoms.",
                "symptoms": [
                    "Vision problems (especially peripheral vision)",
                    "Hormonal imbalances",
                    "Irregular menstrual periods",
                    "Unexplained weight gain/loss",
                    "Fatigue and weakness",
                    "Mood changes"
                ],
                "subtypes": ["Functioning adenomas", "Non-functioning adenomas", "Microadenomas", "Macroadenomas"],
                "treatment_options": [
                    "Medication (dopamine agonists)",
                    "Transsphenoidal surgery",
                    "Radiation therapy",
                    "Hormone replacement therapy"
                ],
                "prognosis": "Generally excellent with appropriate treatment",
                "urgency": "Endocrine evaluation recommended",
                "next_steps": [
                    "Endocrinology consultation",
                    "Hormone level testing",
                    "Visual field testing",
                    "MRI with pituitary protocol"
                ]
            }
        }

        tumor_info = tumor_details[prediction]
        
        return {
            "class": tumor_info["name"],
            "confidence": confidence,
            "tumor_details": {
                "type": tumor_info["type"],
                "severity": tumor_info["severity"],
                "description": tumor_info["description"],
                "symptoms": tumor_info["symptoms"],
                "subtypes": tumor_info["subtypes"],
                "treatment_options": tumor_info["treatment_options"],
                "prognosis": tumor_info["prognosis"],
                "urgency": tumor_info["urgency"],
                "next_steps": tumor_info["next_steps"]
            },
            "recommendation": {
                "medical_disclaimer": "This AI analysis is for informational purposes only and should not replace professional medical diagnosis.",
                "action_required": tumor_info["urgency"],
                "confidence_level": "High" if confidence > 0.9 else "Moderate" if confidence > 0.7 else "Low"
            }
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
