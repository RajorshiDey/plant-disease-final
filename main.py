import json
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# 1. PYDANTIC MODELS (JSON Schema)
class DiagnosticRequest(BaseModel):
    plant_id: int
    symptoms: List[int]

class DiagnosticResponse(BaseModel):
    name: str
    confidence: float
    remedy: str

# 2. LOADING DATASET (JSON)
def load_dataset():
    """Reads the JSON dataset and assigns IDs to plants and symptoms."""
    try:
        with open("dataset.json", "r") as file:
            raw_data = json.load(file)
    except FileNotFoundError:
        print("Error: dataset.json not found!")
        return [], [], []

    # Extracting unique plants and mapping them to IDs
    unique_plants = list(set([entry["plant"] for entry in raw_data]))
    plants = [{"id": i, "name": name} for i, name in enumerate(unique_plants)]

    # Extracting unique symptoms and mapping them to IDs
    unique_symptoms = list(set([sym for entry in raw_data for sym in entry["symptoms"]]))
    symptoms = [{"id": i, "description": desc} for i, desc in enumerate(unique_symptoms)]

    # Mapping the human-readable dataset to IDs for backend processing
    diseases = []
    for entry in raw_data:
        # Finding the ID for the plant
        plant_id = next(p["id"] for p in plants if p["name"] == entry["plant"])
        
        # Finding the IDs for the required symptoms
        symptom_ids = [next(s["id"] for s in symptoms if s["description"] == sym) for sym in entry["symptoms"]]
        
        diseases.append({
            "name": entry["disease"],
            "plant_id": plant_id,
            "required_symptoms": set(symptom_ids),
            "remedy": entry["remedy"]
        })
        
    return plants, symptoms, diseases

# Load into memory on startup
PLANTS, SYMPTOMS, DISEASES = load_dataset()

# 3. FASTAPI APP & ENDPOINTS
app = FastAPI()

#For frontend purpose
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/data")
async def get_form_data():
    """Sends the lists of plants and symptoms to the frontend."""
    return {"plants": PLANTS, "symptoms": SYMPTOMS, "diseases": DISEASES}

@app.get("/api/dataset")
async def get_full_dataset():
    """Serves the complete dataset for the frontend lookup tool."""
    import json
    try:
        with open("dataset.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

@app.post("/api/diagnose", response_model=Optional[DiagnosticResponse])
async def diagnose(request_data: DiagnosticRequest):
    """The JSON-backed Inference Engine."""
    observed_symptoms = set(request_data.symptoms)
    
    # Filter diseases to only those matching the selected plant
    possible_diseases = [d for d in DISEASES if d["plant_id"] == request_data.plant_id]
    
    best_match = None
    highest_confidence = 0.0
    
    # Mathematical biological condition matching
    for disease in possible_diseases:
        required = disease["required_symptoms"]
        
        if not required:
            continue
            
        # Intersect user inputs vs required symptoms
        matches = required.intersection(observed_symptoms)
        confidence = (len(matches) / len(required)) * 100
        
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_match = {
                "name": disease["name"],
                "confidence": confidence,
                "remedy": disease["remedy"]
            }
            
    return best_match

