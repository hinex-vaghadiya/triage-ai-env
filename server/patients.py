"""
Patient templates and deterioration engine for TriageAI.
Generates realistic ER patients with hidden severity, symptoms, and vitals.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class Patient:
    id: str
    name: str
    age: int
    gender: str
    symptoms: str  # What agent sees
    chief_complaint: str  # Brief summary
    hidden_severity: int  # 1=critical, 5=stable (agent can't see)
    hidden_diagnosis: str  # Agent can't see
    required_treatment: str  # "surgery"|"medication"|"observation"|"discharge"
    time_to_deteriorate: int  # Steps before worsening
    current_severity: int  # Tracks real-time severity
    status: str = "waiting"  # waiting|in_bed|in_or|treated|discharged|dead
    assigned_bed: Optional[int] = None
    assigned_doctor: Optional[int] = None
    triage_level: Optional[int] = None
    treatment_given: Optional[str] = None
    steps_waiting: int = 0
    steps_since_last_deterioration: int = 0
    triage_count: int = 0
    examined: bool = False
    vitals: Dict[str, Any] = field(default_factory=dict)

    def to_visible_dict(self, detail_level: str = "basic") -> dict:
        """Return what the agent can see about this patient."""
        base = {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "chief_complaint": self.chief_complaint,
            "status": self.status,
            "vitals": self.vitals,
            "steps_waiting": self.steps_waiting,
        }
        if detail_level == "basic":
            base["symptoms"] = self.symptoms
        elif detail_level == "triaged":
            base["symptoms"] = self.symptoms
            base["triage_level"] = self.triage_level
        elif detail_level == "examined":
            base["symptoms"] = self.symptoms
            base["triage_level"] = self.triage_level
            base["examined"] = True
            base["clinical_notes"] = self._get_exam_notes()
        return base

    def _get_exam_notes(self) -> str:
        if self.current_severity <= 2:
            return f"CRITICAL: Patient shows signs consistent with {self.hidden_diagnosis}. Immediate intervention required."
        elif self.current_severity == 3:
            return f"URGENT: Symptoms suggest possible {self.hidden_diagnosis}. Close monitoring needed."
        else:
            return "STABLE: Vitals within acceptable range. Standard care appropriate."

    def deteriorate(self) -> bool:
        """Worsen patient condition. Returns True if patient died."""
        if self.status in ("treated", "discharged", "dead"):
            return False
        if self.current_severity <= 1:
            self.status = "dead"
            return True
        self.current_severity -= 1
        self._update_vitals_for_severity()
        return False

    def _update_vitals_for_severity(self):
        sev = self.current_severity
        if sev <= 1:
            self.vitals = {"hr": random.randint(130, 170), "bp": f"{random.randint(60,75)}/{random.randint(30,40)}", "spo2": random.randint(70, 80), "temp": round(random.uniform(35.0, 39.5), 1), "consciousness": "unresponsive"}
        elif sev == 2:
            self.vitals = {"hr": random.randint(110, 140), "bp": f"{random.randint(80,95)}/{random.randint(40,55)}", "spo2": random.randint(82, 90), "temp": round(random.uniform(36.0, 39.0), 1), "consciousness": "confused"}
        elif sev == 3:
            self.vitals = {"hr": random.randint(90, 115), "bp": f"{random.randint(95,120)}/{random.randint(55,70)}", "spo2": random.randint(90, 95), "temp": round(random.uniform(36.5, 38.5), 1), "consciousness": "alert"}
        elif sev == 4:
            self.vitals = {"hr": random.randint(75, 95), "bp": f"{random.randint(110,135)}/{random.randint(65,80)}", "spo2": random.randint(94, 98), "temp": round(random.uniform(36.5, 37.8), 1), "consciousness": "alert"}
        else:
            self.vitals = {"hr": random.randint(65, 85), "bp": f"{random.randint(115,130)}/{random.randint(70,85)}", "spo2": random.randint(96, 100), "temp": round(random.uniform(36.3, 37.2), 1), "consciousness": "alert"}


# =====================================================================
# Patient Template Pool (50+ templates)
# =====================================================================

PATIENT_TEMPLATES = [
    # CARDIAC (severity 1-2, critical)
    {"symptoms": "Crushing chest pain radiating to left arm, profuse sweating, nausea, shortness of breath", "chief_complaint": "Chest pain", "hidden_severity": 1, "hidden_diagnosis": "Acute Myocardial Infarction (Heart Attack)", "required_treatment": "surgery", "time_to_deteriorate": 3},
    {"symptoms": "Sudden onset palpitations, dizziness, irregular heartbeat, lightheadedness", "chief_complaint": "Heart palpitations", "hidden_severity": 2, "hidden_diagnosis": "Ventricular Tachycardia", "required_treatment": "medication", "time_to_deteriorate": 4},
    {"symptoms": "Severe chest tightness, unable to lie flat, swollen ankles, extreme fatigue, coughing pink frothy sputum", "chief_complaint": "Breathing difficulty", "hidden_severity": 1, "hidden_diagnosis": "Acute Heart Failure", "required_treatment": "medication", "time_to_deteriorate": 3},
    {"symptoms": "Sudden tearing chest pain radiating to back, sweating, feeling of impending doom", "chief_complaint": "Severe chest/back pain", "hidden_severity": 1, "hidden_diagnosis": "Aortic Dissection", "required_treatment": "surgery", "time_to_deteriorate": 2},

    # TRAUMA (severity 1-3)
    {"symptoms": "Deep laceration on forearm, heavy bleeding, visible muscle tissue, patient dizzy", "chief_complaint": "Arm laceration", "hidden_severity": 2, "hidden_diagnosis": "Deep Forearm Laceration with Arterial Bleed", "required_treatment": "surgery", "time_to_deteriorate": 4},
    {"symptoms": "Fall from height, severe headache, one pupil dilated, confusion, vomiting", "chief_complaint": "Head injury after fall", "hidden_severity": 1, "hidden_diagnosis": "Epidural Hematoma", "required_treatment": "surgery", "time_to_deteriorate": 3},
    {"symptoms": "Car accident, neck pain, tingling in both arms, difficulty moving legs", "chief_complaint": "Neck injury from car accident", "hidden_severity": 2, "hidden_diagnosis": "Cervical Spine Injury", "required_treatment": "observation", "time_to_deteriorate": 5},
    {"symptoms": "Twisted ankle while playing sports, swelling, pain on weight bearing, no deformity", "chief_complaint": "Ankle injury", "hidden_severity": 5, "hidden_diagnosis": "Ankle Sprain Grade 2", "required_treatment": "discharge", "time_to_deteriorate": 15},
    {"symptoms": "Motorcycle accident, obvious deformity of right thigh, severe pain, unable to move leg", "chief_complaint": "Leg injury from motorcycle accident", "hidden_severity": 2, "hidden_diagnosis": "Femur Fracture", "required_treatment": "surgery", "time_to_deteriorate": 5},

    # RESPIRATORY (severity 1-4)
    {"symptoms": "Severe wheezing, using accessory muscles to breathe, unable to speak full sentences, blue lips", "chief_complaint": "Asthma attack", "hidden_severity": 2, "hidden_diagnosis": "Severe Asthma Exacerbation", "required_treatment": "medication", "time_to_deteriorate": 3},
    {"symptoms": "High fever 39.5C, productive cough with green sputum, chest pain on breathing, fatigue for 5 days", "chief_complaint": "Fever and cough", "hidden_severity": 3, "hidden_diagnosis": "Community-Acquired Pneumonia", "required_treatment": "medication", "time_to_deteriorate": 6},
    {"symptoms": "Sudden shortness of breath after long flight, sharp chest pain on deep breath, calf swelling", "chief_complaint": "Sudden breathlessness", "hidden_severity": 2, "hidden_diagnosis": "Pulmonary Embolism", "required_treatment": "medication", "time_to_deteriorate": 3},
    {"symptoms": "Chronic cough worsened, increased sputum, mild shortness of breath on exertion, known COPD patient", "chief_complaint": "Worsening cough", "hidden_severity": 4, "hidden_diagnosis": "COPD Exacerbation", "required_treatment": "medication", "time_to_deteriorate": 8},

    # GI (severity 1-4)
    {"symptoms": "Severe right lower abdominal pain, nausea, fever 38.2C, pain worsens with movement, rebound tenderness", "chief_complaint": "Abdominal pain", "hidden_severity": 2, "hidden_diagnosis": "Acute Appendicitis", "required_treatment": "surgery", "time_to_deteriorate": 5},
    {"symptoms": "Vomiting blood, dark tarry stools, dizziness on standing, history of alcohol use", "chief_complaint": "Vomiting blood", "hidden_severity": 1, "hidden_diagnosis": "Upper GI Hemorrhage", "required_treatment": "surgery", "time_to_deteriorate": 3},
    {"symptoms": "Severe cramping abdominal pain, abdominal distension, unable to pass gas, vomiting", "chief_complaint": "Severe stomach pain", "hidden_severity": 2, "hidden_diagnosis": "Small Bowel Obstruction", "required_treatment": "surgery", "time_to_deteriorate": 5},
    {"symptoms": "Mild stomach ache after eating spicy food, some nausea, no fever, no vomiting", "chief_complaint": "Stomach ache", "hidden_severity": 5, "hidden_diagnosis": "Gastritis", "required_treatment": "discharge", "time_to_deteriorate": 15},

    # NEUROLOGICAL (severity 1-3)
    {"symptoms": "Sudden face drooping on left side, slurred speech, right arm weakness, onset 30 minutes ago", "chief_complaint": "Face drooping and arm weakness", "hidden_severity": 1, "hidden_diagnosis": "Acute Ischemic Stroke", "required_treatment": "medication", "time_to_deteriorate": 2},
    {"symptoms": "Witnessed seizure lasting 3 minutes, now confused and drowsy, no seizure history", "chief_complaint": "Seizure", "hidden_severity": 2, "hidden_diagnosis": "New-Onset Seizure", "required_treatment": "observation", "time_to_deteriorate": 5},
    {"symptoms": "Worst headache of life, sudden onset, stiff neck, sensitivity to light, nausea", "chief_complaint": "Severe headache", "hidden_severity": 1, "hidden_diagnosis": "Subarachnoid Hemorrhage", "required_treatment": "surgery", "time_to_deteriorate": 3},
    {"symptoms": "Throbbing headache on one side, visual aura, nausea, sensitivity to light and sound", "chief_complaint": "Headache", "hidden_severity": 4, "hidden_diagnosis": "Migraine with Aura", "required_treatment": "medication", "time_to_deteriorate": 10},

    # MINOR / STABLE (severity 4-5)
    {"symptoms": "Small cut on finger from kitchen knife, bleeding controlled with pressure, full movement intact", "chief_complaint": "Finger cut", "hidden_severity": 5, "hidden_diagnosis": "Minor Laceration", "required_treatment": "discharge", "time_to_deteriorate": 20},
    {"symptoms": "Rash on both arms appeared this morning, mild itching, no swelling, no breathing issues", "chief_complaint": "Skin rash", "hidden_severity": 5, "hidden_diagnosis": "Contact Dermatitis", "required_treatment": "discharge", "time_to_deteriorate": 20},
    {"symptoms": "Sore throat for 2 days, mild fever 37.8C, able to swallow, no neck swelling", "chief_complaint": "Sore throat", "hidden_severity": 5, "hidden_diagnosis": "Viral Pharyngitis", "required_treatment": "discharge", "time_to_deteriorate": 20},
    {"symptoms": "Lower back pain for 3 days after lifting heavy box, pain on bending, no leg weakness or numbness", "chief_complaint": "Back pain", "hidden_severity": 4, "hidden_diagnosis": "Lumbar Strain", "required_treatment": "discharge", "time_to_deteriorate": 15},
    {"symptoms": "Mild allergic reaction, hives on torso, mild itching, no facial swelling, no breathing issues", "chief_complaint": "Allergic reaction", "hidden_severity": 4, "hidden_diagnosis": "Mild Urticaria", "required_treatment": "medication", "time_to_deteriorate": 12},

    # PEDIATRIC-STYLE (still adult but younger)
    {"symptoms": "19-year-old with sudden severe testicular pain, nausea, swelling on left side, onset 2 hours ago", "chief_complaint": "Severe groin pain", "hidden_severity": 2, "hidden_diagnosis": "Testicular Torsion", "required_treatment": "surgery", "time_to_deteriorate": 4},
    {"symptoms": "High fever 40.1C, severe flank pain, painful urination, shaking chills", "chief_complaint": "Fever and flank pain", "hidden_severity": 2, "hidden_diagnosis": "Pyelonephritis with Sepsis", "required_treatment": "medication", "time_to_deteriorate": 4},

    # AMBIGUOUS PRESENTATIONS (harder difficulty)
    {"symptoms": "Chest pain that worsens with deep breath, mild cough, no fever, recent upper respiratory infection", "chief_complaint": "Chest pain", "hidden_severity": 4, "hidden_diagnosis": "Pleurisy (Viral)", "required_treatment": "discharge", "time_to_deteriorate": 12},
    {"symptoms": "Feeling very anxious, heart racing, tingling in hands, feeling like can't get enough air, trembling", "chief_complaint": "Racing heart and anxiety", "hidden_severity": 4, "hidden_diagnosis": "Panic Attack", "required_treatment": "observation", "time_to_deteriorate": 10},
    {"symptoms": "Abdominal pain, bloating, mild nausea. Patient appears anxious. Pain is diffuse, no rebound tenderness", "chief_complaint": "Abdominal discomfort", "hidden_severity": 4, "hidden_diagnosis": "Functional Abdominal Pain", "required_treatment": "discharge", "time_to_deteriorate": 12},
    {"symptoms": "Dizziness, fatigue, pale skin, shortness of breath on exertion, heavy menstrual bleeding for 2 weeks", "chief_complaint": "Dizziness and fatigue", "hidden_severity": 3, "hidden_diagnosis": "Severe Anemia", "required_treatment": "medication", "time_to_deteriorate": 6},
]

NAMES_MALE = ["Ravi Sharma", "Amit Patel", "Suresh Kumar", "Vikram Singh", "Rajesh Gupta", "Arun Mehta", "Deepak Joshi", "Manoj Verma", "Kiran Rao", "Sanjay Nair", "Arjun Das", "Nikhil Reddy", "Pranav Iyer", "Rohit Bhatia", "Sameer Khan"]
NAMES_FEMALE = ["Priya Sharma", "Anita Devi", "Sunita Patel", "Meera Nair", "Kavita Singh", "Rekha Gupta", "Deepa Joshi", "Lakshmi Rao", "Neha Verma", "Pooja Mehta", "Sonia Das", "Ritu Reddy", "Swati Iyer", "Nandini Bhatia", "Fatima Khan"]


def generate_patients(rng: random.Random, count: int, difficulty: str) -> List[Patient]:
    """Generate a list of patients for an ER scenario."""
    if difficulty == "easy":
        # Mostly clear-cut cases, mix of severities
        critical_count = 1
        urgent_count = 1
        stable_count = count - 2
    elif difficulty == "medium":
        critical_count = 2
        urgent_count = 2
        stable_count = count - 4
    else:  # hard
        critical_count = 3
        urgent_count = 3
        stable_count = count - 6

    # Select templates by severity
    critical_templates = [t for t in PATIENT_TEMPLATES if t["hidden_severity"] <= 1]
    urgent_templates = [t for t in PATIENT_TEMPLATES if t["hidden_severity"] == 2]
    moderate_templates = [t for t in PATIENT_TEMPLATES if t["hidden_severity"] == 3]
    stable_templates = [t for t in PATIENT_TEMPLATES if t["hidden_severity"] >= 4]

    selected = []
    selected.extend(rng.sample(critical_templates, min(critical_count, len(critical_templates))))
    selected.extend(rng.sample(urgent_templates, min(urgent_count, len(urgent_templates))))

    remaining = stable_count
    mod_count = min(remaining // 2, len(moderate_templates))
    selected.extend(rng.sample(moderate_templates, mod_count))
    remaining -= mod_count
    selected.extend(rng.sample(stable_templates, min(remaining, len(stable_templates))))

    # Trim or pad to exact count
    selected = selected[:count]
    rng.shuffle(selected)

    patients = []
    for i, template in enumerate(selected):
        t = copy.deepcopy(template)
        gender = rng.choice(["M", "F"])
        name = rng.choice(NAMES_MALE if gender == "M" else NAMES_FEMALE)
        age = rng.randint(18, 85)

        p = Patient(
            id=f"P{i+1:03d}",
            name=name,
            age=age,
            gender=gender,
            symptoms=t["symptoms"],
            chief_complaint=t["chief_complaint"],
            hidden_severity=t["hidden_severity"],
            hidden_diagnosis=t["hidden_diagnosis"],
            required_treatment=t["required_treatment"],
            time_to_deteriorate=t["time_to_deteriorate"],
            current_severity=t["hidden_severity"],
        )
        p._update_vitals_for_severity()
        patients.append(p)

    return patients
