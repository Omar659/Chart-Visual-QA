Here is a visual structure (layout) proposal for your project, focusing primarily on the **User Interface (Web UI)** described in the implementation track, along with a visual organization of the architecture and workflow.

Since I am a text-based AI, I have created a character-based  *wireframe* , which is ideal for guiding the frontend team (Victor & Min) through Phase C.

### 1. Web Interface Layout (Frontend - React)

This is the proposed design for the main application screen. The idea is to keep the interface clean, focusing on the chart and the question.

**Plaintext**

```
+-----------------------------------------------------------------------------+
|  📊 Chart-Visual-QA                                           [Github]      |
+-----------------------------------------------------------------------------+
|                                                                             |
|  +-----------------------------------------------------------------------+  |
|  |                                                                       |  |
|  |                           [ Upload Area ]                             |  |
|  |                                                                       |  |
|  |   +---------------------------------------------------------------+   |  |
|  |   |                                                               |   |  |
|  |   |      📁 Drag and drop the chart here or [Click to open        |   |  |
|  |   |                       file explorer]                          |   |  |
|  |   |                                                               |   |  |
|  |   |           (After upload, the chart preview goes here)         |   |  |
|  |   |                                                               |   |  |
|  |   +---------------------------------------------------------------+   |  |
|  |                                                                       |  |
|  |                           [ Question Field ]                          |  |
|  |                                                                       |  |
|  |   +---------------------------------------------------------------+   |  |
|  |   | 💬 What do you want to know about this chart?                 |   |  |
|  |   +---------------------------------------------------------------+   |  |
|  |                                                                       |  |
|  |   +---------------------------------------------------------------+   |  |
|  |   |                       [ 🚀 Submit Question ]                  |   |  |
|  |   +---------------------------------------------------------------+   |  |
|  |                                                                       |  |
|  |                           [ Answer Area ]                             |  |
|  |                                                                       |  |
|  |   +---------------------------------------------------------------+   |  |
|  |   |  💡 Answer:                                                   |   |  |
|  |   |  The projected value for 2024 is 42%.                         |   |  |
|  |   |                                                               |   |  |
|  |   |  [✓] Mock: false                    [ Inference time: 1.2s]   |   |  |
|  |   +---------------------------------------------------------------+   |  |
|  |                                                                       |  |
|  +-----------------------------------------------------------------------+  |
|                                                                             |
+-----------------------------------------------------------------------------+
```

**UI States (Phase C):**

* **Empty:** Displays the upload icon and keeps the text field disabled until an image is uploaded.
* **Loading:** Upon clicking "Submit", the button changes to an animated spinner and the text changes to "Processing model...".
* **Error:** If `image` or `question` are empty, display a red alert below the corresponding field.

### 2. System Architecture Layout (The "Contract")

To align the Modeling and Implementation teams, here is the visual flow of the backend. It shows exactly where the "seam" of the project occurs:

**Plaintext**

```
 [ FRONTEND ]                      [ BACKEND (Flask) ]
Victor & Min                      Victor & Min

 🌐 React UI    ------------>   POST /api/ask
(Image + Text)  <------------   JSON { "answer": "..." }
                                        |
                                        | (Integration Contract)
                                        v
                            def run_inference(image, question) -> str:
                                        |
 =======================================|=======================================
                                        |
 [ MODEL PIPELINE ]                     v
 Susanne & Omar                     Preprocess (Resize, OCR, Prompt)
                                        |
                                        v
                                    VLM Model (Zero-shot OR LoRA)
                                        |
                                        v
                                    Postprocess (Parse, Units)
                                        |
                                        v
                                   return "Short Answer"
```

### 3. Tracking Layout (Suggested Kanban Board)

To organize deliverables and milestones (Milestones 1 to 5), the team can structure the work as follows:


| **📋 Backlog (To Do)**              | **⚙️ In Progress (Susanne & Omar)** | **💻 In Progress (Victor & Min)** | **✅ Done / Validation**   |
| ----------------------------------------- | ------------------------------------------- | --------------------------------------- | -------------------------------- |
| **[M4]**Integrate real model into backend | **[M3]**Run Baseline 1 (Zero-shot VLM)      | **[M1]**React + Vite + Flask Setup      | Load and inspect ChartQA dataset |
| **[M5]**Results table + README            | **[M3]**Train Baseline 2 (LoRA)             | **[M2]**Build UI and states (Mock)      | Define inference contract        |
| Error analysis (20 cases)                 | **[M3]**Create pre/post-processing          | Build`POST /api/ask`endpoint          |                                  |
