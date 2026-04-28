import json

new_answer = (
    "At NVIDIA, I worked on PilotNet, which is an end-to-end deep learning model used for autonomous steering.\n\n"
    "What PilotNet Does\n"
    '"PilotNet takes input from a front-facing camera and directly predicts the steering angle for the vehicle.\n'
    "So instead of breaking the problem into multiple steps like lane detection or object detection, it learns everything in one model.\"\n"
    "\U0001f449 Input \u2192 camera image\n"
    "\U0001f449 Output \u2192 steering angle\n\n\n"
    "How It Works (High-Level)\n"
    "\"It's based on a convolutional neural network (CNN):\n"
    "\t\u2022 image goes through convolution layers \n"
    "\t\u2022 features like lanes, road edges, and obstacles are extracted \n"
    "\t\u2022 final layers output a steering value \n"
    "So the model learns driving behavior directly from data.\"\n\n"
    "\U0001f539 Your Role (VERY IMPORTANT)\n"
    "\"My role was focused on validation and quality engineering, not training the model.\n"
    "I worked on:\n"
    "\t\u2022 validating the predicted steering output \n"
    "\t\u2022 testing the model under different driving scenarios \n"
    "\t\u2022 analyzing logs and outputs \n"
    "\t\u2022 identifying edge cases where the model behaved incorrectly\" \n\n"
    "\U0001f539 Example (Makes answer powerful)\n"
    "\"For example, I would check scenarios like:\n"
    "\t\u2022 curved roads \n"
    "\t\u2022 lane markings not clearly visible \n"
    "\t\u2022 low lighting conditions \n"
    "If the predicted steering didn't match expected driving behavior, we flagged it.\"\n\n"
    "\U0001f539 Challenges\n"
    "\"The main challenges were:\n"
    "\t\u2022 model behavior in edge cases \n"
    "\t\u2022 lack of explainability (black-box nature) \n"
    "\t\u2022 ensuring consistency across different environments\"\n\n"
    "How You Handled It\n"
    "\"To handle this, I:\n"
    "\t\u2022 created diverse validation scenarios \n"
    "\t\u2022 analyzed prediction outputs with logs \n"
    "\t\u2022 compared behavior across multiple runs\" \n\n\n"
    "\U0001f539 Where It Fits in System\n"
    "\"PilotNet was part of a larger perception pipeline and was sometimes combined with other models like LaneNet and PathNet in an ensemble system for better accuracy.\"\n\n\n"
    "\U0001f539 Impact\n"
    "\"This helped ensure that the model's steering decisions were reliable and safe before being used in real-world scenarios.\"\n\n\n"
    "I worked across multiple components in the autonomous driving stack:\n"
    "\t\u2022 PilotNet \u2192 end-to-end model that predicts steering angle from camera images \n"
    "\t\u2022 PredictionNet \u2192 predicts future movement and behavior of surrounding vehicles \n"
    "\t\u2022 DriverNet \u2192 monitors driver state and behavior for safety (attention, distraction, etc.) \n"
    "\t\u2022 LidarNet \u2192 processes LiDAR data to detect objects and understand 3D environment \n"
    "\t\u2022 Localization Software \u2192 determines precise vehicle position using sensor + map data \n"
    "\t\u2022 MapNet \u2192 provides map-based context like road structure and navigation paths \n"
    "\t\u2022 ParkNet \u2192 handles parking scenarios like slot detection and maneuvering \n\n"
    "AutoHighBeamNet \u2192 automatically controls high/low beam based on surrounding vehicles and lighting conditions"
)

lines = []
with open('prompts/generic_answers.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line:
            lines.append('')
            continue
        obj = json.loads(line)
        if obj.get('id') == 'role_work_at_nvidia':
            obj['answer'] = new_answer
        lines.append(json.dumps(obj, ensure_ascii=False))

with open('prompts/generic_answers.jsonl', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Done')
