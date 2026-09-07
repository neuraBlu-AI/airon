# RON Project Context

## 1. Project Overview

RON is a real-world autonomous social robot inspired by the B-Bot concept from *Ron's Gone Wrong*.

The goal is NOT to build a simple remote-controlled toy or a stationary voice assistant.

RON should eventually be a physically embodied AI companion that can:

- see and recognize people
- understand its surroundings
- communicate naturally through speech
- display emotions through an animated face
- remember people and previous interactions
- develop a consistent personality
- move autonomously
- follow people
- navigate through rooms
- interact through arm gestures
- react contextually to people and events
- return autonomously to a charging dock
- potentially support safety / elderly-care use cases later

The project should be built incrementally, starting with a stationary "brain" prototype and progressively adding mobility, body, arms and autonomous behavior.

Current robot/device name: **aiRon**

---

# 2. Design Philosophy

RON should feel like a character rather than a computer on wheels.

The most important aspects are:

1. Personality
2. Natural interaction
3. Expressive face
4. Awareness of people and surroundings
5. Memory
6. Autonomous behavior

Technical capabilities should support these goals rather than dominate the user experience.

The final robot should have a friendly, rounded, compact appearance reminiscent of the general B-Bot concept, while ultimately using an original physical design if the project becomes commercial.

Target approximate physical dimensions:

- Height: ~65 cm
- Width: ~42 cm
- Depth: ~34 cm
- Target weight: ~12–17 kg

Most structural/body components should eventually be 3D printed.

---

# 3. Development Strategy

RON is being developed in phases.

## Phase 1 — Brain / Tabletop Prototype

CURRENT PHASE.

Build a stationary prototype consisting of:

- Jetson computer
- OAK-D camera
- microphone array
- 10.1" face display
- speakers
- AI software

No motors, wheels, LiDAR, arms or battery are required yet.

Primary goal:

Create the first convincing "alive" interaction.

Example:

A person enters the camera field.

RON:

1. detects the person
2. identifies them if known
3. tracks their face
4. moves its animated eyes toward them
5. recognizes who they are
6. greets them by name
7. can have a conversation
8. remembers relevant information

Example:

"Hi Pierre, nice to see you again."

The face should visually react while speaking/listening/thinking.

---

## Phase 2 — Mobility

Add:

- differential drive
- two powered wheels
- hidden caster/support wheel
- motor drivers
- microcontroller
- LiDAR
- ROS 2
- SLAM
- autonomous navigation
- obstacle avoidance
- person following

RON should eventually be able to:

- map a home
- navigate between rooms
- follow a selected person
- approach people
- stop safely
- avoid obstacles

---

## Phase 3 — Physical Body

Design and manufacture the body.

Primary manufacturing method:

3D printing.

Desired characteristics:

- rounded / egg-like body
- smooth white exterior
- large front face display
- hidden wheels
- removable service panels
- modular internal frame
- accessible electronics
- ventilation for Jetson
- space for battery and future actuators

CAD should be modular so components can be replaced without redesigning the entire robot.

---

## Phase 4 — Personality / "Soul"

Develop the behavioral system that makes RON feel alive.

This includes:

- personality
- emotional state
- facial expressions
- conversational memory
- person-specific memory
- contextual reactions
- idle behavior
- curiosity
- attention
- social interaction
- relationship modeling

RON should not simply wait for commands.

Eventually it should be capable of initiating interactions when appropriate.

Examples:

- noticing someone entering
- greeting them
- reacting when someone leaves
- remembering unfinished conversations
- noticing unusual events
- asking contextually relevant questions

---

## Phase 5 — Arms and Body Gestures

Add two simple expressive arms.

Initial target:

~3 degrees of freedom per arm.

Complex robotic hands are NOT required initially.

The arms are primarily for:

- pointing
- waving
- expressive gestures
- emotional body language

Possible additional body movement:

Approximately ±8–10° body tilt/pivot for expressive movement.

---

## Phase 6 — Full Autonomous Robot

Integrate:

- vision
- speech
- AI
- memory
- personality
- face
- navigation
- arms
- battery
- charging
- safety systems

Add automatic docking.

RON should autonomously return to its charging station when necessary.

---

# 4. Current Hardware

The following hardware has already been purchased and is available.

## Main Computer

NVIDIA Jetson Orin Nano Super Developer Kit 8GB

Role:

Primary high-level computer.

Responsibilities:

- computer vision
- AI inference
- speech processing
- face rendering
- memory
- navigation
- ROS 2
- high-level behavior

The Jetson should NOT eventually be responsible for low-level motor safety.

---

## Storage

NVMe SSD installed in the Jetson.

Approximate capacity:

500 GB.

Used for:

- operating system
- AI models
- application code
- local memory/database
- vision models
- logs
- cached assets

---

## Vision

Luxonis OAK-D Lite AF

Connected via USB 3.

Capabilities include:

- RGB camera
- stereo cameras
- depth perception
- onboard Myriad X processing

Planned uses:

- face detection
- face recognition
- person tracking
- eye contact / gaze direction
- object recognition
- depth estimation
- gesture recognition
- fall detection
- navigation assistance

Software ecosystem:

DepthAI.

The camera is already physically connected and detected over USB.

---

## Microphones

Seeed Studio ReSpeaker XMOS XVF3800 4-Mic Array

Part number:

101991441

Connection:

USB.

Capabilities include:

- four microphones
- beamforming
- direction of arrival
- acoustic echo cancellation
- automatic gain control
- voice activity detection
- dereverberation
- noise suppression

Planned uses:

- speech recognition
- determining where a speaker is located
- allowing RON to orient attention toward the speaker
- echo cancellation while RON is speaking

---

## Display

Waveshare 10.1EP-CAPLCD

Specifications:

- 10.1"
- 1920 × 1200
- IPS
- capacitive touch
- 10-point touch
- optical bonding
- integrated stereo speakers
- 3.5 mm audio output

Connection:

Jetson DisplayPort
→ DP-to-HDMI
→ Waveshare HDMI

Touch:

USB.

The display is currently working correctly under Ubuntu.

Primary purpose:

RON's animated face.

The screen should normally NOT look like a conventional computer UI.

Instead, most of the screen should behave as RON's face.

---

## Audio Output

For Phase 1, use the speakers integrated into the Waveshare display.

A dedicated speaker/amplifier system can be added later if necessary.

---

# 5. Compute Architecture

RON should eventually use a two-computer architecture.

## High-Level Computer

Jetson Orin Nano.

Handles:

- vision
- AI
- LLM
- speech
- memory
- navigation
- behavior
- face animation
- planning

## Low-Level Controller

Future ESP32 or STM32.

Handles:

- wheel motors
- encoders
- servos
- battery monitoring
- emergency stop
- watchdog
- low-level safety
- deterministic motion control

Important principle:

The Jetson must NOT be the only safety controller for motors.

If Linux, Python, ROS or the AI crashes, the low-level controller must still be able to stop the robot safely.

---

# 6. Software Architecture

Primary Phase 1 programming language:

Python.

Likely technologies:

- Python
- DepthAI
- OpenCV
- PySide6 / Qt
- speech-to-text
- text-to-speech
- LLM APIs and/or local models
- local database / vector memory
- ROS 2 later

C++ will likely be used later for MCU firmware and potentially performance-critical components.

---

# 7. Modular Software Design

Avoid building RON as one giant Python script.

Use independent modules/services.

Suggested architecture:

vision_service
speech_service
audio_service
memory_service
brain_service
face_service

Later:

motion_service
navigation_service
safety_service
docking_service

Conceptually:

CAMERA
   ↓
VISION
   ↓
WORLD STATE
   ↓
BRAIN / BEHAVIOR
 ↙     ↓      ↘
FACE  SPEECH  MEMORY
              ↓
          LONG-TERM STATE

Later:

BRAIN
  ↓
MOTION INTENT
  ↓
ROS / MOTION SERVICE
  ↓
MCU
  ↓
MOTORS / SERVOS

The AI should request actions such as:

"look at Pierre"

or:

"move toward Pierre"

rather than directly controlling PWM/motor values.

---

# 8. Vision System

The OAK-D should eventually maintain a structured understanding of what RON currently sees.

Example internal world state:

{
  "people": [
    {
      "id": "person_001",
      "name": "Pierre",
      "recognized": true,
      "distance": 2.1,
      "position": "left",
      "looking_at_ron": true
    }
  ],
  "objects": [],
  "environment": {},
  "timestamp": "..."
}

The LLM should not have to process raw camera frames continuously.

Instead, the vision subsystem should convert perception into useful structured events/state.

Examples:

PERSON_ENTERED
PERSON_LEFT
KNOWN_PERSON_DETECTED
UNKNOWN_PERSON_DETECTED
PERSON_LOOKING_AT_RON
PERSON_FELL
OBJECT_DETECTED
GESTURE_DETECTED

This reduces latency and AI cost and creates a cleaner architecture.

---

# 9. Face System

The face is one of the most important parts of RON.

The display should render a smooth animated face at approximately 60 FPS.

Likely implementation:

PySide6 / Qt.

The face system should run independently from the LLM.

The LLM should provide high-level emotional/behavioral state rather than individual animation frames.

Example:

{
  "emotion": "happy",
  "attention_x": 0.65,
  "attention_y": 0.40,
  "speaking": true,
  "intensity": 0.7
}

The face renderer converts this into:

- eye movement
- blinking
- pupil movement
- eyebrow/shape changes
- mouth animation
- subtle idle movement

Possible states:

idle
curious
happy
excited
confused
thinking
listening
speaking
sleepy
concerned

Eye position should eventually be connected to person tracking.

If a person's face moves left, RON's eyes should follow naturally.

---

# 10. Speech System

Desired pipeline:

Microphone array
→ VAD
→ speech recognition
→ conversational brain
→ response generation
→ TTS
→ speakers

The microphone array's direction-of-arrival information can eventually influence RON's attention.

Example:

Someone speaks from the right.

RON:

1. detects speech
2. estimates speaker direction
3. moves eyes toward the speaker
4. identifies the person visually
5. responds

Low latency is important.

RON should feel conversational rather than like a voice assistant with long pauses.

---

# 11. Memory

Memory is a core feature.

RON should distinguish between:

## Short-Term Context

Current conversation and immediate environment.

## Person Memory

Information associated with known individuals.

Example:

Pierre:
- preferred language
- previous conversations
- interests
- relationship context
- interaction history

## Episodic Memory

Events.

Examples:

"Pierre showed me a new robot part yesterday."

"Anna asked me to remind her about something."

## Semantic Memory

General learned information and facts.

The system should NOT blindly store every conversation.

Memory should be selective and relevance-based.

---

# 12. Identity / Face Recognition

RON should recognize known people.

Initial approach:

face detection
→ face embedding
→ compare against local identity database
→ assign persistent person ID

Example:

person_001 = Pierre
person_002 = Elizabeth

Unknown people should receive temporary IDs until explicitly identified.

Privacy should be considered from the beginning, especially if the system becomes a commercial product.

---

# 13. Behavioral System

Long-term goal:

RON should behave proactively rather than purely reactively.

Potential behavior loop:

PERCEIVE
↓
UPDATE WORLD STATE
↓
CHECK IMPORTANT EVENTS
↓
UPDATE ATTENTION
↓
DECIDE WHETHER TO ACT
↓
GENERATE HIGH-LEVEL ACTION
↓
EXECUTE
↓
STORE RELEVANT MEMORY

Example:

Camera detects Pierre entering.

Vision:

KNOWN_PERSON_ENTERED(Pierre)

Behavior engine:

Pierre has not interacted with RON today.

Decision:

Greet Pierre.

Face:

look toward Pierre
happy expression

Speech:

"Hey Pierre!"

Memory:

store interaction timestamp

The LLM should be one component of this system, not the entire control loop.

---

# 14. Future Mobility Hardware

Planned:

- differential drive
- two powered wheels
- hidden caster
- wheel encoders
- motor controller
- MCU
- LiDAR
- IMU
- bumper/contact sensors if useful

Navigation stack likely based on ROS 2.

Capabilities:

- SLAM
- localization
- Nav2
- obstacle avoidance
- person following
- room-to-room navigation

---

# 15. Charging

Long-term goal:

Automatic charging dock.

RON should monitor battery level and autonomously decide when to charge.

Example:

Battery < threshold
→ finish/interrupt non-critical activity
→ navigate to dock
→ align with charging contacts
→ verify charging
→ enter charging behavior

LiFePO4 is currently preferred for the eventual main battery due to safety and cycle life, but final battery design has not yet been selected.

---

# 16. Arms

Future arms are intended primarily for communication and personality.

Approximately:

3 DOF per arm.

Possible actuators:

bus servos.

Gestures:

- waving
- pointing
- arms up
- shrugging
- excited movement
- defensive/concerned posture

No complex fingers/hands required for V1.

---

# 17. Elderly-Care / Safety Potential

A possible future application is an elderly-care companion.

Potential capabilities:

- companionship
- medication/routine reminders
- fall detection
- inactivity detection
- checking whether a person responds
- contacting relatives
- escalation to emergency services
- following a person through the home
- voice communication

Example fall workflow:

Vision detects probable fall
↓
RON approaches / looks toward person
↓
RON asks:
"Are you okay?"
↓
Listen for response
↓
No response
↓
Repeat/check
↓
Trigger configured alert

This is NOT part of Phase 1 and would require careful safety, privacy, regulatory and reliability work before real-world deployment.

---

# 18. Current Project Status

Current Phase:

PHASE 1 — BRAIN

Working:

- Jetson Orin Nano Super is operational
- Ubuntu / NVIDIA environment is operational
- NVMe is installed
- 10.1" Waveshare display works
- DisplayPort → HDMI connection works
- touchscreen/display hardware available
- OAK-D Lite is connected via USB
- Linux detects the OAK-D over USB
- ReSpeaker hardware is available
- display's integrated speakers are available

Immediate development priority:

Get the OAK-D working through DepthAI and display a live RGB image.

Then:

1. RGB camera
2. stereo/depth
3. person detection
4. face detection
5. face tracking
6. animated eyes
7. connect eye direction to face position
8. face recognition
9. microphone input
10. speech-to-text
11. TTS
12. conversational brain
13. memory
14. combine everything into first RON interaction

---

# 19. Immediate Milestone

The first major milestone should be:

RON recognizes Pierre and visually reacts.

Target interaction:

Pierre walks into view.

OAK-D:
detects face
→ tracks face
→ recognizes Pierre

RON display:
eyes move toward Pierre
→ blink
→ happy expression

RON:
"Hi Pierre."

Pierre speaks.

ReSpeaker:
captures speech

RON:
understands and responds naturally.

RON remembers the interaction.

This is the first complete demonstration that RON is becoming a social robot rather than simply a collection of hardware.

---

# 20. Engineering Principles

When making implementation decisions for RON:

- Prefer modular architecture.
- Keep perception separate from reasoning.
- Keep AI reasoning separate from safety-critical control.
- Keep animation independent from LLM latency.
- Do not send raw continuous sensor streams to the LLM unnecessarily.
- Convert sensor data into structured world state/events.
- Prefer local processing for low-latency perception.
- Allow cloud AI where it significantly improves intelligence.
- Design APIs between modules early.
- Avoid premature ROS complexity during Phase 1.
- Introduce ROS 2 when mobility/navigation begins.
- Make hardware replaceable.
- Design the body around serviceability.
- Preserve the possibility of commercializing the architecture later.
- Safety-critical behavior must never depend solely on an LLM.
- Do not over-engineer future phases before the Phase 1 interaction works.

---

# 21. Current Next Task

Set up the Luxonis OAK-D Lite AF on the Jetson using DepthAI.

First objectives:

1. Confirm DepthAI can enumerate the device.
2. Display RGB live video.
3. Test stereo/depth output.
4. Measure distance to a person.
5. Detect and track a face.
6. Convert face coordinates into normalized attention coordinates.
7. Feed those coordinates into the future RON face renderer.

Do not begin mobility or ROS 2 work until the Phase 1 perception/interaction foundation is working reliably.

---

# 22. Working From Tickets

Work comes from the tracker, not from a conversation.

Tickets live in Plane, in the project **airon** (identifier `AIRON`), reached
through the MCP server configured in `.mcp.json`. Anything worth doing should
exist as a ticket before it exists as code.

## Branch Per Ticket

Every ticket gets its own branch, and the branch name carries the ticket
identifier. That identifier is what lets a branch, a pull request and a ticket
be matched to each other later without having to ask anybody who wrote what.

Branch naming:

    <TICKET-ID>-<short-kebab-description>

Examples:

    AIRON-2-memory-service
    AIRON-3-conversational-brain
    AIRON-7-full-duplex-audio

Rules:

- One ticket, one branch, one pull request.
- Never commit ticket work directly to `main`.
- Branch from `main`, unless the ticket genuinely depends on another that has
  not merged yet - in which case branch from that one and say so in the pull
  request, so the reviewer knows what to merge first.
- Name the ticket in the pull request, so the tracker and the repository stay
  connected from both ends.
- Move the ticket to the matching state when work starts and when the pull
  request opens, rather than leaving the board to be reconstructed afterwards.

Work that is not a ticket - a fix to this document, tooling configuration -
does not need a ticket branch, but still does not go straight to `main`.

## Who Merges

Pull requests are opened by whoever did the work. They are reviewed and merged
by the repository owner, and by nobody else.

This is not a formality. Every merge so far has been onto a robot that has to
be stood in front of to be judged: whether a reply arrives quickly enough to
feel like conversation, whether a pause is aiRon thinking or aiRon broken,
whether it greeted the right person. None of that is visible in a diff, and
a green test run is not a substitute for having watched the thing run.

Rules:

- Claude opens pull requests. Claude does not merge them, and does not merge
  its own work to `main` under any circumstances.
- A pull request is where work is handed over, not where it is finished.
  Say what was measured, what was not verified, and what the reviewer should
  look at while the robot is in front of them.
- Anything found while working that was not asked for goes in the pull request
  description or a new ticket - not quietly into the branch.
- If a merge really is wanted immediately, the owner says so in that instance.
  It does not become the habit afterwards.
- Deleting branches, retargeting a pull request, and closing tickets are part
  of the handover and can be done as asked - it is the merge to `main` that
  is reserved.

The one thing worth checking on every pull request, because it has gone wrong
twice: **the base branch**. A pull request stacked on another branch does not
always get retargeted to `main` when that branch merges, and merging it then
puts the work somewhere that is not `main` without complaining. #2 landed on
`phase1/vision-face-services` this way and had to be redone as #4.
