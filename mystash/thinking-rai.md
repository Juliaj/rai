

The problem is, people who work on real robotics often don’t have Python skills. And if they don’t have Python skills, they probably don’t have experience with Langchain or other genAI tools.
There’s a huge entry threshold for RAI, and even with an amazing public roadmap, it’s not going to be easy to get many people on board

Do robotics people have scenarios ? What are the problem that they are facing ?

They want to use genAI to do X, Y things and once the success rate is good enough then we can think about extending cases. X an Y being very low level cases 

do some hard thinking, ask questions as "is there a gap? what takes these to next level and make them to run on robots " ?

if they are okay to share, just to evaluate from RAI as framework level on how to reduce the barriers,

I think most of the use cases are revolving around zero-shot object detection

So literally, a single module being used to do 100% of the work


Yes, excellent use case!
Human-in-the-Loop Assistance:

Robot encounters uncertainty (ambiguous object, navigation deadlock, task failure)
Sends help request via WebSocket to web dashboard
Human receives notification, views context (camera feed, map, error state)
Provides input (select correct object, draw path, approve decision)
Response sent back through HTTPConnector to resume operation

Why HTTPConnector fits:

Real-time alerting via WebSocket push notifications
Rich context delivery (images, telemetry) over HTTP
Browser-based = accessible anywhere, no special software
Bidirectional for request/response flow
Multiple supervisors can monitor/respond

Perfect for semi-autonomous systems needing occasional human guidance—factory robots, delivery bots, assistive devices.

RoboTechAI Demo for ROSCon2025

Robotec.ai Advances Agentic AI With Liquid AI and AMD Processor Technology For Robotics

Robotec.ai, a leading deep-tech company specializing in advanced simulation solutions for the testing and deployment of robotics solutions across key global industries, has collaborated with AMD and Liquid AI to demonstrate the first fully autonomous warehouse robot powered exclusively by AMD Ryzen™ AI processors.

The robot leverages Agentic AI capabilities to dynamically plan and execute tasks in real time without reliance on hard-coded scripts. Powered by Liquid AI's next-generation LFM2 Vision Language Models, it seamlessly combines perception, reasoning, and natural language understanding to interpret commands, detect safety hazards such as spills or blocked exits, and autonomously execute corrective actions. The collaboration unlocks the full potential of the platform, thanks to extensive testing in simulated environments. Simulation, created with Open 3D Engine, supports validation of embedded AI on real hardware, avoiding the costs and risks of physical testing. It is a step towards the future of reasoning robots that will intelligently respond to the changing environment around them.
Physical intelligence as a glimpse into the future of industrial robotics

Physical intelligence will create immense value through efficient platforms like mobile manipulators, combined with agentic AI that integrates foundation models and state-of-the-art, reliable robotics.

With this new demonstration, a flexible mobile robot, powered by agentic AI that is running on AMD silicon, operates within a warehouse with mixed traffic. It completes tasks specified by humans using natural language, and adapts to changing conditions through replanning. The robot also serves as an inspection agent, alerting operators whenever unexpected occurrences or safety issues are detected in the warehouse area.

Liquid VLM (LFM2-VL): Multimodal Intelligence for Embodied Autonomy

At the core of this system is Liquid AI’s LFM2-VL, a next-generation Vision Language Model designed for embedded, real-time intelligence. Compact yet powerful, it integrates perception, reasoning, and language understanding into a single multimodal foundation model optimized for AMD hardware. To tailor the model for agentic robotics downstream tasks, Liquid leveraged simulation-derived synthetic data provided by Robotec, enabling domain-specific fine-tuning and robustness in complex industrial environments.
LFM2-VL interprets visual scenes, performs context-aware reasoning, and plans goal-driven actions entirely on-device — eliminating latency and cloud dependency. Its efficiency and responsiveness enable the robot to operate safely and autonomously in dynamic industrial environments.
ROS 2-powered hardware-in-the-loop simulation for rapid OEM prototyping

The Robotec.ai team has created a hardware-in-the-loop (HiL) simulation running a ROS 2 stack and AI inference on a single Ryzen device. In this configuration, the simulation runs on a separate AMD-powered computer and delivers a virtual environment that is indistinguishable from the physical world to the robot. The HiL interface then connects the robot’s sensor and actuator signals directly to the simulator’s inputs and outputs, allowing the same control logic to be applied under realistic, reproducible conditions. The robot used in the simulation, RB-KAIROS+, is designed and produced by Robotnik.

Robotec_AMD_ROSCon_screen_1.jpg

It supports the transition of the HiL setup to the real-world setup, enabling rapid OEM prototyping and demonstrating the solution's robustness. In the long run, HiL accelerates the innovation cycles, significantly speeding up the R&D phase. Robotec.ai has collaborated with AMD on hardware-in-the-loop before, boosting performance with simulation-driven testing.
Human-in-the-loop: orchestrating robots with a UI on an industrial tablet

The platform comes with an easy-to-use tablet UI that displays the robot’s plan, missions, and reasoning steps, alongside live maps, camera feeds, and anomaly reports, which lets human operators monitor robotics operations in real-time and address potential issues as they arise. Seamless human-robot interface keeps humans in the loop to supervise workflows.
AMD platform: an excellent match for Physical AI

The AMD processor proves to be an excellent “brain” for agentic AI while also capable of running the robotics software stack in parallel. It is fast, compact, and efficient, with incredible performance in both speed and power efficiency, as measured by transparent metrics such as tokens, latency, and throughput. Initially demonstrated on an AMD Ryzen processor, Robotec.ai plans to transition to an AMD embedded x86 solution in the near future.
Robotec.ai, AMD and Liquid AI collaborate on Embedded Agentic AI at ROSCon 2025

AMD, Robotec.ai and Liquid AI will showcase Agentic AI in robotics at ROSCon 2025 in Singapore, where participants will have a unique chance to see the demo live at the AMD stand (Booth 17/18). Robotec.ai’s team will be present onsite to introduce the solution to the broader audience.



6/ simulation is a tool, not a destination.

we shouldn't expect sim to be perfect, and it doesn’t have to be. the emerging pattern is: use sim to explore, apply task-aware domain randomization to bridge physics, use ml-driven augmentation/adaptation to bridge vision, then validate with small, frequent real-robot runs. the bar is “good enough to guide real-world improvement,” not “indistinguishable from reality.”




[python-2] 2025-11-29 14:50:33 jubuntu GroundingDinoAgent[277958] INFO Successfully downloaded weights (661.85 MB)
[python-2] 2025-11-29 14:50:36 jubuntu GroundingDinoAgent[277958] INFO GroundingDinoAgent initialized
[python-2] 2025-11-29 14:50:36 jubuntu GroundedSamAgent[277958] INFO Downloading weights from https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt to /home/juliaj/.cache/rai/vision/weights/sam2_hiera_large.pt
[python-2] 2025-11-29 14:51:54 jubuntu GroundedSamAgent[277958] INFO Successfully downloaded weights (856.35 MB)

