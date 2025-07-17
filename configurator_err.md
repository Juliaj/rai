2025-07-17 20:48:13 jubuntu root[1000133] ERROR Audio samples exceed expected range [-1, 1]. Found range: [-0.479641318321228, 1.0419840812683105]. This may cause audio distortion during conversion.

manipulator demo

1. Start the demo

    ```shell
    ros2 launch examples/manipulation-demo.launch.py game_launcher:=demo_assets/manipulation/RAIManipulationDemo/RAIManipulationDemo.GameLauncher
    ```

[RAIManipulationDemo.GameLauncher-1] Module: Attempting to load module:libevdev.so
[RAIManipulationDemo.GameLauncher-1] Module: Failed with error:
[RAIManipulationDemo.GameLauncher-1] libevdev.so: cannot open shared object file: No such file or directory


Running with different model 

llama4

  File "/home/juliaj/.cache/pypoetry/virtualenvs/rai-framework--61gjQXo-py3.12/lib/python3.12/site-packages/ollama/_client.py", line 168, in inner
    raise ResponseError(e.response.text, e.response.status_code) from None
ollama._types.ResponseError: model requires more system memory (60.3 GiB) than is available (55.9 GiB) (status code: 500)


debugging code

[INFO] [1753112908.063097871] [rai_ros2_connector_2b88e9a8094a]: 🔍 [GDINO] Waiting for response...
[INFO] [1753112908.563869412] [rai_ros2_connector_2b88e9a8094a]: 🔍 [GDINO] Response received: 5 detection(s)
[ERROR] [1753112908.564371688] [rai_ros2_connector_2b88e9a8094a]: ❌ [GetObjectPositionsTool] GetGrabbingPointTool failed: 'Pose2D' object has no attribute 'x'
2025-07-21 08:48:28 jubuntu rai.agents.langchain.core.conversational_agent[167254] INFO Tool get_object_positions completed in 0.52 seconds. Tool output: Failed to get grabbing points: 'Pose2D' object has no attribute 'x'




2. Potential work areas
- ros_logs.py
- bench testing
- by pass the initial langchain to sequential task and tool calling, this is to workaround OpenAI pricing


