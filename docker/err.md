component_container-6] [INFO] [1753127065.678174334] [container0]: Instantiate class: rclcpp_components::NodeFactoryTemplate<depth_image_proc::PointCloudXyzrgbNode>
[INFO] [python-2]: sending signal 'SIGINT' to process[python-2]
[INFO] [static_transform_publisher-3]: sending signal 'SIGINT' to process[static_transform_publisher-3]
[INFO] [robot_state_publisher-4]: sending signal 'SIGINT' to process[robot_state_publisher-4]
[INFO] [move_group-5]: sending signal 'SIGINT' to process[move_group-5]
[INFO] [component_container-6]: sending signal 'SIGINT' to process[component_container-6]
[python-2] Traceback (most recent call last):
[python-2]   File "/rai/src/rai_extensions/rai_open_set_vision/scripts/run_vision_agents.py", line 17, in <module>
[python-2]     from rai.agents import wait_for_shutdown
[python-2]   File "/rai/src/rai_core/rai/__init__.py", line 15, in <module>
[python-2]     from .agents import AgentRunner, ReActAgent
[python-2]   File "/rai/src/rai_core/rai/agents/__init__.py", line 16, in <module>
[python-2]     from rai.agents.langchain import BaseStateBasedAgent, ReActAgent
[python-2]   File "/rai/src/rai_core/rai/agents/langchain/__init__.py", line 15, in <module>
[python-2]     from .agent import BaseState, LangChainAgent, newMessageBehaviorType
[python-2]   File "/rai/src/rai_core/rai/agents/langchain/agent.py", line 26, in <module>
[python-2]     from rai.agents.langchain.callback import HRICallbackHandler
[python-2]   File "/rai/src/rai_core/rai/agents/langchain/callback.py", line 20, in <module>
[python-2]     from langchain_core.callbacks import BaseCallbackHandler
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/langchain_core/callbacks/__init__.py", line 23, in <module>
[python-2]     from langchain_core.callbacks.manager import (
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/langchain_core/callbacks/manager.py", line 22, in <module>
[python-2]     from langsmith.run_helpers import get_tracing_context
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/langsmith/run_helpers.py", line 45, in <module>
[python-2]     from langsmith import client as ls_client
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/langsmith/client.py", line 70, in <module>
[static_transform_publisher-3] [INFO] [1753127065.690098357] [rclcpp]: signal_handler(signum=2)
[python-2]     from langsmith import env as ls_env
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/langsmith/env/__init__.py", line 3, in <module>
[python-2]     from langsmith.env._runtime_env import (
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/langsmith/env/_runtime_env.py", line 10, in <module>
[python-2]     from langsmith.utils import get_docker_compose_command
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/langsmith/utils.py", line 43, in <module>
[python-2]     from langsmith import schemas as ls_schemas
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/langsmith/schemas.py", line 716, in <module>
[python-2]     class TracerSessionResult(TracerSession):
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/pydantic/v1/main.py", line 178, in __new__
[python-2]     annotations = resolve_annotations(namespace.get('__annotations__', {}), namespace.get('__module__', None))
[python-2]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[python-2]   File "/root/.cache/pypoetry/virtualenvs/rai-framework-LG7o02pr-py3.12/lib/python3.12/site-packages/pydantic/v1/typing.py", line 398, in resolve_annotations
[python-2]     value = ForwardRef(value, is_argument=False, is_class=True)
[python-2]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[python-2]   File "/usr/lib/python3.12/typing.py", line 897, in __init__
[robot_state_publisher-4] [INFO] [1753127065.690390213] [rclcpp]: signal_handler(signum=2)
[python-2]     code = compile(arg_to_compile, '<string>', 'eval')
[python-2]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[python-2] KeyboardInterrupt
[move_group-5] [INFO] [1753127065.690654607] [rclcpp]: signal_handler(signum=2)
[component_container-6] [INFO] [1753127065.690880424] [rclcpp]: signal_handler(signum=2)
[move_group-5] terminate called after throwing an instance of 'std::runtime_error'
[move_group-5]   what():  context cannot be slept with because it's invalid
[move_group-5] Stack trace (most recent call last) in thread 847898:

