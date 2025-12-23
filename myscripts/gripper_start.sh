pyenv local 3.12.3
source setup_shell.sh
pytest tests/rai_extensions/test_pcl_detection_tools.py -s -v

pytest tests/rai_extensions/test_gripping_points.py::test_gripping_points_manipulation_demo -m "" -s -v

#streamlit run examples/manipulation-demo-streamlit.py

