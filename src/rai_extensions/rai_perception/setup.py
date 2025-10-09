# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "rai_perception"

# Read long description safely
here = Path(__file__).parent.resolve()
readme_path = here / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="rai-perception",
    version="0.1.1",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    package_data={"rai_perception": ["configs/seg_config.yml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=[
        "setuptools",
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "rf-groundingdino>=0.2.0",
        "rai-sam2>=1.1.2",
        "rai_core>=2.0.0a2,<3.0.0",
    ],
    zip_safe=True,
    maintainer="Kajetan Rachwał",
    maintainer_email="kajetan.rachwal@robotec.ai",
    description="Package for object detection, segmentation and gripping point detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    tests_require=["pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    entry_points={
        "console_scripts": [
            "talker = rai_perception.examples.talker:main",
        ],
    },
)
