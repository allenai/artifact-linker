## pip install smolagents[toolkit,docker]
import base64
import json
import pickle
import re
import time
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any

from smolagents.monitoring import AgentLogger
from smolagents.remote_executors import DockerExecutor, RemotePythonExecutor

### notebook requirements 
REQUIREMENTS = [
]

logger = AgentLogger()
notebook = DockerExecutor(REQUIREMENTS,logger)  

