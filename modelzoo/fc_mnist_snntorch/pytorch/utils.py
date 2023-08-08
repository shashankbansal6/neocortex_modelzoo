# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """
    
    params["train_input"]["shuffle"] = params["train_input"].get(
        "shuffle", True
    )
    params["eval_input"]["shuffle"] = params["eval_input"].get("shuffle", False)

    params["model"]["to_float16"] = params["model"].get("to_float16", False)
    
    _device = params["runconfig"].get("target_device", False)
    if _device and _device.lower() == 'gpu':
        device = torch.device("cuda")
    else:
        device =torch.device("cpu")
    params["model"]["device"] = device