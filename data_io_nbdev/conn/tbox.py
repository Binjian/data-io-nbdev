# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/04.conn.tbox.ipynb.

# %% auto 0
__all__ = ['g_tbox_sim_path', 'g_input_json_path', 'g_output_json_path', 'g_download_script_diffon', 'g_download_script_diffoff',
           'TBoxCanException', 'set_tbox_sim_path', 'float_to_hex', 'hex_to_float', 'float_array_to_buffer',
           'parse_arg', 'write_json', 'send_float_array']

# %% ../../nbs/04.conn.tbox.ipynb 3
import argparse
import json
import struct
import subprocess
import pandas as pd
from collections import UserDict
from dataclasses import dataclass, field
from typing import Optional

# %% ../../nbs/04.conn.tbox.ipynb 4
from ..system.decorator import prepend_string_arg

# %% ../../nbs/04.conn.tbox.ipynb 5
@dataclass(kw_only=True)
class TBoxCanException(Exception):
    """Base class for all TBox CAN exceptions (Kvaser exceptions).

    Args:

        err_code (int): error code
        extra_msg (str): extra message
        codes (UserDict): error code and message mapping
    """

    err_code: Optional[int] = 0  # default exception is unknown connection error
    extra_msg: Optional[str] = None
    codes: UserDict = field(default_factory=UserDict)

    def __post_init__(self):
        self.codes = UserDict(  # class attribute, if not given use the default
            {
                0: "success",
                1: "xcp download failure",
                2: "xcp internal error",
                3: "network_unknown_error",
                4: "xcp flashing timeout",
            }
        )
        # print(
        #     f"{{\'header\': \'err_code\': \'{self.err_code}\', "
        #     f"\'msg\': \'{self.codes[self.err_code]}\', "
        #     f"\'extra_msg\': \'{self.extra_msg}\'}}"
        #

# %% ../../nbs/04.conn.tbox.ipynb 6
g_tbox_sim_path = "/home/user/work/045b_demo/tbox-simulator"
g_input_json_path = ""
g_output_json_path = ""
g_download_script_diffon = ""
g_download_script_diffoff = ""

# %% ../../nbs/04.conn.tbox.ipynb 7
def set_tbox_sim_path(tbox_sim_path):
    global g_input_json_path
    global g_output_json_path
    global g_download_script_diffon
    global g_download_script_diffoff
    input_json_path = "/xcp_driver/json/example.json"
    output_json_path = "/xcp_driver/json/download.json"
    download_script_diffon = "/xcp_driver/scripts/download_diffon.sh"
    download_script_diffoff = "/xcp_driver/scripts/download_diffoff.sh"
    g_input_json_path = tbox_sim_path + input_json_path
    g_output_json_path = tbox_sim_path + output_json_path
    g_download_script_diffon = tbox_sim_path + download_script_diffon
    g_download_script_diffoff = tbox_sim_path + download_script_diffoff

# %% ../../nbs/04.conn.tbox.ipynb 8
set_tbox_sim_path(g_tbox_sim_path)

# %% ../../nbs/04.conn.tbox.ipynb 9
def float_to_hex(value):
    h = hex(struct.unpack(">I", struct.pack("<f", value))[0])
    return h

# %% ../../nbs/04.conn.tbox.ipynb 10
def hex_to_float(value):
    return float(struct.unpack(">f", struct.pack("<I", value))[0])

# %% ../../nbs/04.conn.tbox.ipynb 11
def float_array_to_buffer(float_array):
    buffer_value = ""
    for i in range(len(float_array)):
        hex_str = float_to_hex(float_array[i])[2:]
        if len(hex_str) < 8:
            diff = 8 - len(hex_str)
            hex_str = "0" * diff + hex_str
        buffer_value = buffer_value + hex_str
    return buffer_value

# %% ../../nbs/04.conn.tbox.ipynb 12
def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("example_json", help="example json file path")
    parser.add_argument(
        "-o",
        "--output",
        help="<Required> output json file name",
        required=True,
    )
    args = parser.parse_args()
    return args

# %% ../../nbs/04.conn.tbox.ipynb 13
def write_json(output_json_path, example_json_path, data):
    # 1 read example json
    f = open(example_json_path, "r")
    json_obj = json.load(f)
    f.close()
    # 2 write values to json object
    for item in data:
        name = item["name"]
        value = item["value"]
        for i in range(len(json_obj["data"])):
            if json_obj["data"][i]["name"] == name:
                dim = json_obj["data"][i]["dim"]
                value_length = int(json_obj["data"][i]["value_length"])
                length = 1
                for d in dim:
                    length = length * int(d)
                if len(value) != length * value_length * 2:
                    print(len(value))
                    print(length * value_length * 2)
                    print("value length does not match")
                    return
                json_obj["data"][i]["value"] = value
    # 3 write output json
    f = open(output_json_path, "w")
    json_str = json.dumps(json_obj)
    f.write(json_str)
    f.close()

# %% ../../nbs/04.conn.tbox.ipynb 14
@prepend_string_arg("TQD_trqTrqSetNormal_MAP_v")
def send_float_array(
    name: str,  # string for the CAN message name
    float_df: pd.DataFrame,  # the torque table to be flashed onto VBU
    sw_diff: bool = False,  # whether to use diff mode to accelerate flashing
) -> None:
    """
    send float array to tbox simulator

    the decorator prepend_string_arg is to set the default CAN ID for flashing torque table
    send_float_array(name, float_array, sw_diff) --> send_float_array(float_array, sw_diff)
    """

    float_array = float_df.to_numpy().reshape(-1).tolist()
    value_str = float_array_to_buffer(float_array)
    data = [{"name": name, "value": value_str}]
    write_json(g_output_json_path, g_input_json_path, data)
    try:
        if sw_diff:
            xcp_download = subprocess.run(
                [g_download_script_diffon], timeout=3, check=True
            )
        else:
            xcp_download = subprocess.run([g_download_script_diffoff], timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise TBoxCanException(
            err_code=4,
            extra_msg="xcp download timeout",
        )
    except subprocess.CalledProcessError as e:
        raise TBoxCanException(
            err_code=2,
            extra_msg=f"xcp download failed: {e}",
        )
    except Exception as e:
        raise TBoxCanException(
            err_code=1,
            extra_msg=f"xcp download failed: {e}",
        )

    # # print("The exit code was: %d" % xcp_download.returncode)
    # if xcp_download.returncode != 0:
    #     raise TBoxCanException(
    #         err_code=1,
    #         extra_msg="xcp download failed",
    #     )
