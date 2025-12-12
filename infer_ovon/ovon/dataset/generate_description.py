import pickle
import random
import time
from copy import deepcopy
from math import ceil
from typing import List

import openai
from tqdm import tqdm

PROMPT_2D = """
{}
Based on this dictionary which contains information about 2d bounding boxes given in the form of (xmin,ymin,xmax,ymax) in a view of the target object where (0,0) is the top left cornor of the frame, write an language instruction describing the location of the target object, {}, spatially relative to other objects as references.
Don't use any absolute values of the numbers, only use relative directions. Think of it as giving an instruction to a robot agent based on these coordinates. Add a prefix "Instruction: Find the .." or "Instruction: Go to .." to the generated instruction.
"""

PROMPT_3D = """
{}
Based on this dictionary which contains information about 3d bounding boxes of objects with center in (x,y,z) and size of the bounding box along x,y,z given in sizes_x_y_z,
write an language instruction describing the location of {} spatially relative to other objects as references. Don't use any absolute values of the numbers, only use relative directions. Think of it as giving an instruction to a robot agent based on these coordinates.
"""

PROMPT_BOTH = """
{}
Based on this dictionary which contains information about 3d bounding boxes of objects with center in (x,y,z) and size of the bounding box along x,y,z axes given in sizes_x_y_z and 2d bounding boxes given in the form of (xmin,ymin,xmax,ymax) in this particular view,
write an language instruction describing the location of {} spatially relative to other objects as references. Don't use any absolute values of the numbers, only use relative directions. Think of it as giving an instruction to a robot agent based on these coordinates.
"""


def create_html(
    file_name: str,
    objects: List,
) -> None:
    html_head = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Objects Generated Relationships</title>
    </head>
    """
    html_style = """
    <style>
        /* Three image containers (use 25% for four, and 50% for two, etc) */
        .column {
        float: left;
        width: 20.00%;
        padding: 5px;
        }

        /* Clear floats after image containers */
        .box {
        box-sizing: border-box;
        }
        .row {
        display: flex;
        }
    </style>
    """
    html_script = """
    <script>
        var li_relationships = []
        function addRelationships(cb) {
        if (cb.checked) {
            li_relationships.push(cb.id);
        }
        else {
            var index = li_relationships.indexOf(cb.id);
            if (index > -1) {
                li_relationships.splice(index, 1);
            }
        }
        localStorage.setItem("relationships",li_relationships)
        }
    </script>
    """
    len(objects)
    html_body = """<body>
        <h2> Visualising {cnt} Relationships </h2>
        """
    for i, info_dict in enumerate(objects):
        name = info_dict["target_obj_name"]
        id = info_dict["target_obj_id"]
        img_ref = info_dict["img_ref"]
        scene = info_dict["scene"]
        ins_1 = info_dict["Instruction_1"]
        ins_2 = info_dict["Instruction_0.5"]
        ins_3 = info_dict["Instruction_0.75"]

        if i % 3 == 0:
            html_body += """<div class="row">"""
        html_body += f"""
                    <input type="checkbox" id="{scene}_{name}_{id}" name="{scene}_{name}" onclick=addRelationships(this);>
                    <div class="column">
                        <img src="{img_ref.replace('data/','../../../')}" alt="{img_ref}" style="width:100%">
                        <h5>{name}_{id},\n {ins_1} \n {ins_2} \n {ins_3}</h5>
                    </div>
                    """
        if i % 3 == 2:
            html_body += "</div>"

    html_body += """</body>
                </html>"""
    f = open(file_name, "w")
    f.write(html_head + html_style + html_script + html_body)
    f.close()


def gpt_call(
    model="gpt-3.5-turbo",
    prompt="",
):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        response = None
        print(e)
    return response


def prompt_for_obj(obj_dict, type="2d", drop=0):
    PROMPT = ""
    obj_dict.pop("scene")
    obj_dict.pop("img_ref")
    obj_dict.pop("target_obj_color")

    keys_list = list(obj_dict["ref_objects"].keys())
    keys_to_pop = random.sample(keys_list, ceil(len(keys_list) * drop))
    for key in keys_to_pop:
        obj_dict["ref_objects"].pop(key)

    if type == "2d":
        obj_dict.pop("target_obj_3d_bb")
        for key, ref in obj_dict["ref_objects"].items():
            ref.pop("3d_bb")
        PROMPT = PROMPT_2D
    elif type == "3d":
        obj_dict.pop("target_obj_2d_bb")
        for ref in obj_dict["ref_objects"].values():
            ref.pop("2d_bb")
        PROMPT = PROMPT_3D
    else:
        PROMPT = PROMPT_BOTH

    target_obj = obj_dict["target_obj_name"]
    prompt = PROMPT.format(obj_dict, target_obj)
    return prompt


def generate_description_for_scene(meta_file_path, openai=True):
    drop_rate = [0, 0.25, 0.5]
    with open(meta_file_path, "rb") as f:
        object_dict_list = pickle.load(f)
    cnt = 0
    current_list = deepcopy(object_dict_list)

    for obj_dict in tqdm(current_list):
        for d in drop_rate:
            temp_obj_dict = deepcopy(obj_dict)
            prompt = prompt_for_obj(temp_obj_dict, drop=d)
            if openai:
                response = gpt_call(prompt=prompt)
                if response is None:
                    obj_dict[f"Instruction_{1-d}"] = "API_failure"
                    cnt += 1
                    print("API Call failed!")
                else:
                    obj_dict[f"Instruction_{1-d}"] = response["choices"][0]["message"][
                        "content"
                    ]
            else:
                inputs = alpaca_tokenizer(prompt, return_tensors="pt")
                out = alpaca_model.generate(inputs=inputs.input_ids, max_new_tokens=100)
                output_text = alpaca_tokenizer.batch_decode(
                    out, skip_special_tokens=True
                )[0]
                output_text = output_text[len(inputs) :]
                obj_dict[f"Instruction_{1-d}"] = output_text
                print(output_text)
        time.sleep(61)
    print(f"API call failed for {cnt} out of {len(object_dict_list)} objects!")
    return current_list


if __name__ == "__main__":
    openai.api_key = ""
    # alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("/srv/cvmlp-lab/flash1/akutumbaka3/hf_alpaca")
    # alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("/srv/cvmlp-lab/flash1/akutumbaka3/hf_alpaca")
    # print("Model has been loaded")
    scenes_1 = ["vLpv2VX547B"]
    scenes_5 = [
        "U3oQjwTuMX8",
        "b3WpMbPFB6q",
        "aRKASs4e8j1",
        "iePHCSf119p",
        "JptJPosx1Z6",
    ]
    for scene in scenes_1:
        object_view_meta_file = (
            "/nethome/akutumbaka3/files/ovonproject/data/object_views/train/meta/{}.pkl"
            .format(scene)
        )
        final_list = generate_description_for_scene(object_view_meta_file)
        webpage_path = f"/nethome/akutumbaka3/files/ovonproject/data/object_views/train/webpage/{scene}.html"
        create_html(file_name=webpage_path, objects=final_list)
        with open(
            f"/nethome/akutumbaka3/files/ovonproject/data/object_views/train/annotated/{scene}.pkl",
            "wb",
        ) as f:
            pickle.dump(final_list, f)
