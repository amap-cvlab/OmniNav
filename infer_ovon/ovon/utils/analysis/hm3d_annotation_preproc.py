#!/usr/bin/env python

# This script will copy Appen annotated files into an existing HM3D dataset,
# along with making the necessary modifications to the semantic txt file to
# be compatible with Habitat sim.

import math
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from os.path import exists as os_exists
from os.path import isfile as os_isfile
from os.path import join as os_join
from os.path import sep as os_sep
from os.path import split as os_split
from typing import Any, Dict, List, Optional

import config_utils as ut

#
# Destination where HM3D datasets reside.  Under this directory should be the
# various partition directories, and under each of these should be the
# per-scene directories for the scenes belonging to that partition,
# having the format
#   <XXXXX>-<YYYYYYYYYYY>, where
#     <XXXXX> represents a numeric value from 00000 -> 00999
#     <YYYYYYYYYYY> represents an alpha-numeric hash representing the scene

HM3D_DEST_DIR = "data/scene_datasets/hm3d"

#
# Source directory for Appen annotations.  Under this directory should reside individual
# sub directories for each annotated scene, where each sub directory has the format
#   <XXXXX>-<YYYYYYYYYYY>.semantic/Output/
# and the Output sub directory contains at least 2 files that vary only by extension :
#   <YYYYYYYYYYY>.semantic.<glb|txt>, where
#     <XXXXX> represents a numeric value from 00000 -> 00999. This should be the numeric
#             tag for the particular scene, but it does not have to be.  5 0's will suffice
#             if the tag is unknown.
#     <YYYYYYYYYYY> represents an alpha-numeric hash representing the scene.
#
# Source files can also follow format of existing HM3D dataset. Use this if we wish to run the
# reporting capabilities of this script on existing files
# This format is specified above in Destination description
#
# The expected files and what they represent are
#   .txt is the Semantic Scene descriptor file, which holds the information mapping vertex
#        colors to labels and regions
#   .glb is the mesh file for the scene holding both vertex-color-based and texture-based
#        semantic annotations.

# #All the scenes from Appen - put redone scenes in here only when they have been vetted and found
# #to be better than the existing alternatives.
# HM3D_ANNOTATION_SRC_DIR = "/home/john/Datasets In Progress/HM3D_Semantic/Appen_Scenes"
HM3D_ANNOTATION_SRC_DIR = (
    "/home/john/Datasets In Progress/HM3D_Semantic/Alex_Scenes_Sept_22"
)

# HM3D_ANNOTATION_SRC_DIR = "/home/john/Datasets In Progress/HM3D_Semantic/John_Test"

# #The scenes used for the challenge (151)
# HM3D_ANNOTATION_SRC_DIR = (
#     "/home/john/Datasets In Progress/HM3D_Semantic/Appen_Scenes_Challenge"
# )

#
# Desitnation directory to put results of various reporting processes
HM3D_ANNOTATION_RPT_DEST_DIR = "/home/john/Datasets In Progress/HM3D_Semantic"
#
# Directory to find various data files used to perform tasks
HM3D_ANNOTATION_DATA_DIR = (
    "/home/john/Datasets In Progress/HM3D_Semantic/Script_Exec_Data"
)


# Appen annotation source scene directory regex.
# This regex describes the format of the per-scene directories in the Appen work,
# and may change depending on whether the format we receive from Appen changes.
HM3D_SRC_ANNOTATION_SUBDIR_RE = r"(?i)[0-9]{5}-[a-z0-9]{11}\.semantic$"
# Alternate source path matching, for processing file hierarchies that are already
# in the format used by the HM3D dataset
HM3D_SRC_ANNOTATION_SUBDIR_RE_2 = r"(?i)[0-9]{5}-[a-z0-9]{11}$"

#
# Prefix for config files - leave empty string for none. Use this to build configs on a
# subset of scenes for testing without having to view all scenes.
# Make this an empty string if building for entire dataset of annotated scenes.
CONFIG_PREFIX = ""
# Put some stub here to identify the source material used
# CONFIG_PREFIX = "April_25_redos"

##################################################################
## Desired functions to perform


# What function to perform
# PROCESS_AND_COPY_SRC : Copy the src files (either given in Appen format or already in
#     dataset format) to HM3D Dest directory.  Adds sentinel tag at the start of the
#     semantic.txt files if needed. Corrects some specific errors in semantic.txt such as
#     ID numbers out of order, reports on other errors, such as per-line values not matching
#     expected format.  Will also correct for tag spelling if VALIDATE_TAG_SPELLING is
#     set to True
# COUNT_SEMANTIC_COLORS : count instances of unique semantic colors and tags in all scenes
#     in source dir and save output report file describing results. No files are moved to
#     dest dataset location
# CALC_REGIONS : build mapping of region ID numbers to proposals of region names based on
#     annotations found in each region.
#     In order to minimize modifications to Habitat-Sim engine, we will write this mapping
#     in the scene instance files for each of the various scenes, in the user_defined tag
#     for the stage instance.  These scene instance files will be moved to appropriate
#     locations in HM3D_DEST_DIR. This will also create a txt file listing the relative
#     paths of the scene instance files for easy archiving or removal.
class Functions_Available(Enum):
    PROCESS_AND_COPY_SRC = 1
    COUNT_SEMANTIC_COLORS = 2
    CALC_REGIONS = 3


FUNC_TO_DO = Functions_Available.PROCESS_AND_COPY_SRC

##################################################################
# Functions_Available.CALC_REGIONS settings :

#
# Filename holding mapping of tag names to proposed region names.
REGION_PROPOSAL_MAP_NAME = "HM3D_ObjToRegionNameMap.csv"

#
# Potential region names
POSSIBLE_REGION_NAMES = {
    "bathroom",
    "bedroom",
    "dining room",
    "garage",
    "hall/stairwell",
    "kitchen",
    "laundry room",
    "living room",
    "office",
    "rec room",
    "unknown room",
}

#
# Build a set of region-name agnostic tags that do not help region specification
IGNORE_TAGS = set()
IGNORE_TAGS.add("ceiling")
IGNORE_TAGS.add("door")
IGNORE_TAGS.add("floor")
IGNORE_TAGS.add("unknown")
IGNORE_TAGS.add("wall")
IGNORE_TAGS.add("window")
IGNORE_TAGS.add("window frame")

#
# Whether to count cat presence or obj presence per region for voting
COUNT_OBJS_FOR_VOTES = False

#
# Whether to create the scene instance config jsons with the region mappings in the stage
# attributes user defined tag.
SAVE_SCENE_CONFIG_REGION_MAP = False

##################################################################
# Functions_Available.PROCESS_AND_COPY_SRC settings :

#
# Copy scene glbs over to dest. Set this to false if we just wish to remake and move semantic txt files
COPY_GLBS_TO_DEST = False

#
# Whether to load semantic text spelling map file and verify/correct tag spelling.
# This file should be located in parent directory of src
VALIDATE_TAG_SPELLING = True

#
# Whether or not to build annotation scene dataset configs.  These are main scene dataset config
# and individual partition scene dataset configs
BUILD_SD_CONFIGS = True

#
# Whether or not to save source file names in annotation file lists for
# individual partitions when moving files (i.e. basis.glb and navmesh file names). This is so
# that the source files can be easily added to any archive built.
SAVE_SRC_TO_FILESLISTS = True


##############################################################################
## You should not need to modify anything below here


# numeric spans of each data partition.  The directoriues
# holding the scene data are prefixed by numbers in these ranges
HM3D_DATA_PARTITIONS = {
    "minival": (800, 809),
    "test": (900, 999),
    "train": (0, 799),
    "val": (800, 899),
}

# Sentinel string for semantic scene descriptor text files in HM3D
# DO NOT CHANGE WITHOUT CHANGE IN HABITAT-SIM.  This is looked for in Habitat-sim.
# This is added to the first line of the semantic.txt files if it does not already
# exist
HM3D_SSD_STRING = "HM3D Semantic Annotations"


# Validate source semantic txt files' format
# Each line should have the following format :
# 1,4035BC,"ceiling corridor_2",2
# positive int, hex color, string, positive int
def validate_src_SSD(line):
    lineItems = line.split(",")
    # each line should be 4 elements, but element 2 might contain one or more commas
    # so check if there are 4 items, or there are more
    if (len(lineItems) < 4) or (
        len(lineItems) > 4
        and lineItems[2].count('"') != 1
        and lineItems[-2].count('"') != 1
    ):
        return f"Incorrect # of items : should be 4, is {len(lineItems)}"
    if not lineItems[0].strip().isdigit():
        return f"First entry (unique ID) is not positive integer : {lineItems[0]}"
    if not re.compile("^[a-fA-F0-9]{6}$").match(lineItems[1].strip()):
        return f"Second entry (color) is not 6 digit hex value : {lineItems[1]}"
    if len(lineItems[2].strip()) == 0:
        return "Third entry (category) cannot be an empty string"
    if "," in lineItems[2]:
        return "Third entry (category) contains a comma!"
    if not lineItems[-1].strip().isdigit():
        return f"Last entry (region) is not non-negative integer : {lineItems[-1]}"
    return "fine"


# This set will hold all tags found in semantic.txt file that do not have valid mappings in
# the tag_map_dict.
ERROR_CAT_TAGS = set()


# Correct any misspelling of the category tag by checking against loaded mapping
# The mapping dict should have entries for -all- tags found in the semantic.txt file
def validate_cat_tag(line: str, tag_map_dict: dict):
    # isolate category tag
    category_tag = line.split(",", 2)[-1].strip().rsplit(",", 1)[0].strip()
    if tag_map_dict[category_tag] == "":
        # No entry found for specified category tag, so assigning "unknown"
        new_line = line.replace(category_tag, '"unknown"')
        print(
            f"No map entry for category tag in line : {line.strip()} : {category_tag} :"
            f" len : {len(category_tag)}: modified : {new_line} "
        )
        ERROR_CAT_TAGS.add(category_tag)
    else:
        if category_tag != tag_map_dict[category_tag]:
            tag_map_dict["temp_spelling_corrections_list"].append(category_tag)

        new_line = line.replace(category_tag, f"{tag_map_dict[category_tag]}")

    return new_line


# Modify the given semantic text file to include the necessary sentinel string at the
# beginning that Habitat-Sim expects, and then save to appropriate dest directory
def modify_and_copy_SSD(src_filename: str, dest_filename: str, tag_map_dict: dict):
    with open(dest_filename, "w") as dest, open(src_filename, "r") as src:
        # Save tag as first line - this is required by Habitat-Sim to verify semantic txt dataset
        dest.write(f"{HM3D_SSD_STRING}\n")
        i = 1
        printFileOffset = True
        fileIsOffset = False
        # set up an entry to hold how many corrections are made
        if VALIDATE_TAG_SPELLING:
            tag_map_dict["temp_spelling_corrections_list"] = []
        for line in src:
            # If initial tag exists, just ignore it and continue - means we're looking over
            # existing dataset semantic file
            if HM3D_SSD_STRING in line:
                continue
            valid_res = validate_src_SSD(line)
            if valid_res != "fine":
                print(
                    f"!!!! Error in source SSD `{src_filename}` on line {i} :"
                    f" {line} has format error : `{valid_res}`"
                )
            if VALIDATE_TAG_SPELLING:
                # Fix misspellings of category tag and rebuild line appropriately
                line = validate_cat_tag(line, tag_map_dict)

            line_parts = line.split(",", 1)
            if int(line_parts[0].strip()) != i:
                fileIsOffset = True
                line = f"{i},{line_parts[-1]}"

            if fileIsOffset and printFileOffset:
                print(
                    f"Erroneous line in source SSD `{src_filename}` corrected {i} :"
                    f" {line}",
                    end="",
                )
                fileIsOffset = False
                printFileOffset = False
            dest.write(f"{line}")
            i += 1
    if VALIDATE_TAG_SPELLING:
        num_spell_corrections = len(tag_map_dict["temp_spelling_corrections_list"])
        print(
            f"# of Spelling corrections made when saving {dest_filename} :"
            f" {num_spell_corrections}"
        )
        # file_name =
        # tag_map_dict['_num_spell_corrections']


# Build dataset configs for HM3D annotation dataset, along with individual partitions
def build_annotation_configs(part_file_list_dict: dict):
    print("build_annotation_configs:")

    def modify_JSON_paths_tag(
        paths: Dict, file_ext_key: str, path_glob_file: str, rel_file_dirs: List
    ):
        # replace existing list of paths in paths dict with rel_file_names list
        tmp_paths = []
        for file_path in rel_file_dirs:
            tmp_paths.append(os_join(file_path, path_glob_file))
        paths[file_ext_key] = tmp_paths

    # get the filenames of the 5 existing configs
    src_json_configs = ut.get_files_matching_regex(
        HM3D_DEST_DIR, re.compile("hm3d_[^aA].*[_bas]\\.scene_dataset_config\\.json")
    )
    # print("Source json configs : ")
    # for config in src_json_configs:
    #     print(f"\t{config[0]}/{config[-1]}")

    file_dirs_for_config = {}
    tmp_glbl_filedirs = set()
    # only want single representation for each scene
    for key in HM3D_DATA_PARTITIONS:
        tmp_file_dirs = set()
        for filename in part_file_list_dict[key]:
            path_no_filename = os_split(filename)[0] + os_sep
            tmp_file_dir = path_no_filename.split(f"{key}/")[-1]
            tmp_file_dirs.add(tmp_file_dir)
            tmp_glbl_filedirs.add(path_no_filename)
        file_dirs_for_config[key] = sorted(tmp_file_dirs)

    file_dirs_for_config["base"] = sorted(tmp_glbl_filedirs)

    config_filenames = {}
    new_config_filenames = {}
    new_filepaths_for_config = {}
    # build new filenames for annotation configs
    if len(CONFIG_PREFIX) == 0:
        new_config_file_prefix = "hm3d_annotated_"
    else:
        new_config_file_prefix = f"hm3d_annotated_{CONFIG_PREFIX}"
    for config in src_json_configs:
        filename = os_join(config[0], config[-1])
        # want to overwrite old versions
        if "hm3d_annotated_" in filename:
            continue
        # success = os_exists(filename) and os_isfile(filename)
        json_filename = config[-1]
        config_filenames[json_filename] = filename
        new_config_filenames[json_filename] = filename.replace(
            "hm3d_", new_config_file_prefix
        )

        # set default path list
        new_filepaths_for_config[json_filename] = file_dirs_for_config["base"]
        for part, scene_path_list in file_dirs_for_config.items():
            if part in json_filename:
                new_filepaths_for_config[json_filename] = scene_path_list
                break

    for config_key, src_config_filename in config_filenames.items():
        dest_config_filename = new_config_filenames[config_key]
        scene_path_list = new_filepaths_for_config[config_key]
        if len(scene_path_list) == 0:
            continue
        print(
            f"{config_key} # files : {len(scene_path_list)} :"
            f" \n\t{src_config_filename}\n\t{dest_config_filename}"
        )

        # load each existing json config, appropriately modify it, and then save as new configs
        if BUILD_SD_CONFIGS:
            src_json_config = ut.load_json_into_dict(src_config_filename)
            for key, json_obj in src_json_config.items():
                # Modify only those configs that hold paths
                if "paths" in json_obj:
                    # Modify both stages and scene instance configs
                    # this is the dictionary of lists of places to look for specified config type files
                    paths_dict = src_json_config[key]["paths"]
                    for path_type_key, lu_path_list in paths_dict.items():
                        # assume the list has at least one element
                        lu_glob_file = lu_path_list[0].split(os_sep)[-1]
                        modify_JSON_paths_tag(
                            paths_dict, path_type_key, lu_glob_file, scene_path_list
                        )
                # add tag/value in stages dflt attributes denoting the
                # annotation dataset supports texture semantics
                if "stages" in key:
                    dflt_attribs = src_json_config[key]["default_attributes"]
                    dflt_attribs["has_semantic_textures"] = True

            ut.save_json_to_file(src_json_config, dest_config_filename)

        # add subdirectory-qualified file paths to new configs to file list dictionary `base` key so that they will
        # be included in zip file
        rel_config_filename = dest_config_filename.split(HM3D_DEST_DIR)[-1].split(
            os_sep, 1
        )[-1]
        print(
            f"Adding rel_config_filename : {rel_config_filename} to file list"
            " dictionary `base` key."
        )
        part_file_list_dict["base"].append(rel_config_filename)


# Save a text file holding all the relative paths to the various semantic.txt, semantic.glb and other
# assets for each semantically annotated scene, for easy archiving.  Includes per-partition file listings.
def save_annotated_file_lists(part_file_list_dict: dict):
    print("save_annotated_file_lists: ", end="")
    # build output_list from dictionary of partition-based file names
    output_files = [
        val for _, sublist in part_file_list_dict.items() for val in sublist
    ]

    # write text files that hold listings of appropriate relative filepaths for
    # annotated files as well as for each partition's configs for all scenes that
    # have annotations

    # single file listing holding all annotated file filenames (annotated glb and txt only - no source)
    # and annotated config filenames
    if SAVE_SRC_TO_FILESLISTS:
        use_src_prfx = "_and_src"
    else:
        use_src_prfx = ""
    if len(CONFIG_PREFIX) == 0:
        new_file_list_prefix = f"HM3D_annotation{use_src_prfx}"
    else:
        new_file_list_prefix = f"HM3D_annotation{use_src_prfx}_{CONFIG_PREFIX}"

    with open(
        os_join(
            HM3D_DEST_DIR, f"{new_file_list_prefix.replace(use_src_prfx, '')}_files.txt"
        ),
        "w",
    ) as dest:
        dest.write("\n".join(sorted(output_files)))

    # save listing of all files (semantic, render, navmesh) for each partition:
    # each entry is a tuple with idx 0 == dest filename; idx 1 == list of paths
    partition_lists = {}
    for part in HM3D_DATA_PARTITIONS:
        partition_lists[part] = (
            os_join(HM3D_DEST_DIR, f"{new_file_list_prefix}_{part}_dataset.txt"),
            [],
        )
    # for each entry in output_files list, check which partition it belongs to
    # all partition files should have hm3d-<partition>-habitat as the lowest
    # relative directory
    for filename in sorted(output_files):
        # print(f"filename : {filename}")
        fileparts = filename.split(os_sep)
        if len(fileparts) < 2:
            # filename is root level and not part of a partition
            continue
        file_part_key = fileparts[0]  # .split("-")[1].strip()
        partition_lists[file_part_key][1].append(filename)
        # add source file filenames to filelisting if desired to build archive of all source files
        # checking for '.semantic.txt' to only perform this once, and only for valid scenes (not configs)
        if SAVE_SRC_TO_FILESLISTS and ".semantic.txt" in filename:
            filename_base = filename.split(".semantic.txt")[0]
            partition_lists[file_part_key][1].append(f"{filename_base}.basis.glb")
            partition_lists[file_part_key][1].append(f"{filename_base}.basis.navmesh")
    # write each per-partition file listing to build per-partition annotated-only dataset archives
    # (holding semantic, render and navmesh assets)
    for _, v in partition_lists.items():
        if len(v[1]) == 0:
            continue
        with open(v[0], "w") as dest:
            dest.write("\n".join(v[1]))
    print("Success!!")


# Process and copy source files to destination directory, verifying process completed recording all files written
def proc_and_copy_file(
    src_dir: str,
    data_subdict: dict,
    type_key: str,
    failures: dict,
    part_file_list: list,
    tag_spelling_dict: dict,
):
    # This will verify that the passed file exists and is a file. Use to verify copying result.
    def verify_file(filename: str, src_dir: str, type_key: str, failures: Dict):
        success = os_exists(filename) and os_isfile(filename)
        proc = "has"
        if not success:
            proc = "HAS NOT"
            failures[src_dir][type_key] = filename
        if "SSD" in type_key:
            print(
                f"\t\tSSD File {proc} been successfully modified and copied to"
                f" {filename}"
            )
        else:
            print(f"\t\tGLB File {proc} been successfully copied to {filename}")
        return success

    # destination full file path
    dest_filename = data_subdict["dest_full_path"]
    if "SSD" in type_key:
        # Appropriately modify SSD text file to be in Habitat format and copy to appropriate destination
        modify_and_copy_SSD(
            data_subdict["src_full_path"], dest_filename, tag_spelling_dict
        )
    else:
        # Copy glb asset to HM3D Destination directly
        shutil.copy(data_subdict["src_full_path"], dest_filename)

    success = verify_file(dest_filename, src_dir, type_key, failures)
    if success:
        part_file_list.append(data_subdict["dest_subdir_file"])


# Load and prune dictionary containing spelling errors to be corrected in tags
def load_tag_spelling_dict():
    # This dictionary maps possible tags to be found in annotations to correct
    # tags that should be used in semantic txt file.
    tag_map_filename = os_join(
        HM3D_ANNOTATION_DATA_DIR, "HM3D_Semantic_Misspelling_Map.csv"
    )
    # success = os_exists(tag_map_filename) and os_isfile(tag_map_filename)
    # print(f"Filename : {tag_map_filename} | File exists? : {success}")
    tag_map_dict = defaultdict(lambda: "")
    lines = []
    with open(tag_map_filename, "r") as src:
        lines = src.readlines()
    for line in lines:
        # Preserve quote locations to preserve leading and trailing whitespace in
        # hand annotations so this can be corrected for if appropriate
        line_list = line.strip().rsplit(",", 1)
        tag_map_dict[line_list[0]] = line_list[1]
        # print(f"Key :`{line_list[0]}`    \t|     val :`{line_list[1]}`")
    # print(f"Num entries : {len(tag_map_dict)}")
    # Missing annotations are forced to unknown
    tag_map_dict['""'] = '"unknown"'
    return tag_map_dict


# Process all SSD / semantic.txt files and copy them to dest dir, along with (possibly) all src glbs.
# Build json scene dataset configs for base dataset and partitions
# Build txt files holding file listings for easy archiving.
def process_and_copy_files(file_names_and_paths: dict):
    # dictionary keyed by partition valued by list of partition subdir and filename of written file
    part_file_list_dict = defaultdict(lambda: [])

    # Failures here will be files that did not get copied (or modified if appropriate)
    failures = defaultdict(lambda: {})
    # Load dictionary mapping misspellings in tags to proper spellings
    tag_spelling_dict = load_tag_spelling_dict()

    # build dictionary of lists for spelling corrections
    tag_spell_corr_dict = defaultdict(lambda: [])

    # move semantic glbs and scene descriptor text files
    for src_dir, data_dict_list in sorted(file_names_and_paths.items()):
        # print(f"Src : {src_dir} : # of data_dicts : {len(data_dict_list)} ", end="")
        for data_dict in data_dict_list:
            # print(f"{data_dict}")
            partition_tag = data_dict["dest_part_tag"]
            # modify src SSD and save to dest
            proc_and_copy_file(
                src_dir,
                data_dict["SSD"],
                "SSD",
                failures,
                part_file_list_dict[partition_tag],
                tag_spelling_dict,
            )
            # if correcting spelling of semantic tags, save list of corrected spelling errors
            # List of orriginal mistakes is at tag_spelling_dict['temp_spelling_corrections_list']
            if VALIDATE_TAG_SPELLING:
                tag_spell_corr_dict[data_dict["scene_name"]] = tag_spelling_dict[
                    "temp_spelling_corrections_list"
                ]
            if COPY_GLBS_TO_DEST:
                # Copy glb file to appropriate location
                proc_and_copy_file(
                    src_dir,
                    data_dict["GLB"],
                    "GLB",
                    failures,
                    part_file_list_dict[partition_tag],
                    tag_spelling_dict,
                )

    # save all missing category tags from spelling-correction mapping of semantic.txt files if performed
    if VALIDATE_TAG_SPELLING:
        with open(
            os_join(HM3D_ANNOTATION_RPT_DEST_DIR, "My_HM3D_Missing_Tags_Taxonomy.csv"),
            "w",
        ) as dest:
            for tag in sorted(ERROR_CAT_TAGS):
                dest.write(f"{tag}\n")

    # save per-scene corrections of tag spelling corrections
    if VALIDATE_TAG_SPELLING:
        with open(
            os_join(HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Scene_Spelling_Fixes.csv"),
            "w",
        ) as dest:
            dest.write("Scene name,Original Tag,Corrected Tag,Num Corrections\n")
            for scene_name, fix_list in sorted(tag_spell_corr_dict.items()):
                fix_dict = defaultdict(int)
                for tag in fix_list:
                    fix_dict[tag] += 1
                for tag, count in sorted(fix_dict.items()):
                    dest.write(f"{scene_name},{tag},{tag_spelling_dict[tag]},{count}\n")

    num_files_written = 0

    for _, files in part_file_list_dict.items():
        num_files_written += len(files)
    print(
        f"# of src files processed : {len(file_names_and_paths)} | # of dest files"
        f" written : {num_files_written} | # of failures : {len(failures)}\n ",
        end="",
    )
    # Get relative paths to all 5 annotation configs (base and each partition), as well as
    # build scene dataset configs,if requested
    build_annotation_configs(part_file_list_dict)

    # save filenames of the annotation files that have been written to a txt file in the dest dir,
    # to facilitate archiving
    save_annotated_file_lists(part_file_list_dict)

    # display failures if they have occurred
    if len(failures) > 0:
        print("\n!!!!! The following files failed to be written : ", end="")
        for src_dir, fail_dict in failures.items():
            for file_type, file_name in fail_dict.items():
                print(
                    f"Src : {src_dir} :\n\tType : {file_type} | Filename :"
                    f" {file_name} ",
                    end="",
                )


###########################################################
# Annotation color reports


# Parse passed SSD file and build a dictionary of information.
#       Key
# 1. "color_data" : Find colors and instance tags for all annotations in the passed SSD text file for a single scene.
# 2. "region_data" : Find the regions
#
# WARNING : more than one representation/tag of any single color in the same annotation txt
# file will cause mislabeling, since only the tag of only the last instance of any color will
# be processed by Habitat-Sim
# Return : Default tictionary keyed by color with value being dictionary holding
# count of instances and list of object tags for each color in the given scene
def parse_SSD_file(ssd_filename: str):
    res_dict = {}
    counts_dict = defaultdict(lambda: {"count": 0, "names": []})
    regions_dict = defaultdict(lambda: defaultdict(lambda: []))
    # count the # of occurrences of colors in each scene. SHOULD ONLY BE 1 PER COLOR!
    with open(ssd_filename, "r") as src:
        for line in src:
            # If initial tag exists, just ignore it and continue - means we're looking over existing dataset semantic file
            if HM3D_SSD_STRING in line:
                continue
            items = line.split(",")
            color = items[1].strip()
            # All annotation tags should have leading and trailing quotes in SSD
            tag = line.split('"')[1].strip()
            name = f"{tag}_{items[0]}"
            counts_dict[color]["names"].append(name)
            region = int(items[-1].strip())
            regions_dict[region][tag].append(name)
            if not re.compile("^[a-fA-F0-9]{6}$").match(color):
                counts_dict[color]["count"] = -1
                print(
                    f"WARNING!!! : In file :{ssd_filename} : Color has bad form :"
                    f" {color} : names : {counts_dict[color]['names']}"
                )
            else:
                counts_dict[color]["count"] += 1
    res_dict["color_data"] = counts_dict
    res_dict["region_data"] = regions_dict
    return res_dict


# Build dictionary keyed by scene name of semantic data, specified by data_key
# Currently supports :
# "color_data" : per-scene dictionary keyed by color holding counts and names of objects of that color
# "region_data" : per scene dictionary keyed by region number in that scene, holding names of objects
#                 in that region
def build_per_scene_SSD_data(file_names_and_paths: dict, data_key: str):
    per_scene_counts = {}
    for _, data_dict_list in sorted(file_names_and_paths.items()):
        for data_dict in data_dict_list:
            scenename = data_dict["scene_name"]
            ssd_filename = data_dict["SSD"]["src_full_path"]
            # print(f"{scenename} scene name from {ssd_filename}")
            per_scene_counts[scenename] = parse_SSD_file(ssd_filename)[data_key]

    print(
        f"\n{len(per_scene_counts)} Scenes have been processed for {data_key} counts.\n"
    )
    return per_scene_counts


# Count all the colors and annotations in the scenes by going through every
# semantic annotation file and recording every color present, along with the object tag.
#
# WARNING : more than one representation/tag of any single color in the same annotation txt
# file will cause mislabeling, since only the tag of only the last instance of any color will
# be processed by Habitat-Sim
#
# Return : dictionary keyed by scene name, where value is default dict keyed by color, value
def per_scene_clr_counts(file_names_and_paths: dict):
    # build dictionary holding per-scene color counts and object names
    per_scene_counts = build_per_scene_SSD_data(file_names_and_paths, "color_data")

    print(f"\n{len(per_scene_counts)} Scenes have been processed for color counts.\n")
    # display results
    total_items = 0
    for scenename, count_dict in per_scene_counts.items():
        for color, count_and_names in count_dict.items():
            count = count_and_names["count"]
            if count > 0:
                total_items += count
            names = count_and_names["names"]
            if count > 1:
                print(
                    f"!!! In scene {scenename} : Bad Color Count : '{color}' is present"
                    f" {count} times : {names}"
                )
            elif count < 0:
                print(
                    f"!!!!!!!! In scene {scenename} : Count for color : '{color}' is"
                    f" less than 0 : {count} times : {names}. This is due to erroneous"
                    " color being found. "
                )
    print(f"Total Items :{total_items}\n", end="")
    return per_scene_counts


# Save generated per-category (acquired by splitting annotation name) counts and names
def save_obj_cat_counts(dest_filename: str, obj_types_per_count: dict):
    with open(dest_filename, "w") as dest:
        dest.write("Object Type Name; # of instances in semantic text files\n")
        for _, obj_types_and_names in sorted(obj_types_per_count.items(), reverse=True):
            # print(f"Object types present {count} times :")
            for objtype, namesList in obj_types_and_names:
                # allNamesList = ";".join(namesList)
                # dest.write(f"{objtype};{len(namesList)};{allNamesList}\n")
                dest.write(f"{objtype};{len(namesList)}\n")


# Build reports based on color annotations in semantic.txt files
def color_annotation_reports(file_names_and_paths: dict):
    # go through every semantic annotation file and perform a count of each color present.
    # Dictionary keyed by scene, value is
    per_scene_counts = per_scene_clr_counts(file_names_and_paths)

    # build dictionary keyed by object "types", holding dataset-wide counts and object names.
    obj_names_per_objtype = defaultdict(lambda: {"count": 0, "names": []})
    for _, count_dict in per_scene_counts.items():
        for _, count_and_names in count_dict.items():
            names = count_and_names["names"]
            for name in names:
                name_parts = name.split("_")
                objtype = name_parts[0].strip()
                if not name_parts[1].strip().isdigit():
                    print(
                        f"Object for objtype {objtype} is not parsed properly on `_` :"
                        f" complete name is {name}"
                    )
                obj_names_per_objtype[objtype]["count"] += 1
                obj_names_per_objtype[objtype]["names"].append(name)
    # display all object types and the counts and object names having that type
    print("\n\n")
    obj_types_per_count = defaultdict(lambda: [])
    for objtype, count_and_names in obj_names_per_objtype.items():
        count = count_and_names["count"]
        names = count_and_names["names"]
        obj_types_per_count[count].append((objtype, names))

    outputFileName = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "HM3D_CountsOfObjectTypes.csv"
    )
    save_obj_cat_counts(outputFileName, obj_types_per_count)

    print("ObjType as specified in annotation text :")
    for _, obj_types_and_names in sorted(obj_types_per_count.items(), reverse=True):
        # print(f"Object types present {count} times :")
        for objtype, names in obj_types_and_names:
            print(f"\t{objtype} | # names {len(names)}")


###########################################################
# Region Proposal derivation


def load_region_proposals():
    # load dictionary of region proposals based on object counts and add voting mechanism
    #
    # dictionary mapping tag to region information

    tmp_dict = build_cat_annotation_record_dict()
    # category name as key, value is dictionary holding vote of co-region occupants with specified known region name
    # need deep copy since tmp_dict may have sub-dicts
    derived_proposal_dict = defaultdict(lambda: deepcopy(tmp_dict))

    synonym_tags_raw = defaultdict(lambda: set())

    with open(os_join(HM3D_ANNOTATION_DATA_DIR, REGION_PROPOSAL_MAP_NAME), "r") as src:
        # skip first/header line
        next(src)
        for line in src:
            # if quotes in line, get rid of them, no need to preserve them for this function
            line = line.strip().replace('"', "")
            # list of values in line
            line_vals = line.split(",")
            # idx 1 is count of value across dataset
            tag = line_vals[0].strip()
            votes = int(line_vals[2].strip())
            proposal = line_vals[3].strip()
            derived_proposal_dict[tag]["votes_per_region"] = votes
            if len(proposal) == 0:
                derived_proposal_dict[tag]["ground_truth"] = "unknown room"
            else:
                derived_proposal_dict[tag]["ground_truth"] = proposal
                synonym_tags_raw[proposal].add(tag)
    known = 0
    unknown = 0

    # keyed by proposal, val is string tag to aggregate
    synonym_tags = {}
    for proposal, tags_set in sorted(synonym_tags_raw.items()):
        synonym_tags[proposal] = ":".join([f"{item}" for item in sorted(tags_set)])
        print(
            f"\"{proposal}\" = {','.join([f'{item}' for item in sorted(tags_set)])}\n"
        )

    for _, val_dict in sorted(derived_proposal_dict.items()):
        val = val_dict["ground_truth"]
        if val == "unknown room":
            unknown += 1
            # print(f"?{key}?")
        else:
            known += 1
            val_dict["synonym_tag"] = synonym_tags[val]
            # roomname_is_valid = val in POSSIBLE_REGION_NAMES
            # print(
            #     f"`{key}` maps to `{val}` and val is a valid region name: {roomname_is_valid}"
            # )
    # add entries for synonym tags
    for proposal, synonym_tag_str in sorted(synonym_tags.items()):
        syn_tag_key = f"Aggregate for {proposal}:{synonym_tag_str}"
        derived_proposal_dict[syn_tag_key]["ground_truth"] = proposal
        derived_proposal_dict[syn_tag_key]["synonym_tag"] = synonym_tag_str

    print(
        f"Currently have {known} known region named tags and {unknown} unknown region"
        " named tags."
    )

    return derived_proposal_dict


# Specify the format of the per-category region information dictionary, used in lambda defaultdict ctor and
# elsewhere to build region proposal voting structure
def build_cat_annotation_record_dict():
    # This dict will record co-region membership votes - 1 for each cat name this object shares a region with
    tmp_dict = {}
    # build sub-dict for votes and one for weighted votes
    tmp_dict["votes"] = {k: 0 for k in POSSIBLE_REGION_NAMES}
    tmp_dict["weighted_votes"] = {k: 0 for k in POSSIBLE_REGION_NAMES}

    # number of votes the owning tag gets per region cat/instance for weighting
    tmp_dict["votes_per_region"] = 0
    # build sub-dict for co-region members and the # of times they share a region with this category
    tmp_dict["neighbors_count"] = defaultdict(int)
    # temp holding for per-tag neighbors, keyed by tag, value is set holding obj names.
    tmp_dict["neighbors_dict"] = defaultdict(lambda: set())
    # add entry to keep total object instance count for the owning object cat name
    tmp_dict["object_count"] = 0
    tmp_dict["objects"] = set()

    # entry for # of regions this category is present in across entire dataset
    tmp_dict["region_count"] = 0
    tmp_dict["regions"] = set()
    # entry for # of scenes this category is present in
    tmp_dict["scene_count"] = 0
    tmp_dict["scenes"] = set()

    # entry to hold per scene dict of per region dicts of obj counts
    # per scene dict holding per region object counts
    per_region_dict = defaultdict(lambda: {"obj_count": 0, "objs_set": set()})
    pre_scene_dict = defaultdict(lambda: deepcopy(per_region_dict))
    tmp_dict["regions_present"] = deepcopy(pre_scene_dict)

    # add entry for ground truth assignment from file
    tmp_dict["ground_truth"] = "unknown room"
    # string holding synonym key
    tmp_dict["synonym_tag"] = ""
    return tmp_dict


def build_scene_annotation_record_dict():
    # build per scene per region dictionary of data
    tmp_vote_dict = defaultdict(lambda: {k: 0 for k in POSSIBLE_REGION_NAMES})
    # per scene per region dictionary
    tmp_dict = defaultdict(
        lambda: defaultdict(
            lambda: {
                "vote_dicts": deepcopy(tmp_vote_dict),
                # per tag tag_info
                "tag_info": defaultdict(lambda: {"neighbor_instances_count": 0}),
                "neighbor_count": 0,
                "neighborhood": set(),
            }
        )
    )
    return tmp_dict


def propose_regions(file_names_and_paths: dict):
    print("Derive region proposals based on tags of objects that share region.")
    # build dictionary holding per-scene regions and lists of object names in that region
    per_scene_regions = build_per_scene_SSD_data(file_names_and_paths, "region_data")
    # display results to console
    # debug_show_region_tags(per_scene_regions)

    # Save per-scene, per-region category tag membership
    save_scene_region_tag_maps(per_scene_regions)

    # function to aggregate votes per region by cat
    def countCats(obj_list: list):
        return 1

    # function to aggregate votes per region by obj
    def countObjs(obj_list: list):
        return len(obj_list)

    vote_count_dict = {}
    # if we want to give every object a vote, use countObjs, otherwise use countCats
    if COUNT_OBJS_FOR_VOTES:
        vote_count_dict["count_func"] = countObjs
        vote_count_dict["vote_type_str"] = "Per_Obj"
    else:
        vote_count_dict["count_func"] = countCats
        vote_count_dict["vote_type_str"] = "Per_Cat"

    # build a per-category listing of co-region categories's specified "ground truth" region names
    # for proposals for un-labeled categories and save to file
    # Also holds info per tag of countt of membership
    proposal_dict, per_scene_per_region_dict = build_tag_info_dict(
        per_scene_regions, vote_count_dict
    )
    # save category-region neighbor region proposal votes and neighbor counts mappings
    save_cat_region_maps_and_neighbors(
        proposal_dict, per_scene_per_region_dict, vote_count_dict
    )


# # Potential region names
# POSSIBLE_REGION_NAMES = {
#     "bathroom",
#     "bedroom",
#     "dining room",
#     "garage",
#     "hall/stairwell",
#     "kitchen",
#     "laundry room",
#     "living room",
#     "office",
#     "rec room",
#     "unknown room",
# }


#
# build a per-category listing of co-region categories's or objects specified "ground truth" region names
# for proposals for un-labeled categories from file
def build_tag_info_dict(per_scene_regions: dict, vote_count_dict: dict):
    # what function will be used to count votes, either by cat or by obj
    count_func = vote_count_dict["count_func"]
    # build dict holding known proposals from file, along with structure to record votes and instances
    # key is category name, value is dictionary holding "ground_truth" region name (from file) if known, and
    # votes of the region names of other categories present in the same region
    proposal_dict = load_region_proposals()

    # dictionary to hold per scene, per region proposal votes and other info

    per_scene_region_votes_dict = build_scene_annotation_record_dict()

    # Inner function to aggregaate per scene per region stats for a particular object
    def add_to_proposaldict(
        proposal_dict: dict,
        vote_dict: dict,
        neighbor_dict: dict,
        tag: str,
        region: str,
        scenename: str,
        obj_list: list,
    ):
        scene_region_key = f"{scenename}_{region}"
        proposal_dict[tag]["regions"].add(scene_region_key)
        proposal_dict[tag]["scenes"].add(scenename)

        for obj in obj_list:
            proposal_dict[tag]["objects"].add(obj)

            proposal_dict[tag]["regions_present"][scenename][region]["objs_set"].add(
                obj
            )

        for region_proposal in POSSIBLE_REGION_NAMES:
            proposal_dict[tag]["votes"][region_proposal] += vote_dict["vote_dict"][
                region_proposal
            ]
            proposal_dict[tag]["weighted_votes"][region_proposal] += vote_dict[
                "weighted_vote_dict"
            ][region_proposal]

        for neighbor_tag, _ in sorted(neighbor_dict.items()):
            # don't count tag itself
            if neighbor_tag == tag:
                continue
            # Add unique object name to neighbors dict's set - add set so that objects are not added multiple times
            # in the case of the aggregate tags
            proposal_dict[tag]["neighbors_dict"][neighbor_tag].add(
                f"{scene_region_key}_{neighbor_tag}"
            )

    # go through all per_scene_regions and specify the votes for region label from a particular
    # object in that region based on region mates who have known region name assignments
    for scenename, region_dict in sorted(per_scene_regions.items()):
        # per region in scene
        # for region, tag_dict in sorted(region_dict.items()):
        for region, tag_dict in sorted(region_dict.items()):
            # keys in tag_dict are object cat names (tag)
            # values in tag_dict are lists of object instance names (<tag>_<line #>)

            scene_region_key = f"{scenename}_{region}"
            # use this dict to specify the vote and neighbor counts for each object in the current region
            region_neighbor_dict = build_cat_annotation_record_dict()
            # all the categories in a region will have the same votes (per cat, not per object)-
            # PUMP votes for bed-> bedroom
            for tag, obj_list in tag_dict.items():
                num_objs = len(obj_list)
                # Get known region tags for each region member
                tag_region = proposal_dict[tag]["ground_truth"]
                region_neighbor_dict["votes"][tag_region] += count_func(obj_list)
                region_neighbor_dict["weighted_votes"][tag_region] += (
                    count_func(obj_list) * proposal_dict[tag]["votes_per_region"]
                )

                # add one for each neighborhood tag
                region_neighbor_dict["neighbors_count"][tag] += 1
                per_scene_region_votes_dict[scenename][region]["tag_info"][tag][
                    "neighbor_instances_count"
                ] += num_objs
                per_scene_region_votes_dict[scenename][region][
                    "neighbor_count"
                ] += num_objs
                per_scene_region_votes_dict[scenename][region]["neighborhood"].update(
                    obj_list
                )

            vote_dicts = {
                "vote_dict": region_neighbor_dict["votes"],
                "weighted_vote_dict": region_neighbor_dict["weighted_votes"],
            }
            per_scene_region_votes_dict[scenename][region]["vote_dicts"] = dict(
                vote_dicts
            )

            neighbor_dict = region_neighbor_dict["neighbors_count"]
            # print(
            #     f"\t{region} : {build_string_of_region_votes(vote_dict, True)} :|: {build_string_of_cat_vals(tag_dict.keys())}"
            # )
            # Add this region's votes to each tag found in region
            for tag, obj_list in sorted(tag_dict.items()):
                # Build list of unique object names qualified by scene and region
                tagged_obj_list = [f"{scene_region_key}_{obj}" for obj in obj_list]
                add_to_proposaldict(
                    proposal_dict,
                    vote_dicts,
                    neighbor_dict,
                    tag,
                    region,
                    scenename,
                    tagged_obj_list,
                )
                # If this object has a synonym tag then populate its proposal information
                if len(proposal_dict[tag]["synonym_tag"].strip()) != 0:
                    # add values to proposed aggregation tag entry in proposal_dict as well
                    add_to_proposaldict(
                        proposal_dict,
                        vote_dicts,
                        neighbor_dict,
                        "Aggregate for"
                        f" {proposal_dict[tag]['ground_truth']}:{proposal_dict[tag]['synonym_tag']}",
                        region,
                        scenename,
                        tagged_obj_list,
                    )
            # Aggregate counts per scene per region of objects
            for _, info_dict in sorted(proposal_dict.items()):
                num_objs = len(
                    info_dict["regions_present"][scenename][region]["objs_set"]
                )

                if num_objs != 0:
                    info_dict["regions_present"][scenename][region]["obj_count"] = len(
                        info_dict["regions_present"][scenename][region]["objs_set"]
                    )
                    info_dict["regions_present"][scenename][region]["objs_set"] = set()
                else:
                    del info_dict["regions_present"][scenename][region]

            # Aggregate counts per scene per region of neighbors

    # Get total counts for all scenes, regions and objects for all tags
    for _, info_dict in proposal_dict.items():
        info_dict["scene_count"] = len(info_dict["scenes"])
        info_dict["region_count"] = len(info_dict["regions"])
        info_dict["object_count"] = len(info_dict["objects"])
        for neighbor_tag, neighbors_list in info_dict["neighbors_dict"].items():
            info_dict["neighbors_count"][neighbor_tag] = len(neighbors_list)

    return proposal_dict, per_scene_region_votes_dict


# Determine a room proposal string based on votes and write record to file
def proposal_from_votes(
    dest: Any, scene_and_region: str, vote_dict: dict, ttl_scene_rooms: dict
):
    # build per-scene, per-region vote string
    per_region_vote_str = ",".join(
        [f"{vote_dict[k]}" for k in sorted(POSSIBLE_REGION_NAMES)]
    )

    # don't count unknown tag votes
    del vote_dict["unknown room"]
    # find max votes, and record if duplicates
    max_votes = -1
    tmp_proposal_list = []
    for proposal, votes in vote_dict.items():
        if votes > max_votes:
            max_votes = votes
            tmp_proposal_list = [proposal]
        elif votes == max_votes:
            tmp_proposal_list.append(proposal)

    room_ttl_key = "unknown room"
    room_proposal = "Unknown room"

    if max_votes > 0:
        if len(tmp_proposal_list) > 1:
            room_proposal = (
                f"Tie: {' & '.join([name.capitalize() for name in tmp_proposal_list])}"
            )
        else:
            room_ttl_key = tmp_proposal_list[0]
            room_proposal = room_ttl_key.capitalize()

    # increment scene-wide aggregator
    ttl_scene_rooms[room_ttl_key] += 1

    # determine room proposal and aggregate counts of rooms across scene
    # room_proposal = proposal_from_votes(vote_dict_to_use, ttl_scene_rooms)
    # write to file
    dest.write(f"{scene_and_region},{per_region_vote_str}, {room_proposal}\n")

    return room_proposal


# Write per-category report files
def save_per_cat_data(
    proposal_dict: dict,
    vote_type_str: str,
    votes_header_str: str,
    print_debug: Optional[bool] = True,
):
    per_tag_region_props_filename = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, f"Per_Category_Region_{vote_type_str}_Votes.csv"
    )
    per_tag_region_neighborhoods = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Category_Region_Neighbors.csv"
    )
    per_tag_scene_region_info = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Scene_Region_Cat_Prescence.csv"
    )
    per_category_counts_ignore = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Category_Counts_Uncommon.csv"
    )

    print(
        "Save\n\tper-tag region proposals as"
        f" {per_tag_region_props_filename}\n\tper-tag region neighbors as"
        f" {per_tag_region_neighborhoods}\n\tper-tag scene and region info as"
        f" {per_tag_scene_region_info}"
    )
    if print_debug:
        print("All per-tag votes :")
    # Save per category pre region name votes as dest1 and save per category neighbor listings as dest2 and per-tag scene and region info as dest3
    with (
        open(
            os_join(per_tag_region_props_filename),
            "w",
        ) as dest1,
        open(
            per_tag_region_neighborhoods,
            "w",
        ) as dest2,
        open(
            per_tag_scene_region_info,
            "w",
        ) as dest3,
        open(per_category_counts_ignore, "w") as dest4,
    ):
        header_tag = (
            "Category Tag,# of Scenes,# of Regions,Regions per Scene,# of"
            " Instances,Mean Instances per Scene,Mean Instances per Region"
        )
        dest1.write(f"{header_tag},GT Name (if specified),{votes_header_str}\n")
        dest2.write(f"{header_tag},# of Neighbors,Neighbors and per-region counts\n")
        dest3.write(f"Scene ID,Region ID,{header_tag},Instance count in region\n")
        dest4.write("Category,# Instances\n")

        for tag, region_data_dict in sorted(proposal_dict.items()):
            num_regions = region_data_dict["region_count"]
            num_instances = region_data_dict["object_count"]
            num_scenes = region_data_dict["scene_count"]
            regions_per_scene = num_regions / num_scenes
            instances_per_scene = num_instances / num_scenes
            instances_per_region = num_instances / num_regions
            neighbor_dict = region_data_dict["neighbors_count"]
            # dict per scene per region of counts of objects and tags present
            regions_present = region_data_dict["regions_present"]

            neighbor_str = build_str_of_shared_cats(neighbor_dict, False, True)
            scene_regions_str = build_str_of_scene_regions(regions_present, False)
            vote_dict = region_data_dict["votes"]
            gt_str = ""
            gt_file_str = ""
            if region_data_dict["ground_truth"] != "unknown room":
                gt_file_str = region_data_dict["ground_truth"]
                gt_str = f"Specified Ground Truth name : {gt_file_str}\n\t"
            if print_debug:
                print(
                    f"\n{tag} :\nIn {num_scenes} scenes, {num_regions} regions,"
                    f" {num_instances} objs total"
                )
                print(
                    f"\t{gt_str}Per Region Votes {vote_type_str} :"
                    f" {build_string_of_region_votes(vote_dict, True)}"
                )
                print(f"\tScene-Regions present :{scene_regions_str}")
                print(f"\tNeighbors :{neighbor_str}")
                syn_tag = (
                    region_data_dict["synonym_tag"]
                    if len(region_data_dict["synonym_tag"].strip()) > 0
                    else "none"
                )
                print(f"Synonym tags : {syn_tag}\n")
            per_cat_vote_str = ",".join(
                [f"{vote_dict[k]}" for k in sorted(POSSIBLE_REGION_NAMES)]
            )

            # data shared between both files
            shared_data_str = f"{tag},{num_scenes},{num_regions},{regions_per_scene},{num_instances},{instances_per_scene},{instances_per_region}"
            # write to all files
            # write per tag region votes
            dest1.write(f"{shared_data_str},{gt_file_str},{per_cat_vote_str}\n")
            # write per tag neighborhoods
            dest2.write(f"{shared_data_str},{len(neighbor_dict)},{neighbor_str}\n")
            # write per scene per region counts for object instance presence
            if tag not in IGNORE_TAGS:
                if "Aggregate" not in tag:
                    dest4.write(f"{tag},{num_instances}\n")
                for scene, region_dict in sorted(regions_present.items()):
                    for region, counts in sorted(region_dict.items()):
                        dest3.write(
                            f"{scene},{region},{shared_data_str},{counts['obj_count']}\n"
                        )


# calculate the moments of the passed list and return the results in a dictionary
def calc_stat_moments(data: list):
    # convenience for kahan
    def calcSumAndC(sumAndC: list, y: float):
        t = sumAndC[0] + y
        sumAndC[1] = (t - sumAndC[0]) - y
        sumAndC[0] = t

    numVals = len(data)
    # list of moments : mean, var, std, skew, kurt, exkurt
    mmnts = []
    # calculate mean while minimizing float error
    sumMu = data[0]
    # data_min = data[0]
    # data_max = data[0]
    cMu = 0.0
    y = 0.0
    t = 0.0
    for i in range(1, numVals):
        # data_min = min(data[i], data_min)
        # data_max = max(data[i], data_max)
        y = data[i] - cMu
        t = sumMu + y
        cMu = (t - sumMu) - y
        sumMu = t

    mean = sumMu / numVals

    tDiff = 0
    tDiffSq = 0
    # initialize for Kahan summation method
    valMMean = data[0] - mean
    sumAndCSq = [valMMean * valMMean, 0.0]
    sumAndCCu = [(sumAndCSq[0]) * valMMean, 0.0]
    sumAndCQu = [(sumAndCSq[0]) * (sumAndCSq[0]), 0.0]
    # kahan summation to address magnitude issues in adding 2 values of largely different magnitudes
    for i in range(1, numVals):
        tDiff = data[i] - mean
        tDiffSq = tDiff * tDiff
        calcSumAndC(sumAndCSq, tDiffSq - sumAndCSq[1])
        calcSumAndC(sumAndCCu, (tDiffSq * tDiff) - sumAndCCu[1])
        calcSumAndC(sumAndCQu, (tDiffSq * tDiffSq) - sumAndCQu[1])

    var = sumAndCSq[0] / numVals
    std = math.sqrt(var)
    skew = (sumAndCCu[0] / numVals) / (std * var)
    kurt = (sumAndCQu[0] / numVals) / (var * var)
    excKurt = kurt - 3.0

    mmnts = {
        "mean": mean,
        "var": var,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "excKurt": excKurt,
    }
    # we want to maintain insertion order
    res_str = ",".join(f"{val}" for _, val in mmnts.items())
    mmnts["mmnt_str"] = res_str
    return mmnts


# Save per Scene/per Region data to files
def save_per_scene_region_data(
    per_scene_per_region_dict: dict, vote_type_str: str, votes_header_str: str
):
    # per scene per region votes of region/room label based on category tag proposals
    per_scene_per_region_votes = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Scene_Region_Votes.csv"
    )
    per_scene_per_region_weighted_votes = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Scene_Region_Weighted_Votes.csv"
    )
    # per scene region/room label assignments, based on category tag proposals and per-region vote
    per_scene_total_votes = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Scene_Total_Votes.csv"
    )
    per_scene_total_weighted_votes = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Scene_Total_Weighted_Votes.csv"
    )
    with (
        open(
            per_scene_per_region_votes,
            "w",
        ) as dest4,
        open(
            per_scene_per_region_weighted_votes,
            "w",
        ) as dest5,
        open(
            per_scene_total_votes,
            "w",
        ) as dest6,
        open(
            per_scene_total_weighted_votes,
            "w",
        ) as dest7,
    ):
        room_counts_hdr_str = (
            f"# {'s,# '.join([name.capitalize() for name in sorted(POSSIBLE_REGION_NAMES)])}s"
        )
        dest4.write(f"Scene Name,Region #,{votes_header_str},Room Proposal\n")
        dest5.write(f"Scene Name,Region #,{votes_header_str},Weighted Room Proposal\n")
        dest6.write(
            "Scene"
            f" Name,{room_counts_hdr_str},{vote_type_str} Votes:,{votes_header_str}\n"
        )
        dest7.write(
            "Scene"
            f" Name,{room_counts_hdr_str},{vote_type_str} Votes:,{votes_header_str}\n"
        )

        for scenename, region_dict in sorted(per_scene_per_region_dict.items()):
            ttl_scene_votes = {k: 0 for k in POSSIBLE_REGION_NAMES}
            # annotated region proposal layout
            ttl_scene_rooms = {k: 0 for k in POSSIBLE_REGION_NAMES}
            ttl_scene_rooms_weighted = {k: 0 for k in POSSIBLE_REGION_NAMES}
            for region, per_region_dict in sorted(region_dict.items()):
                # scene and region str
                scene_and_region = f"{scenename},{region}"
                # get per scene per region info
                vote_dicts = per_region_dict["vote_dicts"]
                # don't weight aggregated totals for scene level
                for k in sorted(POSSIBLE_REGION_NAMES):
                    ttl_scene_votes[k] += vote_dicts["vote_dict"][k]

                # calculate and write proposal for non-weighted votes
                proposal_from_votes(
                    dest4, scene_and_region, vote_dicts["vote_dict"], ttl_scene_rooms
                )

                # calculate and write proposal for weighted votes
                proposal_from_votes(
                    dest5,
                    scene_and_region,
                    vote_dicts["weighted_vote_dict"],
                    ttl_scene_rooms_weighted,
                )

            ttl_vote_region_str = ",".join(
                [f"{ttl_scene_votes[k]}" for k in sorted(POSSIBLE_REGION_NAMES)]
            )
            ttl_rooms_count_str = ",".join(
                [f"{ttl_scene_rooms[k]}" for k in sorted(POSSIBLE_REGION_NAMES)]
            )
            ttl_rooms_count_weighted_str = ",".join(
                [
                    f"{ttl_scene_rooms_weighted[k]}"
                    for k in sorted(POSSIBLE_REGION_NAMES)
                ]
            )
            dest6.write(f"{scenename},{ttl_rooms_count_str},,{ttl_vote_region_str}\n")
            dest7.write(
                f"{scenename},{ttl_rooms_count_weighted_str},,{ttl_vote_region_str}\n"
            )

    # per scene per region neighbors
    # ttl_num_scenes = len(per_scene_per_region_dict)
    per_scene_per_region_neighborhoods = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Scene_Region_Neighborhoods.csv"
    )
    # per scene region, category and object counts and averages
    per_scene_neighborhoods = os_join(
        HM3D_ANNOTATION_RPT_DEST_DIR, "Per_Scene_Neighborhood_Stats.csv"
    )
    with (
        open(
            per_scene_per_region_neighborhoods,
            "w",
        ) as dest8,
        open(
            per_scene_neighborhoods,
            "w",
        ) as dest9,
    ):
        dest8.write(
            "Scene Name,Region #,# Unique Categories,# Objects,Object Instances in"
            " Region,Category\n"
        )

        dest9.write(
            "Scene Name,# Regions,# Unique Categories,Per Region Cat Mean,Per Region"
            " Cat Var,Per Region Cat Std,Per Region Cat Skew,Per Region Cat Kurt,Per"
            " Region Cat Ex Kurt,# Objects,Per Region Object Mean,Per Region Object"
            " Var,Per Region Object Std,Per Region Object Skew,Per Region Object"
            " Kurt,Per Region Object Ex Kurt\n"
        )
        ttl_regions = 0
        ttl_objs = 0
        ttl_cats = set()
        # aggregate for moments analysis
        per_scene_num_regions = []
        per_scene_num_objs = []
        per_scene_num_cats = []

        # dict of aggregate moments
        per_scene_region_obj_stats = {}
        per_scene_region_cat_stats = {}

        for scenename, region_dict in sorted(per_scene_per_region_dict.items()):
            num_objs_in_scene = 0
            unique_cats_per_scene = set()
            num_regions = len(region_dict)
            per_scene_num_regions.append(num_regions)
            # Per scene region aggregates for moments analysis
            per_region_num_objs = []
            per_region_num_cats = []

            for region, per_region_dict in sorted(region_dict.items()):
                # Object instances
                num_neighbors = per_region_dict["neighbor_count"]
                num_objs_in_scene += num_neighbors
                per_region_num_objs.append(num_neighbors)

                # Category counts
                tag_info_dict = per_region_dict["tag_info"]
                num_categories = len(tag_info_dict)
                per_region_num_cats.append(num_categories)
                # Counting total unique categories per scene, across all regions
                unique_cats_per_scene.update(tag_info_dict.keys())
                # build a string of neighbor objects - includes ignored tags since these are actual objects
                obj_list = sorted(per_region_dict["neighborhood"])

                # obj_list = [
                #     obj
                #     for obj in sorted(per_region_dict["neighborhood"])
                #     if obj.split("_")[0] not in IGNORE_TAGS
                # ]

                neighbor_str = ":".join(obj_list)

                dest8.write(
                    f"{scenename},{region},{num_categories},{num_neighbors},{neighbor_str}\n"
                )
            # find the moments of this scene's per-region object and category data
            # per scene dist of object counts across all regions
            per_scene_region_obj_stats[scenename] = calc_stat_moments(
                per_region_num_objs
            )
            # per scene dist of unique category counts across all regions
            per_scene_region_cat_stats[scenename] = calc_stat_moments(
                per_region_num_cats
            )

            num_unique_cats_per_scene = len(unique_cats_per_scene)
            per_scene_num_cats.append(num_unique_cats_per_scene)

            ttl_cats.update(unique_cats_per_scene)
            per_scene_num_objs.append(num_objs_in_scene)
            dest9.write(
                f"{scenename},{num_regions},{num_unique_cats_per_scene},{per_scene_region_cat_stats[scenename]['mmnt_str']},{num_objs_in_scene},{per_scene_region_obj_stats[scenename]['mmnt_str']}\n"
            )
            ttl_objs += num_objs_in_scene
            ttl_regions += num_regions

        # find moments of num regions, num unique-per-region cats and num objects across all scenes
        # distribution of region counts across all scenes
        scene_wide_stats = {}
        scene_wide_stats["region"] = calc_stat_moments(per_scene_num_regions)
        # distribution of unique-per-region category counts across all scenes
        scene_wide_stats["cat"] = calc_stat_moments(per_scene_num_cats)
        # distribution of objects across all scenes
        scene_wide_stats["obj"] = calc_stat_moments(per_scene_num_objs)

        ttl_num_cats = len(ttl_cats)

        dest9.write("\n,# Total Regions,# Total Categories,# Total Objects\n")
        dest9.write(f"All Scenes,{ttl_regions},{ttl_num_cats},{ttl_objs}\n")
        for k, stat_dict in scene_wide_stats.items():
            target = k.capitalize()
            dest9.write(
                f"\n{target}s,Per Scene {target} Mean,Per Scene {target} Var,Per Scene"
                f" {target} Std,Per Scene {target} Skew,Per Scene {target} Kurt,Per"
                f" Scene {target} Ex Kurt\n"
            )
            dest9.write(f"All Scenes,{stat_dict['mmnt_str']}\n")


# save per category region name proposal votes
def save_cat_region_maps_and_neighbors(
    proposal_dict: dict,
    per_scene_per_region_dict: dict,
    vote_count_dict: dict,
    print_debug: Optional[bool] = True,
):
    vote_type_str = vote_count_dict["vote_type_str"]

    votes_header_str = ",".join(
        [name.capitalize() for name in sorted(POSSIBLE_REGION_NAMES)]
    )

    # Save per category data to files
    save_per_cat_data(proposal_dict, vote_type_str, votes_header_str, print_debug)

    # Save per Scene/per Region data to files
    save_per_scene_region_data(
        per_scene_per_region_dict, vote_type_str, votes_header_str
    )


# Save per-scene, per-region category tag membership. This will not write any ignored tags,
# unless a particular region only contains ignored tags, which will be specified in output.
# Tags may be ignored because they are ubiquitous across annotations (i.e. wall, ceiling) and
# add little in the way of salient region specificity
def save_scene_region_tag_maps(per_scene_regions: dict):
    # Build a string to write to file all tags being ignored
    IGNORE_TAGS_str = ";".join(IGNORE_TAGS)
    # save per-region lists of tags to csv
    with open(
        os_join(HM3D_ANNOTATION_RPT_DEST_DIR, "Region_Tag_Mappings.csv"),
        "w",
    ) as dest:
        dest.write(
            "Scene,Region,# of Tags,Relevant Tags in Region (Excluding :"
            f" {IGNORE_TAGS_str})\n"
        )
        for scenename, region_dict in per_scene_regions.items():
            for region, tag_dict in sorted(region_dict.items()):
                # keys in tag_dict are object cat names (tag)
                # values in tag_dict are lists of object instance names (<tag>_<line #>)
                tag_string = build_string_of_cat_vals(tag_dict.keys())

                dest.write(
                    f"{scenename},{region},{len(tag_dict.keys())},{tag_string}\n"
                )


# Build a string of scene-region hashes and object counts present. Scene-region hash is
# <scene name>_<region #> and is key of dict, counts is value
def build_str_of_scene_regions(regions_present: dict, sep_per_line: bool):
    join_char = "|"
    if sep_per_line:
        join_char = "\n"

    return f"{join_char}".join(
        [
            f"{scene}_{region}:{counts['obj_count']}"
            for scene, region_dict in sorted(regions_present.items())
            for region, counts in sorted(region_dict.items())
        ]
    )


# Build string of passed category tags and counts. These are the neighbors of a
# particular tag (tags that share the same region) and the counts of the regions this category shares
def build_str_of_shared_cats(
    neighbor_dict: dict, sep_per_line: bool, ignore_common: bool
):
    join_char = "|"
    start_char = ""
    if sep_per_line:
        join_char = "\n"
        start_char = "\t"
    # return string sorted by count
    if ignore_common:
        return f"{join_char}".join(
            [
                f"{start_char}{k}:{v}"
                for k, v in sorted(
                    neighbor_dict.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if k not in IGNORE_TAGS
            ]
        )
    else:
        return f"{join_char}".join(
            [
                f"{start_char}{k}:{v}"
                for k, v in sorted(
                    neighbor_dict.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
        )


# Build string of category tags present in specified tags list, ignoring
# tags in IGNORE_TAGS unless those are the only ones present in the
# tags list
def build_string_of_cat_vals(tags: list):
    obj_list = [k for k in tags if k not in IGNORE_TAGS]
    if len(obj_list) > 0:
        tag_string = ";".join(obj_list)
    else:
        # Don't want an empty string so include IGNORE_TAGS
        tag_string = f"Only contains ignored tags : { ';'.join(tags)}"
    return tag_string


# Build a comma-sep string of vote results for each possible region name from passed votes dictionary
# possibly ignoring 0-vote entries
def build_string_of_region_votes(
    votes_dict: dict, ignore_0_votes: Optional[bool] = False
):
    if ignore_0_votes:
        val_str = ",".join(
            [
                f"{k}:{votes_dict[k]}"
                for k in sorted(POSSIBLE_REGION_NAMES)
                if votes_dict[k] != 0
            ]
        )
    else:
        val_str = ",".join(
            [f"{k}:{votes_dict[k]}" for k in sorted(POSSIBLE_REGION_NAMES)]
        )
    return val_str


def debug_show_region_tags(per_scene_regions: dict):
    num_regions = 0
    for scenename, region_dict in per_scene_regions.items():
        print(f"Scene :{scenename} has {len(region_dict)} regions")
        num_regions += len(region_dict)
        for region, tag_dict in sorted(region_dict.items()):
            print(
                f"\t#{region} has {len(tag_dict)} unique object tags :"
                f" {';'.join(tag_dict.keys())}"
            )
            # for tag, nameslist in sorted(tag_dict.items()):
            #     if tag != "unknown":
            #         print(f"\t\t{tag} : size : {len(nameslist)} : {nameslist}")
            #     else:
            #         print(f"\t\t{tag} : size : {len(nameslist)}")

    print(f"Total regions in all {len(per_scene_regions)} scenes is {num_regions}")


###########################################################
# Used by all functions


# Build dictionary of src and dest file names and paths
# Key is source dir of scene, value is list of dictionaries specifying various
# source and destination directories, subdirectories and filenames for each partition
# the file may belong to.  This usually will be a list of a single dictionary, but will have
# two entires for val/minival scenes.
def build_file_listing():
    print(f"Building source file listing from dir {HM3D_ANNOTATION_SRC_DIR}")

    # go through annotation directory, find paths to all files of
    # interest (annotated glbs and semantic lexicon/text files)
    # and match to desired destination in HM3D install directory
    def buildListingFromRegex(subdir_RE):
        # directory name pattern to find annotation files
        annotation_dir_pattern = re.compile(subdir_RE)
        # get directory listing
        return ut.get_directories_matching_regex(
            HM3D_ANNOTATION_SRC_DIR, annotation_dir_pattern
        )

    # directory listing
    dir_listing = buildListingFromRegex(HM3D_SRC_ANNOTATION_SUBDIR_RE)
    if len(dir_listing) == 0:
        print("No files found using first subdir regex pattern, attempting alternate.")
        # no files with first dir regex, try alt
        dir_listing = buildListingFromRegex(HM3D_SRC_ANNOTATION_SUBDIR_RE_2)

    print(f"Size of dir_listing  : {len(dir_listing)}")
    # destination directory will be based on numeric field, if available
    file_names_and_paths = defaultdict(lambda: [])
    for src_dir, dirname_full in dir_listing:
        src_dirname_full = os_join(src_dir, dirname_full)
        dirname = dirname_full.split(".semantic")[0]
        scene_hash = dirname.split("-")[-1]
        # get directory and file names for both semantic src files
        src_file_list = ut.get_files_matching_regex(
            src_dirname_full, re.compile(scene_hash + ".semantic.")
        )
        if len(src_file_list) != 2:
            print(
                f"Problem with source dir files {dirname_full} : Unable to find 2"
                f" source files ({len(src_file_list)} files instead) so skipping this"
                " source dir."
            )
            continue
        # find appropriate destination directory for given source scene
        dest_dir_list = ut.get_directories_matching_regex(
            HM3D_DEST_DIR, re.compile(".*" + scene_hash + "$")
        )
        # Must find at least 1 dest. Might have 2 if val and minival
        if len(dest_dir_list) > 0:
            for dest_dir in dest_dir_list:
                scene_dest_dir = os_join(dest_dir[0], dest_dir[1])
                partition_subdir = dest_dir[0].split(os_sep)[-1]
                partition_tag = partition_subdir
                scene_dest_subdir = os_join(partition_subdir, dest_dir[1])
                src_files = defaultdict(lambda: {})
                for src_full_dir, _, src_filename in src_file_list:
                    if ".txt" in src_filename:
                        key = "SSD"
                    else:
                        key = "GLB"

                    # fully qualified source path and filename
                    src_files[key]["src_full_path"] = os_join(
                        src_full_dir, src_filename
                    )
                    # TODO perhaps we wish to rename the file in the destination? If so, do so here, instead of using src_filename
                    dest_filename = src_filename
                    # subdir and filename
                    src_files[key]["dest_subdir_file"] = os_join(
                        scene_dest_subdir, dest_filename
                    )
                    # fully qualified destination path/filename
                    src_files[key]["dest_full_path"] = os_join(
                        scene_dest_dir, dest_filename
                    )

                tmp_dict = {
                    "dest_dir": scene_dest_dir,
                    "dest_part_tag": partition_tag,
                    "scene_name": dest_dir[1],
                }
                for k, v in src_files.items():
                    tmp_dict[k] = v
                file_names_and_paths[src_dirname_full].append(tmp_dict)

        else:
            print(
                f"Problem with source dir {dirname_full} : Unable to find destination"
                f" dir due to {len(dest_dir_list)} matching destinations in current"
                " HM3D dataset so skipping this source dir.",
                end="",
            )
            for bad_dest in dest_dir_list:
                print(f"\t{bad_dest}", end="")

            continue

    return file_names_and_paths


###########################################################
# Debug


# This will display the contents of the file_names_and_paths dict built by build_file_listing()
def debug_file_list_dict(file_names_and_paths: dict):
    for src_dir, data_dict_list in sorted(file_names_and_paths.items()):
        print(f"src_dir : {src_dir} w/list of size {len(data_dict_list)}: ")
        for data_dict in data_dict_list:
            print("\tdata_dict :")
            for k, v in sorted(data_dict.items()):
                if isinstance(v, dict):
                    print(f"\t\tKey: {k} :")
                    for k1, v1 in sorted(v.items()):
                        print(f"\t\t\tSubkey : {k1} : Value : {v1}")
                else:
                    print(f"\t\tKey: {k} : Value: {v}")
        print("\n")


# This will compare two lists to make sure they are equal
def debug_compare_two_lists(list1: list, list2: list):
    import functools

    if functools.reduce(
        lambda x, y: x and y,
        map(lambda p, q: p == q, sorted(list1), sorted(list2)),
        True,
    ):
        print("The lists are the same")
    else:
        print(
            f"The lists are not the same : list1 : {len(list1)} : list2 : {len(list2)}"
        )


def main():
    file_names_and_paths = build_file_listing()
    # debug_file_list_dict(file_names_and_paths)
    # return

    if FUNC_TO_DO == Functions_Available.PROCESS_AND_COPY_SRC:
        # process src files and copy (along with potentially editing/correcting)
        # semantic txt files and (possibly) semantic.glb files to dest dir
        process_and_copy_files(file_names_and_paths)
    elif FUNC_TO_DO == Functions_Available.COUNT_SEMANTIC_COLORS:
        # go through every semantic annotation file and perform a count of each color present.
        color_annotation_reports(file_names_and_paths)
    elif FUNC_TO_DO == Functions_Available.CALC_REGIONS:
        # find reasonable region name proposals based on object tags within a region for
        # each region in scene, and write output to user_defined tag in scene dataset files
        # for each scene.
        propose_regions(file_names_and_paths)
    else:
        print(f"Unknown/unsupported function requested : {FUNC_TO_DO}. Aborting!")


if __name__ == "__main__":
    main()
