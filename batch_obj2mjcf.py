#!/usr/bin/env python3
"""
Batch processing script for obj2mjcf conversion.

This script processes all subdirectories under a given directory that contain OBJ files,
applying obj2mjcf conversion in-place without creating additional directory layers.

Usage:
    python batch_obj2mjcf.py meshes/visual [--options...]

Equivalent to running obj2mjcf individually on each subdirectory but with in-place modification.
"""

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import trimesh
import tyro
from PIL import Image
from termcolor import cprint

# Import from the local obj2mjcf package
from obj2mjcf.cli import Args, CoacdArgs, process_obj, resize_texture, decompose_convex, parse_mtl_name
from obj2mjcf.material import Material
from obj2mjcf.mjcf_builder import MJCFBuilder
from obj2mjcf import constants


class VisualOnlyMJCFBuilder(MJCFBuilder):
    """Extended MJCFBuilder that can optionally skip collision geometries."""
    
    def __init__(
        self,
        filename: Path,
        mesh,
        materials: List[Material],
        work_dir: Path = Path(),
        decomp_success: bool = False,
        visual_only: bool = False,
    ):
        super().__init__(filename, mesh, materials, work_dir, decomp_success)
        self.visual_only = visual_only
    
    def build(self, add_free_joint: bool = False) -> None:
        """Build MJCF with optional collision geometry skipping."""
        from lxml import etree
        
        # Constants.
        filename = self.filename
        mtls = self.materials

        # Start assembling xml tree.
        root = etree.Element("mujoco", model=filename.stem)

        # Add defaults.
        self.add_visual_and_collision_default_classes(root)

        # Add assets.
        asset_elem = self.add_assets(root, mtls)

        # Add worldbody.
        worldbody_elem = etree.SubElement(root, "worldbody")
        obj_body = etree.SubElement(worldbody_elem, "body", name=filename.stem)
        if add_free_joint:
            etree.SubElement(obj_body, "freejoint")

        # Add visual geometries to object body.
        self.add_visual_geometries(obj_body, asset_elem)
        
        # Only add collision geometries if not visual_only
        if not self.visual_only:
            self.add_collision_geometries(obj_body, asset_elem)

        # Create the tree.
        tree = etree.ElementTree(root)
        etree.indent(tree, space=constants.XML_INDENTATION, level=0)
        self.tree = tree


def find_obj_directories(root_dir: Path) -> List[Path]:
    """Find all subdirectories that contain OBJ files."""
    obj_dirs = []
    
    if not root_dir.exists() or not root_dir.is_dir():
        raise ValueError(f"Directory {root_dir} does not exist or is not a directory")
    
    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            # Check if this directory contains any .obj files
            obj_files = list(subdir.glob("*.obj"))
            if obj_files:
                obj_dirs.append(subdir)
                logging.info(f"Found directory with OBJ files: {subdir}")
    
    return obj_dirs


def process_obj_inplace(obj_file: Path, args: Args, visual_only: bool = False) -> None:
    """
    Process a single OBJ file in-place without creating a subdirectory.
    
    This is a modified version of the original process_obj function that works
    directly in the source directory instead of creating a work_dir.
    """
    work_dir = obj_file.parent  # Work directly in the source directory
    logging.info(f"Processing {obj_file} in-place in {work_dir}")

    # Decompose the mesh into convex pieces if desired.
    decomp_success = False
    if args.decompose:
        # For decomposition, we need to create temporary collision files
        decomp_success = decompose_convex(obj_file, work_dir, args.coacd_args)

    # Check if the OBJ files references an MTL file.
    with obj_file.open("r") as f:
        mtl_name = parse_mtl_name(f.readlines())

    process_mtl = mtl_name is not None
    sub_mtls: List[List[str]] = []
    mtls: List[Material] = []
    
    if process_mtl:
        # Make sure the MTL file exists. The MTL filepath is relative to the OBJ's.
        mtl_filename = obj_file.parent / mtl_name
        if not mtl_filename.exists():
            raise RuntimeError(
                f"The MTL file {mtl_filename.resolve()} referenced in the OBJ file "
                f"{obj_file} does not exist"
            )
        logging.info(f"Found MTL file: {mtl_filename}")

        # Parse the MTL file into separate materials.
        with open(mtl_filename, "r") as f:
            lines = f.readlines()
        # Remove comments.
        lines = [
            line for line in lines if not line.startswith(constants.MTL_COMMENT_CHAR)
        ]
        # Remove empty lines.
        lines = [line for line in lines if line.strip()]
        # Remove trailing whitespace.
        lines = [line.strip() for line in lines]
        # Split at each new material definition.
        for line in lines:
            if line.startswith("newmtl"):
                sub_mtls.append([])
            sub_mtls[-1].append(line)
        for sub_mtl in sub_mtls:
            mtls.append(Material.from_string(sub_mtl))

        # Process each material.
        for mtl in mtls:
            logging.info(f"Found material: {mtl.name}")
            if mtl.map_Kd is not None:
                texture_path = Path(mtl.map_Kd)
                texture_name = texture_path.name
                src_filename = obj_file.parent / texture_path
                if not src_filename.exists():
                    raise RuntimeError(
                        f"The texture file {src_filename} referenced in the MTL file "
                        f"{mtl.name} does not exist"
                    )
                
                # Convert JPEG to PNG if needed (MuJoCo only supports PNG textures)
                if texture_path.suffix.lower() in [".jpg", ".jpeg"]:
                    image = Image.open(src_filename)
                    # Remove the original JPEG file
                    os.remove(src_filename)
                    # Create PNG version
                    png_filename = (work_dir / texture_path.stem).with_suffix(".png")
                    image.save(png_filename)
                    texture_name = png_filename.name
                    mtl.map_Kd = texture_name
                    # Update the MTL file to reference the PNG instead
                    # This will be handled when we rewrite the MTL file later
                
                # Resize texture if needed
                final_texture_path = work_dir / texture_name
                resize_texture(final_texture_path, args.texture_resize_percent)
        
        # Rewrite the MTL file with updated texture references
        if any(mtl.map_Kd and Path(mtl.map_Kd).suffix.lower() in [".jpg", ".jpeg"] for mtl in mtls):
            with open(mtl_filename, "w") as f:
                for sub_mtl in sub_mtls:
                    for line in sub_mtl:
                        # Update texture references from JPEG to PNG
                        if line.startswith("map_Kd"):
                            parts = line.split()
                            if len(parts) >= 2:
                                texture_path = Path(parts[1])
                                if texture_path.suffix.lower() in [".jpg", ".jpeg"]:
                                    new_texture = texture_path.with_suffix(".png")
                                    line = f"map_Kd {new_texture.name}"
                        f.write(line + "\n")
        
        logging.info("Done processing MTL file")

    logging.info("Processing OBJ file with trimesh")
    mesh = trimesh.load(
        obj_file,
        split_object=True,
        group_material=True,
        process=False,
        maintain_order=False,
    )

    # Process the mesh - this will create submesh files in the same directory
    if isinstance(mesh, trimesh.base.Trimesh):
        # Single mesh - overwrite the original OBJ file
        logging.info(f"Saving processed mesh to {obj_file}")
        mesh.export(obj_file.as_posix(), include_texture=True, header=None)
    else:
        # Multiple submeshes - create separate files
        logging.info("Grouping and saving submeshes by material")
        # First, remove the original OBJ file since we're replacing it with submeshes
        obj_file.unlink()
        
        for i, geom in enumerate(mesh.geometry.values()):
            savename = work_dir / f"{obj_file.stem}_{i}.obj"
            logging.info(f"Saving submesh {savename}")
            geom.export(savename.as_posix(), include_texture=True, header=None)

    # Handle edge case where material file has many materials but OBJ only references one
    if isinstance(mesh, trimesh.base.Trimesh) and len(mtls) > 1:
        # Find the material that is referenced
        with open(obj_file, "r") as f:
            lines = f.readlines()
        mat_name = None
        for line in lines:
            if line.startswith("usemtl"):
                mat_name = line.split()[1]
                break
        
        if mat_name:
            # Trim out the extra materials
            for smtl in sub_mtls:
                if smtl[0].split()[1] == mat_name:
                    break
            sub_mtls = [smtl]
            mtls = [Material.from_string(smtl)]

    mtls = list({obj.name: obj for obj in mtls}.values())

    # Clean up any temporary material files created during trimesh processing
    for file in [
        x
        for x in work_dir.glob("**/*")
        if x.is_file() and "material_0" in x.name and not x.name.endswith(".png")
    ]:
        file.unlink()

    # Build an MJCF if requested
    if args.save_mjcf or args.compile_model:
        builder = VisualOnlyMJCFBuilder(
            obj_file, mesh, mtls, 
            work_dir=work_dir, 
            decomp_success=decomp_success,
            visual_only=visual_only
        )
        builder.build(add_free_joint=args.add_free_joint)

        # Compile and step the physics to check for any errors
        if args.compile_model:
            builder.compile_model()

        # Save MJCF file
        if args.save_mjcf:
            builder.save_mjcf()


def main():
    """Main entry point for the batch processing script."""
    parser = argparse.ArgumentParser(
        description="Batch process OBJ files in subdirectories using obj2mjcf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "root_dir",
        help="Root directory containing subdirectories with OBJ files"
    )
    parser.add_argument(
        "--obj-filter",
        help="Only convert obj files matching this regex"
    )
    parser.add_argument(
        "--save-mjcf",
        action="store_true",
        help="Save an example XML (MJCF) file"
    )
    parser.add_argument(
        "--compile-model",
        action="store_true", 
        help="Compile the MJCF file to check for errors"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Approximate mesh decomposition using CoACD"
    )
    parser.add_argument(
        "--texture-resize-percent",
        type=float,
        default=1.0,
        help="Resize the texture to this percentage of the original size"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files without asking"
    )
    parser.add_argument(
        "--add-free-joint",
        action="store_true",
        help="Add a free joint to the root body"
    )
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Only create visual geometries, skip collision geometries"
    )
    
    # CoACD arguments
    parser.add_argument(
        "--preprocess-resolution",
        type=int,
        default=50,
        help="Resolution for manifold preprocess (20~100), default = 50"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Concavity threshold for terminating the decomposition (0.01~1), default = 0.05"
    )
    parser.add_argument(
        "--max-convex-hull",
        type=int,
        default=-1,
        help="Max # convex hulls in the result, -1 for no maximum limitation"
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=100,
        help="Number of search iterations in MCTS (60~2000), default = 100"
    )
    parser.add_argument(
        "--mcts-max-depth",
        type=int,
        default=3,
        help="Max search depth in MCTS (2~7), default = 3"
    )
    parser.add_argument(
        "--mcts-nodes",
        type=int,
        default=20,
        help="Max number of child nodes in MCTS (10~40), default = 20"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=2000,
        help="Sampling resolution for Hausdorff distance calculation (1e3~1e4), default = 2000"
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Enable PCA pre-processing, default = false"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling, default = 0"
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Convert to Path object
    root_dir = Path(args.root_dir)

    # Find all directories containing OBJ files
    try:
        obj_dirs = find_obj_directories(root_dir)
    except ValueError as e:
        cprint(f"Error: {e}", "red")
        sys.exit(1)

    if not obj_dirs:
        cprint(f"No subdirectories with OBJ files found in {root_dir}", "yellow")
        sys.exit(0)

    cprint(f"Found {len(obj_dirs)} directories with OBJ files", "green")

    # Create Args object for obj2mjcf processing
    coacd_args = CoacdArgs(
        preprocess_resolution=args.preprocess_resolution,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        mcts_nodes=args.mcts_nodes,
        resolution=args.resolution,
        pca=args.pca,
        seed=args.seed,
    )

    obj2mjcf_args = Args(
        obj_dir="",  # Will be set per directory
        obj_filter=args.obj_filter,
        save_mjcf=args.save_mjcf,
        compile_model=args.compile_model,
        verbose=args.verbose,
        decompose=args.decompose,
        coacd_args=coacd_args,
        texture_resize_percent=args.texture_resize_percent,
        overwrite=args.overwrite,
        add_free_joint=args.add_free_joint,
    )

    # Process each directory
    for obj_dir in obj_dirs:
        cprint(f"\nProcessing directory: {obj_dir}", "cyan")
        
        # Get all OBJ files in this directory
        obj_files = list(obj_dir.glob("*.obj"))
        
        # Filter files if regex is provided
        if args.obj_filter:
            obj_files = [
                f for f in obj_files 
                if re.search(args.obj_filter, f.name) is not None
            ]
        
        if not obj_files:
            cprint(f"  No matching OBJ files found in {obj_dir}", "yellow")
            continue

        # Process each OBJ file in this directory
        for obj_file in obj_files:
            cprint(f"  Processing {obj_file.name}", "yellow")
            try:
                process_obj_inplace(obj_file, obj2mjcf_args, visual_only=args.visual_only)
                cprint(f"  ✓ Successfully processed {obj_file.name}", "green")
            except Exception as e:
                cprint(f"  ✗ Error processing {obj_file.name}: {e}", "red")
                logging.exception(f"Error processing {obj_file}")

    cprint(f"\nBatch processing completed!", "green")


if __name__ == "__main__":
    main()
