#!/usr/bin/env python3
"""
Automated motion data preparation pipeline for rollout generation.

This script handles two data sources:
1. Protomotion retargeting data: SMPLH -> Keypoints -> Pyroki Retargeting -> Proto -> PT
2. GMR retargeting data: SMPLX (GMR format) -> Proto -> PT

Usage:
    python prepare_rollout_data.py unified \\
        --smplh-root /data/amass_smplh \\
        --gmr-root /mnt/Exp_HDD/dataset/gmr_data \\
        --motion-list test_scripts/filtered_feat_p.txt \\
        --output-dir /data/rollout_output \\
        --humanoid-type g1 \\
        --proto-python /path/to/proto/python \\
        --pyroki-python /path/to/pyroki/python \\
        --force-remake
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import yaml
from tqdm import tqdm
import csv
import time


def load_motion_list(txt_path: Path) -> List[str]:
    """Load motion identifiers from txt file."""
    entries = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(line)
    return entries

def get_project_root() -> Path:
    return Path(__file__).resolve().parent

def run_command(cmd: List[str], cwd: Path, description: str = "Running command") -> int:
    """Run a shell command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CWD: {cwd}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(get_project_root()) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    return result.returncode


def run_shell_command(cmd: List[str], cwd: Path, description: str = "Running command") -> int:
    """Run a shell command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"CWD: {cwd}")

    result = subprocess.run(cmd, cwd=cwd, shell=False, check=False)
    return result.returncode


project_root =  Path(__file__).resolve().parent

def smpl_to_smplx(key_name):
    key_name = key_name.replace("SSM_synced", "SSM")
    key_name = key_name.replace("MPI_HDM05", "HDM05")
    key_name = key_name.replace("MPI_mosh", "MoSh")
    key_name = key_name.replace("MPI_Limits", "PosePrior")
    key_name = key_name.replace("TCD_handMocap", "TCDHands")
    key_name = key_name.replace("Transitions_mocap", "Transitions")
    key_name = key_name.replace("DFaust_67", "DFaust")
    key_name = key_name.replace("BioMotionLab_NTroje", "BMLrub")
    key_name = key_name.replace("EyesJapanDataset","Eyes_Japan_Dataset")

    return key_name

def step1_copy_and_convert_smplx(args, motion_list: List[str]) -> int:
    """Copy SMPLX files to temp dir and convert to proto format."""
    print("\n" + "="*60)
    print("STEP 1: Copy SMPLX files and convert to ProtoMotions format")
    print("="*60)

    # Create temp directory structure
    temp_smplx_dir = Path(args.output_path) / 'temp_smplx'

    temp_smplx_dir.mkdir(parents=True, exist_ok=True)

    # Copy SMPLX files preserving directory structure
    left_list = []
    copied = 0
    for motion_path in tqdm(motion_list, desc="Copying SMPLX files"):
        motion_path = smpl_to_smplx(motion_path).replace('_poses.', '_stageii.').replace(".npy", ".pkl").replace(".pkl", ".npz").replace(' ', '_')
        src = Path(args.smpl_path) / motion_path
        dst = temp_smplx_dir / motion_path

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.exists():
            if dst.exists():
                continue

            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Warning: File not found: {src}")
            raise RuntimeError
            left_list.append(motion_path)

    print(f"Copied {copied}/{len(motion_list)} SMPLX files")

    # Run conversion - convert_amass_to_proto outputs alongside input files
    script = project_root / 'data' / "scripts" / "convert_amass_to_proto.py"

    cmd = [
        sys.executable, str(script),
        str(temp_smplx_dir),  # positional: AMASS_ROOT_DIR
        "--humanoid-type", "smpl",
        "--output-fps", "30",
    ]
    return run_command(cmd, project_root, "Converting SMPLX to ProtoMotions format")

def step2_package_smplx_for_retargeting(args, motion_list: List[str]) -> Tuple[int, Path]:
    """Package proto files into .pt for retargeting."""
    print("\n" + "="*60)
    print("STEP 2: Package proto for retargeting")
    print("="*60)
    num = 0
    times = []
    file_path = Path(args.output_path) / f'timesta.csv'
    file_path.parent.mkdir(parents=True, exist_ok=True)
    for motion_path_o in tqdm(motion_list, desc="Generating retargeting YAML"):
        try:
            output_path = Path(args.output_path) / f'{num}'
            # Generate YAML pointing to proto files
            # Note: convert_amass_to_proto creates .motion files in the same directory as .npz
            temp_yaml = output_path / "_temp_smplx_retarget.yaml"
            temp_smplx_dir = Path(args.output_path) / 'temp_smplx'
            motions = []
        # for motion_path in tqdm(motion_list, desc="Generating retargeting YAML"):
            motion_path = smpl_to_smplx(motion_path_o).replace('_poses.', '_stageii.').replace(".npy", ".pkl").replace(".pkl", ".npz").replace(' ', '_')
            # Proto file path after conversion (same dir as npz)
            proto_path = temp_smplx_dir / motion_path
            proto_path = proto_path.parent / proto_path.name.replace(".npz", ".motion")
            proto_path = proto_path.parent / proto_path.name.replace("-", "_").replace(" ", "_").replace("(", "_").replace(")", "_")

            if proto_path.exists():
                motions.append({
                    "file": str(proto_path),
                    "key": motion_path
                })
            else:
                print(f'Warning: Not Found the proto:')
                print(proto_path)

            print(f"Found {len(motions)}/{len(motion_list)} proto files")

            if not motions:
                print("Error: No proto files found!")
                temp_yaml.unlink(missing_ok=True)
                return

            config_data = {"motions": motions}
            with open(temp_yaml, 'w') as f:
                yaml.dump(config_data, f)

            script = project_root / "protomotions" / "components" / "motion_lib.py"

            output_file = output_path / "smplx_for_retargeting.pt"

            cmd = [
                sys.executable, str(script),
                "--motion-path", str(temp_yaml),
                "--output-file", str(output_file),
                "--device", 'cpu',
            ]

            result = run_command(cmd, project_root, "Packaging for retargeting")

            temp_yaml.unlink(missing_ok=True)

            """Run the retarget_amass_to_robot.sh shell script."""
            print("\n" + "="*60)
            print("STEP 3: Run Protomotion Retargeting (via shell script)")
            print("="*60)

            if not args.proto_python:
                print("Error: --proto-python argument is required for Protomotion retargeting")
                return 1

            if not args.pyroki_python:
                print("Error: --pyroki-python argument is required for Protomotion retargeting")
                return 1

            # Validate Python interpreters
            proto_py = Path(args.proto_python)
            pyroki_py = Path(args.pyroki_python)

            if not proto_py.exists():
                print(f"Error: Proto Python not found: {proto_py}")
                return 1

            if not pyroki_py.exists():
                print(f"Error: PyRoki Python not found: {pyroki_py}")
                return 1

            # Verify they are valid executables
            if not os.access(str(proto_py), os.X_OK):
                print(f"Error: Proto Python is not executable: {proto_py}")
                return 1

            if not os.access(str(pyroki_py), os.X_OK):
                print(f"Error: PyRoki Python is not executable: {pyroki_py}")
                return 1

            print(f"Proto Python: {proto_py}")
            print(f"PyRoki Python: {pyroki_py}")

            script = project_root / "scripts" / "fps_retarget.sh"


            if not os.access(str(script), os.X_OK):
                print(f"Making shell script executable: {script}")
                os.chmod(str(script), 0o755)

            # The shell script takes: proto_python pyroki_python amass_pt_file output_dir robot_type
            cmd = [
                str(script),
                str(proto_py),
                str(pyroki_py),
                str(output_file),
                str(output_path),
                'g1',
            ]
            s_time = time.time()
            run_shell_command(cmd, project_root, "Running retargeting")
            e_time = time.time()

            proc_time = e_time - s_time
            info = [motion_path_o, proc_time]
            times.append(info)

            num += 1
        except Exception as e:
            print(e)

        finally:
            file_path = str(Path(args.output_path) / f'timesta.csv')
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)

                header = ['Motion', 'Processing Time']
                writer.writerow(header)

                for row in times:
                    writer.writerow(row)


            print(f'Finished, file saved to {file_path}')

def step5_package_final(args, motion_list: List[str]) -> Tuple[int, Path, Path]:
        """Package both to .pt files with unified ordering."""
        print("\n" + "="*60)
        print("STEP 5: Package to .pt files with unified ordering")
        print("="*60)

        # Generate Protomotion YAML (from retargeted output)
        temp_proto_yaml = Path(args.output_path) / "_temp_proto_final.yaml"

        proto_motions = []
        for motion_path in tqdm(motion_list, desc="Generating final Proto YAML"):
            motion_path = smpl_to_smplx(motion_path).replace('_poses.', '_stageii.').replace(".npy", ".pkl").replace(".pkl", ".npz").replace(' ', '_')
            p = Path(motion_path)
            motion_key = Path(*p.parts[-2:]) 
            motion_key = str(motion_key).replace(' ', '_').replace('.npz', '_keypoints_retargeted').replace('-', '_').replace('/', '_' )
            proto_file = Path(args.output_path) / f"retargeted_g1_proto" / f"{motion_key}.motion"
            print(proto_file)
            print(proto_file.exists())
            if proto_file.exists():
                proto_motions.append({
                    "file": str(proto_file),
                    "key": motion_path
                })
        with open(temp_proto_yaml, 'w') as f:
            yaml.dump({"motions": proto_motions}, f)


        # Package both
        script = project_root / "protomotions" / "components" / "motion_lib.py"

        proto_output = Path(args.output_path) / f"protomotion_g1.pt"

        # Package Protomotion
        cmd = [
            sys.executable, str(script),
            "--motion-path", str(temp_proto_yaml),
            "--output-file", str(proto_output),
            "--device", 'cpu',
        ]

        result1 = run_command(cmd, project_root, "Packaging final Protomotions .pt")
        temp_proto_yaml.unlink(missing_ok=True)

def create_parser():
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the results",
    )
    parser.add_argument(
        "--smpl-path",
        type=str,
        default='/mnt/Exp_HDD/dataset/amass_smplh_g_unzip',
    )
    parser.add_argument(
        "--motion-list",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--proto-python",
        type=str,
        default='/usr/bin/python3.10',
    )
    parser.add_argument(
        "--pyroki-python",
        type=str,
        default='/opt/venvs/py310_env/bin/python',
    )
    parser.add_argument(
        "--no-clip",
        default=False,
        action='store_true'
    )
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    motion_list = load_motion_list(args.motion_list)
    step1_copy_and_convert_smplx(args, motion_list)
    _, file = step2_package_smplx_for_retargeting(args, motion_list)
    # step3_run_retarget_shell(args, file)
    # step5_package_final(args, motion_list)


if __name__ == "__main__":
    main()
